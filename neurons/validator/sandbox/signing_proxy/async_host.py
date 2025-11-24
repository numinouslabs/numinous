import asyncio
import base64
import os

import httpx
from aiohttp import web
from bittensor_wallet import Wallet


class AsyncValidatorSigningProxy:
    def __init__(self, wallet: Wallet, proxy_upstream_url: str, port: int = 8888):
        self.wallet = wallet
        self.proxy_upstream_url = proxy_upstream_url
        self.port = port
        self.app = web.Application()
        self.app.router.add_route("*", "/{path:.*}", self.handle_request)
        self.client = None

    async def handle_request(self, request: web.Request) -> web.Response:
        try:
            body = await request.read()

            try:
                signature = self.wallet.hotkey.sign(body)
                signature_b64 = base64.b64encode(signature).decode("utf-8")
            except Exception as e:
                print(
                    f"[SIGNING-PROXY] ERROR: Failed to sign {request.method} {request.path}: {e}",
                    flush=True,
                )
                return web.Response(status=500, text=f"Failed to sign request: {e}")

            forward_headers = dict(request.headers)
            forward_headers["Authorization"] = f"Bearer {signature_b64}"
            forward_headers["Validator"] = self.wallet.hotkey.ss58_address
            forward_headers["Validator-Public-Key"] = self.wallet.hotkey.public_key.hex()
            forward_headers["Validator-Version"] = os.environ.get("VALIDATOR_VERSION", "unknown")

            if body:
                forward_headers["Content-Length"] = str(len(body))

            for header in ["Host", "Connection", "Keep-Alive", "Transfer-Encoding"]:
                forward_headers.pop(header, None)

            target_url = f"{self.proxy_upstream_url}{request.path_qs}"

            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body if body else None,
                timeout=120.0,
                follow_redirects=False,
            )

            return web.Response(
                status=response.status_code,
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() not in ("transfer-encoding", "connection", "keep-alive")
                },
                body=response.content,
            )

        except asyncio.TimeoutError:
            print(f"[SIGNING-PROXY] TIMEOUT: {request.method} {request.path}", flush=True)
            return web.Response(status=504, text="Gateway timeout")
        except Exception as e:
            print(f"[SIGNING-PROXY] ERROR: {request.method} {request.path}: {e}", flush=True)
            return web.Response(status=500, text=f"Proxy error: {str(e)}")

    async def start_client(self, app: web.Application):
        self.client = httpx.AsyncClient(timeout=120.0, http2=True)

    async def cleanup_client(self, app: web.Application):
        if self.client:
            await self.client.aclose()

    def start(self) -> None:
        self.app.on_startup.append(self.start_client)
        self.app.on_cleanup.append(self.cleanup_client)

        web.run_app(self.app, host="0.0.0.0", port=self.port, print=None)
