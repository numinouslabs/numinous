import asyncio
import base64
import os
from datetime import datetime, timezone

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
        self.client_lock = asyncio.Lock()

    async def handle_request(self, request: web.Request) -> web.Response:
        request_start = asyncio.get_event_loop().time()

        try:
            body = await request.read()

            try:
                signature = self.wallet.hotkey.sign(body)
                signature_b64 = base64.b64encode(signature).decode("utf-8")
            except Exception as e:
                error_msg = f"Failed to sign request: {type(e).__name__}: {str(e)}"
                print(
                    f"[SIGNING-PROXY] ERROR: {request.method} {request.path}: {error_msg}",
                    flush=True,
                )
                return web.Response(status=500, text=error_msg)

            forward_headers = dict(request.headers)
            forward_headers["Authorization"] = f"Bearer {signature_b64}"
            forward_headers["Validator"] = self.wallet.hotkey.ss58_address
            forward_headers["Validator-Public-Key"] = self.wallet.hotkey.public_key.hex()
            forward_headers["Validator-Version"] = os.environ.get("VALIDATOR_VERSION", "unknown")

            if request.path.startswith("/api/gateway/desearch/"):
                timestamp = datetime.now(timezone.utc).isoformat()
                timestamp_signature = self.wallet.hotkey.sign(timestamp.encode("utf-8"))
                forward_headers["X-Validator-Hotkey"] = self.wallet.hotkey.ss58_address
                forward_headers["X-Validator-Signature"] = timestamp_signature.hex()
                forward_headers["X-Validator-Timestamp"] = timestamp

            if body:
                forward_headers["Content-Length"] = str(len(body))

            for header in ["Host", "Connection", "Keep-Alive", "Transfer-Encoding"]:
                forward_headers.pop(header, None)

            target_url = f"{self.proxy_upstream_url}{request.path_qs}"

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = await self.client.request(
                        method=request.method,
                        url=target_url,
                        headers=forward_headers,
                        content=body if body else None,
                        timeout=120.0,
                        follow_redirects=False,
                    )

                    elapsed = asyncio.get_event_loop().time() - request_start

                    if response.status_code >= 400:
                        try:
                            error_body = response.json()
                            error_detail = error_body.get("detail", str(error_body))
                        except Exception:
                            error_detail = response.text[:200]

                        print(
                            f"[SIGNING-PROXY] FAILED: {request.method} {request.path} -> "
                            f"{response.status_code} ({elapsed:.2f}s): {error_detail}",
                            flush=True,
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

                except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                    error_type = type(e).__name__
                    error_msg = str(e) if str(e) else repr(e)
                    if attempt < max_retries - 1:
                        print(
                            f"[SIGNING-PROXY] Connection error (attempt {attempt + 1}/{max_retries}): "
                            f"{request.method} {request.path}: {error_type}: {error_msg}",
                            flush=True,
                        )
                        print(
                            "[SIGNING-PROXY] Recreating HTTP client and retrying...",
                            flush=True,
                        )
                        async with self.client_lock:
                            await self.client.aclose()
                            await self._create_new_client()
                        continue
                    else:
                        print(
                            "[SIGNING-PROXY] Connection error (final attempt): "
                            f"{request.method} {request.path}: {error_type}: {error_msg}",
                            flush=True,
                        )
                        raise

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - request_start
            print(
                f"[SIGNING-PROXY] TIMEOUT: {request.method} {request.path} (after {elapsed:.2f}s)",
                flush=True,
            )
            return web.Response(status=504, text="Gateway timeout")
        except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
            elapsed = asyncio.get_event_loop().time() - request_start
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else repr(e)
            print(
                f"[SIGNING-PROXY] ERROR: {request.method} {request.path}: "
                f"{error_type}: {error_msg} (after {elapsed:.2f}s, {max_retries} attempts)",
                flush=True,
            )
            return web.Response(status=502, text=f"Connection error: {error_type}: {error_msg}")
        except httpx.HTTPStatusError as e:
            elapsed = asyncio.get_event_loop().time() - request_start
            print(
                f"[SIGNING-PROXY] ERROR: {request.method} {request.path}: "
                f"HTTP {e.response.status_code} from gateway (after {elapsed:.2f}s)",
                flush=True,
            )
            return web.Response(
                status=e.response.status_code,
                text=f"Gateway returned {e.response.status_code}",
            )
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - request_start
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else repr(e)
            print(
                f"[SIGNING-PROXY] ERROR: {request.method} {request.path}: "
                f"{error_type}: {error_msg} (after {elapsed:.2f}s)",
                flush=True,
            )
            return web.Response(status=500, text=f"Proxy error: {error_type}: {error_msg}")

    async def _create_new_client(self):
        limits = httpx.Limits(
            max_keepalive_connections=50, max_connections=100, keepalive_expiry=30.0
        )
        self.client = httpx.AsyncClient(timeout=120.0, http2=False, limits=limits)
        print(
            "[SIGNING-PROXY] HTTP client created (HTTP/1.1, pool: 50 keepalive / 100 max)",
            flush=True,
        )

    async def start_client(self, app: web.Application):
        await self._create_new_client()

    async def cleanup_client(self, app: web.Application):
        if self.client:
            await self.client.aclose()

    def start(self) -> None:
        self.app.on_startup.append(self.start_client)
        self.app.on_cleanup.append(self.cleanup_client)

        web.run_app(self.app, host="0.0.0.0", port=self.port, print=None)
