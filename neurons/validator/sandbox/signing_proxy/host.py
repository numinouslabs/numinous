import base64
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from bittensor_wallet import Wallet


class ValidatorSigningProxyHandler(BaseHTTPRequestHandler):
    wallet: Wallet = None
    proxy_upstream_url: str = None

    def log_message(self, format, *args):
        if "error" in format.lower() or self.command in ("POST", "PUT", "PATCH", "DELETE"):
            print(f"[SIGNING-PROXY] {format % args}", flush=True)

    def _forward_request(self, method: str):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""

            if not self.wallet:
                print(
                    f"[SIGNING-PROXY] ERROR: Wallet not configured for {method} {self.path}",
                    flush=True,
                )
                self.send_error(500, "Wallet not configured")
                return

            try:
                signature = self.wallet.hotkey.sign(body)
                signature_b64 = base64.b64encode(signature).decode("utf-8")
            except Exception as e:
                print(
                    f"[SIGNING-PROXY] ERROR: Failed to sign {method} {self.path}: {e}", flush=True
                )
                self.send_error(500, f"Failed to sign request: {e}")
                return

            forward_headers = dict(self.headers)
            forward_headers["Authorization"] = f"Bearer {signature_b64}"
            forward_headers["Validator"] = self.wallet.hotkey.ss58_address
            forward_headers["Validator-Public-Key"] = self.wallet.hotkey.public_key.hex()
            forward_headers["Validator-Version"] = os.environ.get("VALIDATOR_VERSION", "unknown")

            if self.path.startswith("/api/gateway/desearch/"):
                timestamp = datetime.now(timezone.utc).isoformat()
                timestamp_signature = self.wallet.hotkey.sign(timestamp.encode("utf-8"))
                forward_headers["X-Validator-Hotkey"] = self.wallet.hotkey.ss58_address
                forward_headers["X-Validator-Signature"] = timestamp_signature.hex()
                forward_headers["X-Validator-Timestamp"] = timestamp

            if body:
                forward_headers["Content-Length"] = str(len(body))

            forward_headers.pop("Host", None)

            target_url = f"{self.proxy_upstream_url}{self.path}"
            response = requests.request(
                method=method,
                url=target_url,
                headers=forward_headers,
                data=body if body else None,
                timeout=120,
                allow_redirects=False,
            )

            self.send_response(response.status_code)
            for key, value in response.headers.items():
                key_lower = key.lower()
                if key_lower not in (
                    "transfer-encoding",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                ):
                    self.send_header(key, value)
            self.end_headers()

            if response.content:
                self.wfile.write(response.content)
                self.wfile.flush()

        except Exception as e:
            print(f"[SIGNING-PROXY] ERROR: Proxy error on {method} {self.path}: {e}", flush=True)
            self.send_error(500, f"Proxy error: {str(e)}")

    def do_GET(self):
        self._forward_request("GET")

    def do_POST(self):
        self._forward_request("POST")

    def do_PUT(self):
        self._forward_request("PUT")

    def do_DELETE(self):
        self._forward_request("DELETE")

    def do_PATCH(self):
        self._forward_request("PATCH")


class SigningProxyServer:
    def __init__(self, wallet: Wallet, proxy_upstream_url: str, port: int = 8888):
        self.wallet = wallet
        self.proxy_upstream_url = proxy_upstream_url
        self.port = port
        self.server = None

        ValidatorSigningProxyHandler.wallet = wallet
        ValidatorSigningProxyHandler.proxy_upstream_url = proxy_upstream_url

    def start(self) -> None:
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), ValidatorSigningProxyHandler)
        self.server.serve_forever()
