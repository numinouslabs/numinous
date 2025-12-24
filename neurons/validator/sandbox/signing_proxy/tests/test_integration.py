import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

from neurons.validator.sandbox.signing_proxy.host import SigningProxyServer


class TestSigningProxyIntegration:
    @pytest.fixture
    def mock_wallet(self):
        mock_wallet = MagicMock()
        mock_hotkey = MagicMock()
        mock_hotkey.ss58_address = "5CCgXySACBvSJ9mz76FwhksstiGSaNuNr5fMqCYJ8efioFaE"
        mock_hotkey.public_key.hex.return_value = "0x1234567890abcdef"
        mock_hotkey.sign.return_value = b"mock_signature_bytes"
        mock_wallet.hotkey = mock_hotkey
        return mock_wallet

    @pytest.fixture
    def mock_gateway(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        received_requests = []

        class MockGatewayHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                received_requests.append(
                    {
                        "method": "GET",
                        "path": self.path,
                        "headers": dict(self.headers),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"result": "ok"}')

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b""
                received_requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"result": "ok"}')

        server = HTTPServer(("127.0.0.1", 9999), MockGatewayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        yield received_requests

        server.shutdown()

    @pytest.fixture
    def signing_proxy(self, mock_wallet, mock_gateway):
        proxy = SigningProxyServer(
            wallet=mock_wallet, proxy_upstream_url="http://127.0.0.1:9999", port=8889
        )

        thread = threading.Thread(target=proxy.start, daemon=True)
        thread.start()
        time.sleep(0.2)

        yield proxy

        if proxy.server:
            proxy.server.shutdown()

    def test_desearch_request_adds_validator_headers(self, signing_proxy, mock_gateway):
        response = requests.post(
            "http://127.0.0.1:8889/api/gateway/desearch/ai/search",
            json={"query": "test", "run_id": "test-run"},
            timeout=5,
        )

        assert response.status_code == 200
        assert len(mock_gateway) == 1

        forwarded_request = mock_gateway[0]
        headers = forwarded_request["headers"]

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "Validator" in headers
        assert "Validator-Public-Key" in headers
        assert "Validator-Version" in headers

        assert "X-Validator-Hotkey" in headers
        assert "X-Validator-Signature" in headers
        assert "X-Validator-Timestamp" in headers

        assert headers["X-Validator-Signature"] == b"mock_signature_bytes".hex()

    def test_chutes_request_no_desearch_headers(self, signing_proxy, mock_gateway):
        response = requests.post(
            "http://127.0.0.1:8889/api/gateway/chutes/chat/completions",
            json={"model": "test", "run_id": "test-run"},
            timeout=5,
        )

        assert response.status_code == 200
        assert len(mock_gateway) == 1

        forwarded_request = mock_gateway[0]
        headers = forwarded_request["headers"]

        assert "Authorization" in headers
        assert "Validator" in headers
        assert "Validator-Public-Key" in headers
        assert "Validator-Version" in headers

        assert "X-Validator-Hotkey" not in headers
        assert "X-Validator-Signature" not in headers
        assert "X-Validator-Timestamp" not in headers

    def test_desearch_get_request_adds_headers(self, signing_proxy, mock_gateway):
        response = requests.get(
            "http://127.0.0.1:8889/api/gateway/desearch/web/search?query=bitcoin",
            timeout=5,
        )

        assert response.status_code == 200
        assert len(mock_gateway) == 1

        forwarded_request = mock_gateway[0]
        headers = forwarded_request["headers"]

        assert "X-Validator-Hotkey" in headers
        assert "X-Validator-Signature" in headers
        assert "X-Validator-Timestamp" in headers

    def test_multiple_requests_different_timestamps(self, signing_proxy, mock_gateway):
        response1 = requests.get(
            "http://127.0.0.1:8889/api/gateway/desearch/twitter?query=crypto", timeout=5
        )
        time.sleep(0.01)
        response2 = requests.get(
            "http://127.0.0.1:8889/api/gateway/desearch/twitter?query=bitcoin", timeout=5
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert len(mock_gateway) == 2

        timestamp1 = mock_gateway[0]["headers"]["X-Validator-Timestamp"]
        timestamp2 = mock_gateway[1]["headers"]["X-Validator-Timestamp"]

        assert timestamp1 != timestamp2
