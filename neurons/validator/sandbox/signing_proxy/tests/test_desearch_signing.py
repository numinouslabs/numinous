import base64
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from neurons.validator.sandbox.signing_proxy.host import ValidatorSigningProxyHandler


class TestDesearchSigning:
    @pytest.fixture
    def handler(self, mock_wallet):
        ValidatorSigningProxyHandler.wallet = mock_wallet
        ValidatorSigningProxyHandler.proxy_upstream_url = "http://gateway:8000"

        with patch.object(
            ValidatorSigningProxyHandler, "__init__", lambda x, *args, **kwargs: None
        ):
            handler = ValidatorSigningProxyHandler(None, None, None)
            handler.rfile = BytesIO(b'{"query": "test"}')
            handler.wfile = BytesIO()
            handler.headers = MagicMock()
            handler.headers.get.return_value = "17"
            handler.headers.__iter__ = lambda self: iter([])
            handler.command = "POST"
            handler.send_response = MagicMock()
            handler.send_header = MagicMock()
            handler.end_headers = MagicMock()
            handler.send_error = MagicMock()
        return handler

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_desearch_path_has_both_standard_and_desearch_headers(
        self, mock_request, handler, mock_wallet
    ):
        handler.path = "/api/gateway/desearch/twitter"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        handler._forward_request("POST")

        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        headers = call_kwargs["headers"]

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "Validator" in headers
        assert headers["Validator"] == mock_wallet.hotkey.ss58_address
        assert "Validator-Public-Key" in headers
        assert "Validator-Version" in headers

        assert "X-Validator-Hotkey" in headers
        assert headers["X-Validator-Hotkey"] == mock_wallet.hotkey.ss58_address
        assert "X-Validator-Signature" in headers
        assert "X-Validator-Timestamp" in headers

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_non_desearch_path_only_has_standard_headers(self, mock_request, handler, mock_wallet):
        handler.path = "/api/gateway/chutes/chat/completions"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        handler._forward_request("POST")

        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        headers = call_kwargs["headers"]

        assert "Authorization" in headers
        assert "Validator" in headers
        assert "Validator-Public-Key" in headers
        assert "Validator-Version" in headers

        assert "X-Validator-Hotkey" not in headers
        assert "X-Validator-Signature" not in headers
        assert "X-Validator-Timestamp" not in headers

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_timestamp_signature_uses_hex_encoding(self, mock_request, handler, mock_wallet):
        handler.path = "/api/gateway/desearch/ai/search"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        mock_wallet.hotkey.sign.return_value = b"\x01\x02\x03\x04"

        handler._forward_request("POST")

        call_kwargs = mock_request.call_args[1]
        headers = call_kwargs["headers"]

        assert headers["X-Validator-Signature"] == "01020304"

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_timestamp_format_is_iso8601_utc(self, mock_request, handler):
        handler.path = "/api/gateway/desearch/web/crawl"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        handler._forward_request("GET")

        call_kwargs = mock_request.call_args[1]
        headers = call_kwargs["headers"]
        timestamp = headers["X-Validator-Timestamp"]

        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo == timezone.utc

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_desearch_signature_signs_timestamp_not_body(self, mock_request, handler, mock_wallet):
        handler.path = "/api/gateway/desearch/twitter"
        body = b'{"query": "bitcoin"}'
        handler.rfile = BytesIO(body)
        handler.headers.get.return_value = str(len(body))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        handler._forward_request("POST")

        sign_calls = mock_wallet.hotkey.sign.call_args_list
        assert len(sign_calls) == 2

        body_signature_call = sign_calls[0][0][0]
        timestamp_signature_call = sign_calls[1][0][0]

        assert body_signature_call == body
        assert timestamp_signature_call != body_signature_call
        datetime.fromisoformat(timestamp_signature_call.decode("utf-8"))

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_multiple_desearch_paths_all_get_headers(self, mock_request, handler):
        paths = [
            "/api/gateway/desearch/twitter",
            "/api/gateway/desearch/ai/search",
            "/api/gateway/desearch/web/crawl",
            "/api/gateway/desearch/ai/search/links/web",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        for path in paths:
            handler.path = path
            handler.rfile = BytesIO(b"test")
            handler._forward_request("GET")

            call_kwargs = mock_request.call_args[1]
            headers = call_kwargs["headers"]

            assert "X-Validator-Hotkey" in headers
            assert "X-Validator-Signature" in headers
            assert "X-Validator-Timestamp" in headers

    @patch("neurons.validator.sandbox.signing_proxy.host.requests.request")
    def test_authorization_header_still_uses_base64(self, mock_request, handler, mock_wallet):
        handler.path = "/api/gateway/desearch/twitter"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b"test response"
        mock_request.return_value = mock_response

        mock_wallet.hotkey.sign.side_effect = [b"body_signature", b"timestamp_signature"]

        handler._forward_request("POST")

        call_kwargs = mock_request.call_args[1]
        headers = call_kwargs["headers"]

        expected_auth = f"Bearer {base64.b64encode(b'body_signature').decode('utf-8')}"
        assert headers["Authorization"] == expected_auth
