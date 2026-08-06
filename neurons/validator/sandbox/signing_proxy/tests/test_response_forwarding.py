import asyncio
import gzip
import json
import zlib
from unittest.mock import MagicMock

import httpx
import pytest
import requests
from aiohttp import web

from neurons.validator.sandbox.signing_proxy.async_host import AsyncValidatorSigningProxy

UPSTREAM_PORT = 8931
PROXY_PORT = 8932

LARGE_PAYLOAD = {"articles": [{"headline": "h" * 120, "index": i} for i in range(20)], "count": 20}
SMALL_PAYLOAD = {"ok": True}


async def _upstream_handler(request: web.Request) -> web.Response:
    kind = request.path.strip("/")

    if kind == "echo-accept-encoding":
        return web.json_response({"accept_encoding": request.headers.get("Accept-Encoding", "")})

    if kind == "negotiated":
        accepted = request.headers.get("Accept-Encoding", "")
        if "zstd" in accepted:
            return web.Response(
                body=b"\x28\xb5\x2f\xfd" + b"not-really-zstd",
                headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            )
        body = gzip.compress(json.dumps(LARGE_PAYLOAD).encode())
        return web.Response(
            body=body,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
        )

    if kind == "gzip-json":
        body = gzip.compress(json.dumps(LARGE_PAYLOAD).encode())
        return web.Response(
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(body)),
            },
        )
    if kind == "deflate-json":
        body = zlib.compress(json.dumps(LARGE_PAYLOAD).encode())
        return web.Response(
            body=body,
            headers={"Content-Type": "application/json", "Content-Encoding": "deflate"},
        )
    if kind == "gzip-error":
        body = gzip.compress(json.dumps({"detail": "boom"}).encode())
        return web.Response(
            body=body,
            status=500,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
        )
    if kind == "plain-large":
        return web.json_response(LARGE_PAYLOAD)
    if kind == "plain-small":
        return web.json_response(SMALL_PAYLOAD)
    if kind == "error-json":
        return web.json_response({"detail": "nope"}, status=400)
    if kind == "empty":
        return web.Response(status=204)
    if kind == "text":
        return web.Response(text="hello world", content_type="text/plain")
    if kind == "unicode":
        return web.json_response({"text": "Iran — Hormuz “fees” ✓"})

    return web.Response(status=404, text="unknown")


def _make_wallet() -> MagicMock:
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5CCgXySACBvSJ9mz76FwhksstiGSaNuNr5fMqCYJ8efioFaE"
    wallet.hotkey.public_key.hex.return_value = "abcdef"
    wallet.hotkey.sign.return_value = b"signature"
    return wallet


@pytest.fixture
async def proxy_url():
    upstream = web.Application()
    upstream.router.add_route("*", "/{path:.*}", _upstream_handler)
    upstream_runner = web.AppRunner(upstream)
    await upstream_runner.setup()
    await web.TCPSite(upstream_runner, "127.0.0.1", UPSTREAM_PORT).start()

    proxy = AsyncValidatorSigningProxy(
        wallet=_make_wallet(),
        proxy_upstream_url=f"http://127.0.0.1:{UPSTREAM_PORT}",
        port=PROXY_PORT,
    )
    proxy.client = httpx.AsyncClient()
    proxy_runner = web.AppRunner(proxy.app)
    await proxy_runner.setup()
    await web.TCPSite(proxy_runner, "127.0.0.1", PROXY_PORT).start()

    yield f"http://127.0.0.1:{PROXY_PORT}"

    await proxy.client.aclose()
    await proxy_runner.cleanup()
    await upstream_runner.cleanup()


async def _post(base_url: str, path: str, client: str):
    def _request():
        library = requests if client == "requests" else httpx
        response = library.post(f"{base_url}/{path}", json={"run_id": "test"}, timeout=10)
        return response.status_code, response.content

    return await asyncio.to_thread(_request)


class TestCompressedResponseForwarding:
    @pytest.mark.parametrize("client", ["requests", "httpx"])
    @pytest.mark.parametrize("path", ["gzip-json", "deflate-json"])
    async def test_compressed_body_is_readable(self, path: str, client: str, proxy_url: str):
        status, body = await _post(proxy_url, path, client)

        assert status == 200
        assert json.loads(body)["count"] == 20

    @pytest.mark.parametrize("client", ["requests", "httpx"])
    async def test_compressed_error_body_is_readable(self, client: str, proxy_url: str):
        status, body = await _post(proxy_url, "gzip-error", client)

        assert status == 500
        assert json.loads(body)["detail"] == "boom"

    @pytest.mark.parametrize("client", ["requests", "httpx"])
    async def test_content_encoding_is_not_forwarded(self, client: str, proxy_url: str):
        response = await asyncio.to_thread(
            lambda: requests.post(f"{proxy_url}/gzip-json", json={"run_id": "t"}, timeout=10)
        )

        assert "content-encoding" not in {k.lower() for k in response.headers}


class TestUpstreamEncodingNegotiation:
    """The proxy strips Content-Encoding, so it must only ask upstream for
    encodings it can actually decode. httpx advertises zstd by default but the
    proxy image ships no zstandard codec, so forwarding the agent's own
    Accept-Encoding would hand agents a body still compressed."""

    @pytest.mark.parametrize("advertised", ["gzip, deflate, zstd", "gzip, deflate, br", "*"])
    async def test_undecodable_encodings_not_forwarded(self, advertised: str, proxy_url: str):
        def _request():
            return requests.post(
                f"{proxy_url}/echo-accept-encoding",
                json={"run_id": "test"},
                headers={"Accept-Encoding": advertised},
                timeout=10,
            ).json()

        forwarded = (await asyncio.to_thread(_request))["accept_encoding"]

        assert "zstd" not in forwarded
        assert "br" not in forwarded
        assert forwarded == "gzip, deflate"

    @pytest.mark.parametrize("advertised", ["gzip, deflate, zstd", "gzip, deflate"])
    async def test_negotiated_response_is_readable(self, advertised: str, proxy_url: str):
        """A CDN honouring zstd would hand back a body the proxy cannot decode."""

        def _request():
            return requests.post(
                f"{proxy_url}/negotiated",
                json={"run_id": "test"},
                headers={"Accept-Encoding": advertised},
                timeout=10,
            )

        response = await asyncio.to_thread(_request)

        assert response.status_code == 200
        assert json.loads(response.content)["count"] == 20


class TestUncompressedResponseForwarding:
    @pytest.mark.parametrize("client", ["requests", "httpx"])
    @pytest.mark.parametrize(
        "path,expected_status",
        [
            ("plain-large", 200),
            ("plain-small", 200),
            ("error-json", 400),
            ("empty", 204),
            ("unicode", 200),
        ],
    )
    async def test_status_preserved(
        self, path: str, expected_status: int, client: str, proxy_url: str
    ):
        status, _ = await _post(proxy_url, path, client)

        assert status == expected_status

    @pytest.mark.parametrize("client", ["requests", "httpx"])
    async def test_large_plain_body_intact(self, client: str, proxy_url: str):
        status, body = await _post(proxy_url, "plain-large", client)

        assert status == 200
        assert len(json.loads(body)["articles"]) == 20

    @pytest.mark.parametrize("client", ["requests", "httpx"])
    async def test_non_json_body_intact(self, client: str, proxy_url: str):
        status, body = await _post(proxy_url, "text", client)

        assert status == 200
        assert body == b"hello world"

    @pytest.mark.parametrize("client", ["requests", "httpx"])
    async def test_unicode_body_intact(self, client: str, proxy_url: str):
        _, body = await _post(proxy_url, "unicode", client)

        assert json.loads(body)["text"] == "Iran — Hormuz “fees” ✓"
