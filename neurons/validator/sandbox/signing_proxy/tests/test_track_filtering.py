import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neurons.validator.sandbox.signing_proxy.async_host import AsyncValidatorSigningProxy


@pytest.fixture
def registry_dir():
    with tempfile.TemporaryDirectory(prefix="test_registry_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_wallet():
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5CCgXySACBvSJ9mz76FwhksstiGSaNuNr5fMqCYJ8efioFaE"
    wallet.hotkey.public_key.hex.return_value = "0x1234567890abcdef"
    wallet.hotkey.sign.return_value = b"mock_signature"
    return wallet


class TestCheckTrackAccess:
    def _make_proxy(self, registry_dir: Path) -> AsyncValidatorSigningProxy:
        import os

        os.environ["RUN_REGISTRY_DIR"] = str(registry_dir)
        try:
            proxy = AsyncValidatorSigningProxy.__new__(AsyncValidatorSigningProxy)
            proxy.run_registry_dir = registry_dir
            proxy.track_cache = {}
            return proxy
        finally:
            os.environ.pop("RUN_REGISTRY_DIR", None)

    def test_signal_blocked_on_disallowed_endpoint(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses", body)

        assert result is not None
        assert result.status == 403

    def test_signal_blocked_on_desearch(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-2"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/desearch/ai/search", body)

        assert result is not None
        assert result.status == 403

    def test_signal_blocked_on_perplexity(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-3"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/perplexity/chat/completions", body)

        assert result is not None
        assert result.status == 403

    def test_signal_allowed_on_chutes(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-chutes"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/chutes/chat/completions", body)

        assert result is None

    def test_signal_allowed_on_numinous_indicia(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-indicia"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/numinous-indicia/x-osint", body)

        assert result is None

    def test_signal_allowed_on_openai_inference_only(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-openai-inference"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses/inference", body)

        assert result is None

    def test_signal_blocked_on_openai_responses(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-signal-openai-responses"
        (registry_dir / run_id).write_text("SIGNAL")

        body = json.dumps({"run_id": run_id}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses", body)

        assert result is not None
        assert result.status == 403

    def test_main_allowed_on_everything(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        run_id = "test-run-main"
        (registry_dir / run_id).write_text("MAIN")

        body = json.dumps({"run_id": run_id}).encode()

        for endpoint in [
            "/api/gateway/openai/responses",
            "/api/gateway/desearch/ai/search",
            "/api/gateway/chutes/chat/completions",
            "/api/gateway/perplexity/chat/completions",
            "/api/gateway/vericore/calculate-rating",
            "/api/gateway/numinous-indicia/x-osint",
        ]:
            result = proxy._check_track_access(endpoint, body)
            assert result is None, f"MAIN should be allowed on {endpoint}"

    def test_no_body_allowed(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        result = proxy._check_track_access("/api/gateway/openai/responses", b"")
        assert result is None

    def test_no_run_id_in_body_allowed(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        body = json.dumps({"model": "gpt-4"}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses", body)
        assert result is None

    def test_unknown_run_id_allowed(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        body = json.dumps({"run_id": "nonexistent-run"}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses", body)
        assert result is None

    def test_invalid_json_body_allowed(self, registry_dir: Path):
        proxy = self._make_proxy(registry_dir)
        result = proxy._check_track_access("/api/gateway/openai/responses", b"not json")
        assert result is None

    def test_no_registry_dir_allows_everything(self):
        proxy = AsyncValidatorSigningProxy.__new__(AsyncValidatorSigningProxy)
        proxy.run_registry_dir = None
        proxy.track_cache = {}

        body = json.dumps({"run_id": "some-run"}).encode()
        result = proxy._check_track_access("/api/gateway/openai/responses", body)
        assert result is None


class TestTrackCache:
    def test_caches_track_after_first_read(self, registry_dir: Path):
        proxy = AsyncValidatorSigningProxy.__new__(AsyncValidatorSigningProxy)
        proxy.run_registry_dir = registry_dir
        proxy.track_cache = {}

        run_id = "cached-run"
        (registry_dir / run_id).write_text("SIGNAL")

        assert proxy._get_track(run_id) == "SIGNAL"
        assert run_id in proxy.track_cache

        # Delete file — should still return cached value
        (registry_dir / run_id).unlink()
        assert proxy._get_track(run_id) == "SIGNAL"

    def test_returns_none_for_missing_run(self, registry_dir: Path):
        proxy = AsyncValidatorSigningProxy.__new__(AsyncValidatorSigningProxy)
        proxy.run_registry_dir = registry_dir
        proxy.track_cache = {}

        assert proxy._get_track("nonexistent") is None
        assert "nonexistent" not in proxy.track_cache
