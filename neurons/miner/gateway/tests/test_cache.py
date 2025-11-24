from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from neurons.miner.gateway.app import app
from neurons.miner.gateway.cache import _cache, _cache_lock, generate_request_hash
from neurons.validator.models.chutes import ChutesCompletion


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clear_cache():
    with _cache_lock:
        _cache.clear()
    yield
    with _cache_lock:
        _cache.clear()


class TestCache:
    def test_same_request_generates_same_hash(self):
        endpoint = "test_endpoint"
        payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}

        hash1 = generate_request_hash(endpoint, payload)
        hash2 = generate_request_hash(endpoint, payload)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_different_content_generates_different_hashes(self):
        endpoint = "test_endpoint"
        payload1 = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        payload2 = {"model": "test", "messages": [{"role": "user", "content": "goodbye"}]}

        assert generate_request_hash(endpoint, payload1) != generate_request_hash(
            endpoint, payload2
        )

    def test_parameter_order_normalization(self):
        endpoint = "test_endpoint"
        payload1 = {"model": "test", "temperature": 0.7, "max_tokens": 100}
        payload2 = {"max_tokens": 100, "model": "test", "temperature": 0.7}

        assert generate_request_hash(endpoint, payload1) == generate_request_hash(
            endpoint, payload2
        )

    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_cache_hit_reuses_result(self, mock_client_class, client: TestClient):
        mock_response = ChutesCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="deepseek-ai/DeepSeek-R1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        response1 = client.post("/api/gateway/chutes/chat/completions", json=request_body)
        response2 = client.post("/api/gateway/chutes/chat/completions", json=request_body)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json()
        assert mock_instance.chat_completion.call_count == 1

    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_run_id_excluded_from_cache_key(self, mock_client_class, client: TestClient):
        mock_response = ChutesCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="deepseek-ai/DeepSeek-R1",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock(return_value=mock_response)

        request_body_1 = {
            "run_id": str(uuid4()),
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }
        request_body_2 = {**request_body_1, "run_id": str(uuid4())}

        response1 = client.post("/api/gateway/chutes/chat/completions", json=request_body_1)
        response2 = client.post("/api/gateway/chutes/chat/completions", json=request_body_2)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert mock_instance.chat_completion.call_count == 1

    def test_cache_lock_prevents_race_conditions(self):
        endpoint = "test_endpoint"
        payload = {"test": "data"}

        def access_cache():
            hash_key = generate_request_hash(endpoint, payload)
            with _cache_lock:
                if hash_key not in _cache:
                    _cache[hash_key] = {"result": "test"}
            return _cache.get(hash_key)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_cache) for _ in range(100)]
            results = [f.result() for f in futures]

        assert all(r == {"result": "test"} for r in results)
        assert len(_cache) == 1
