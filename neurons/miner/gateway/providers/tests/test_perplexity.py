import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.perplexity import PerplexityClient
from neurons.validator.models.perplexity import PerplexityCompletion


class TestPerplexityClient:
    @pytest.fixture
    def client(self):
        return PerplexityClient(api_key="test_api_key")

    async def test_chat_completion_success(self, client: PerplexityClient):
        mock_response = {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
                "search_context_size": "medium",
            },
            "citations": ["https://example.com"],
            "search_results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "snippet": "Example snippet",
                    "source": "web",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="sonar",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.2,
            )

            assert isinstance(result, PerplexityCompletion)
            assert result.id == "cmpl-123"
            assert result.model == "sonar"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Test response"
            assert result.citations == ["https://example.com"]
            assert len(result.search_results) == 1
            assert result.search_results[0].title == "Example"

    async def test_chat_completion_with_optional_params(self, client: PerplexityClient):
        mock_response = {
            "id": "cmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "sonar-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response with filters"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 80,
                "total_tokens": 110,
                "search_context_size": "high",
            },
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="sonar-pro",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.5,
                max_tokens=200,
                search_recency_filter="day",
            )

            assert isinstance(result, PerplexityCompletion)
            assert result.id == "cmpl-456"
            assert result.model == "sonar-pro"
            assert result.usage.search_context_size == "high"

    async def test_chat_completion_minimal_response(self, client: PerplexityClient):
        mock_response = {
            "id": "cmpl-789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Minimal response"},
                    "finish_reason": "stop",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="sonar",
                messages=[{"role": "user", "content": "Test"}],
            )

            assert isinstance(result, PerplexityCompletion)
            assert result.id == "cmpl-789"
            assert result.usage is None
            assert result.citations is None
            assert result.search_results is None

    async def test_chat_completion_error_raised(self, client: PerplexityClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="sonar", messages=[{"role": "user", "content": "Test"}]
                )

            assert exc.value.status == 500

    async def test_chat_completion_authentication_error(self, client: PerplexityClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=401,
                body=b"Unauthorized",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="sonar", messages=[{"role": "user", "content": "Test"}]
                )

            assert exc.value.status == 401

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="Perplexity API key is not set"):
            PerplexityClient(api_key="")

        with pytest.raises(ValueError, match="Perplexity API key is not set"):
            PerplexityClient(api_key=None)

    async def test_chat_completion_with_kwargs(self, client: PerplexityClient):
        mock_response = {
            "id": "cmpl-999",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response with extra params"},
                    "finish_reason": "stop",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.perplexity.ai/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="sonar",
                messages=[{"role": "user", "content": "Test"}],
                top_p=0.9,
                frequency_penalty=0.5,
            )

            assert isinstance(result, PerplexityCompletion)
            assert result.id == "cmpl-999"
