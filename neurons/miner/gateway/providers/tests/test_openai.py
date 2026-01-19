import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.openai import OpenAIClient
from neurons.validator.models.openai import OpenAIResponse


class TestOpenAIClient:
    @pytest.fixture
    def client(self):
        return OpenAIClient(api_key="test_api_key")

    async def test_create_response_success(self, client: OpenAIClient):
        mock_response = {
            "id": "resp-123",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5-mini",
            "output": [
                {
                    "id": "msg-123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Test response",
                            "logprobs": [],
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.openai.com/v1/responses",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                model="gpt-5-mini",
                input=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
            )

            assert isinstance(result, OpenAIResponse)
            assert result.id == "resp-123"
            assert result.model == "gpt-5-mini"
            assert len(result.output) == 1
            assert result.output[0].content[0].text == "Test response"

    async def test_create_response_with_optional_params(self, client: OpenAIClient):
        mock_response = {
            "id": "resp-456",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5",
            "output": [
                {
                    "id": "msg-456",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Response with tools"},
                    ],
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.openai.com/v1/responses",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                model="gpt-5",
                input=[{"role": "user", "content": "Test"}],
                temperature=0.5,
                max_output_tokens=100,
                tools=[{"type": "web_search"}],
            )

            assert isinstance(result, OpenAIResponse)
            assert result.id == "resp-456"
            assert result.model == "gpt-5"

    async def test_create_response_error_raised(self, client: OpenAIClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.openai.com/v1/responses",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.create_response(
                    model="gpt-5-mini", input=[{"role": "user", "content": "Test"}]
                )

            assert exc.value.status == 500

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIClient(api_key="")

        with pytest.raises(ValueError, match="OpenAI API key is not set"):
            OpenAIClient(api_key=None)
