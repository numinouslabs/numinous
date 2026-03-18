import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.azure_openai import AzureOpenAIClient
from neurons.validator.models.azure_openai import AzureOpenAIResponse


class TestAzureOpenAIClient:
    @pytest.fixture
    def client(self):
        return AzureOpenAIClient(
            api_key="test_api_key",
            resource_name="test-resource",
            api_version="2024-02-15-preview",
        )

    async def test_create_response_success(self, client: AzureOpenAIClient):
        mock_response = {
            "id": "resp-123",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5-mini-deployment",
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
                "https://test-resource.openai.azure.com/openai/deployments/gpt-5-mini-deployment/responses?api-version=2024-02-15-preview",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                deployment="gpt-5-mini-deployment",
                input=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
            )

            assert isinstance(result, AzureOpenAIResponse)
            assert result.id == "resp-123"
            assert result.model == "gpt-5-mini-deployment"
            assert len(result.output) == 1
            assert result.output[0].content[0].text == "Test response"

    async def test_create_response_with_optional_params(self, client: AzureOpenAIClient):
        mock_response = {
            "id": "resp-456",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5-deployment",
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
                "https://test-resource.openai.azure.com/openai/deployments/gpt-5-deployment/responses?api-version=2024-02-15-preview",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                deployment="gpt-5-deployment",
                input=[{"role": "user", "content": "Test"}],
                temperature=0.5,
                max_output_tokens=100,
                tools=[{"type": "web_search"}],
            )

            assert isinstance(result, AzureOpenAIResponse)
            assert result.id == "resp-456"
            assert result.model == "gpt-5-deployment"

    async def test_create_response_error_raised(self, client: AzureOpenAIClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://test-resource.openai.azure.com/openai/deployments/gpt-5-mini-deployment/responses?api-version=2024-02-15-preview",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.create_response(
                    deployment="gpt-5-mini-deployment",
                    input=[{"role": "user", "content": "Test"}],
                )

            assert exc.value.status == 500

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="Azure OpenAI API key is not set"):
            AzureOpenAIClient(api_key="", resource_name="test-resource")

        with pytest.raises(ValueError, match="Azure OpenAI API key is not set"):
            AzureOpenAIClient(api_key=None, resource_name="test-resource")

    def test_client_initialization_invalid_resource_name(self):
        with pytest.raises(ValueError, match="Azure OpenAI resource name is not set"):
            AzureOpenAIClient(api_key="test_key", resource_name="")

        with pytest.raises(ValueError, match="Azure OpenAI resource name is not set"):
            AzureOpenAIClient(api_key="test_key", resource_name=None)

    async def test_web_search_action_types(self, client: AzureOpenAIClient):
        mock_response = {
            "id": "resp-789",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5-mini-deployment",
            "output": [
                {
                    "id": "ws-1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "test query", "queries": ["test query"]},
                },
                {
                    "id": "ws-2",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "open_page", "url": "https://example.com"},
                },
                {
                    "id": "ws-3",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "find_in_page",
                        "url": "https://example.com",
                        "pattern": "<title",
                    },
                },
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Found results"}],
                },
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://test-resource.openai.azure.com/openai/deployments/gpt-5-mini-deployment/responses?api-version=2024-02-15-preview",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                deployment="gpt-5-mini-deployment",
                input=[{"role": "user", "content": "Test"}],
                tools=[{"type": "web_search"}],
            )

            assert isinstance(result, AzureOpenAIResponse)
            assert len(result.output) == 4

            search_action = result.output[0].action
            assert search_action.type == "search"
            assert search_action.query == "test query"
            assert search_action.queries == ["test query"]
            assert search_action.url is None

            open_page_action = result.output[1].action
            assert open_page_action.type == "open_page"
            assert open_page_action.url == "https://example.com"
            assert open_page_action.query is None

            find_in_page_action = result.output[2].action
            assert find_in_page_action.type == "find_in_page"
            assert find_in_page_action.url == "https://example.com"
            assert find_in_page_action.pattern == "<title"
            assert find_in_page_action.query is None

    async def test_different_api_versions(self):
        """Test that different API versions work correctly."""
        client_v1 = AzureOpenAIClient(
            api_key="test_key",
            resource_name="test-resource",
            api_version="2023-12-01-preview",
        )

        mock_response = {
            "id": "resp-version",
            "object": "response",
            "created_at": 1677652288,
            "model": "gpt-5",
            "output": [
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Version test"}],
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://test-resource.openai.azure.com/openai/deployments/gpt-5-deployment/responses?api-version=2023-12-01-preview",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client_v1.create_response(
                deployment="gpt-5-deployment",
                input=[{"role": "user", "content": "Test"}],
            )

            assert isinstance(result, AzureOpenAIResponse)
            assert result.id == "resp-version"

    async def test_model_fallback_when_not_in_response(self, client: AzureOpenAIClient):
        mock_response = {
            "id": "resp-no-model",
            "object": "response",
            "created_at": 1677652288,
            "output": [
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Test"}],
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://test-resource.openai.azure.com/openai/deployments/my-deployment/responses?api-version=2024-02-15-preview",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.create_response(
                deployment="my-deployment",
                input=[{"role": "user", "content": "Test"}],
            )

            assert isinstance(result, AzureOpenAIResponse)
            assert result.model == "my-deployment"