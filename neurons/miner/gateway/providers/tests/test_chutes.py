import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.chutes import ChutesClient
from neurons.validator.models.chutes import ChutesCompletion, ChuteStatus


class TestChutesClient:
    @pytest.fixture
    def client(self):
        return ChutesClient(api_key="test_api_key")

    async def test_chat_completion_success(self, client: ChutesClient):
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-ai/DeepSeek-R1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://llm.chutes.ai/v1/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="deepseek-ai/DeepSeek-R1",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
            )

            assert isinstance(result, ChutesCompletion)
            assert result.id == "chatcmpl-123"
            assert result.model == "deepseek-ai/DeepSeek-R1"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Test response"

    async def test_chat_completion_with_optional_params(self, client: ChutesClient):
        mock_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "Qwen/Qwen3-32B",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response with tools"},
                    "finish_reason": "stop",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://llm.chutes.ai/v1/chat/completions",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="Qwen/Qwen3-32B",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.5,
                max_tokens=100,
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

            assert isinstance(result, ChutesCompletion)
            assert result.id == "chatcmpl-456"
            assert result.model == "Qwen/Qwen3-32B"

    async def test_chat_completion_error_raised(self, client: ChutesClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://llm.chutes.ai/v1/chat/completions",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="test-model", messages=[{"role": "user", "content": "Test"}]
                )

            assert exc.value.status == 500

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="Chutes API key is not set"):
            ChutesClient(api_key="")

        with pytest.raises(ValueError, match="Chutes API key is not set"):
            ChutesClient(api_key=None)

    async def test_get_chutes_status_success(self, client: ChutesClient):
        mock_response = [
            {
                "chute_id": "chute-123",
                "name": "deepseek-ai/DeepSeek-R1",
                "timestamp": "2025-11-11T12:00:00Z",
                "utilization_current": 0.85,
                "utilization_5m": 0.75,
                "utilization_15m": 0.70,
                "utilization_1h": 0.65,
                "rate_limit_ratio_5m": 0.1,
                "rate_limit_ratio_15m": 0.08,
                "rate_limit_ratio_1h": 0.05,
                "total_requests_5m": 100.0,
                "total_requests_15m": 280.0,
                "total_requests_1h": 1000.0,
                "completed_requests_5m": 90.0,
                "completed_requests_15m": 257.0,
                "completed_requests_1h": 950.0,
                "rate_limited_requests_5m": 10.0,
                "rate_limited_requests_15m": 23.0,
                "rate_limited_requests_1h": 50.0,
                "instance_count": 5,
                "action_taken": "scale_up",
                "target_count": 6,
                "total_instance_count": 10,
                "active_instance_count": 5,
                "scalable": True,
                "scale_allowance": 3,
                "avg_busy_ratio": 0.75,
                "total_invocations": 1500.0,
                "total_rate_limit_errors": 50.0,
            },
            {
                "chute_id": "chute-456",
                "name": "Qwen/Qwen3-32B",
                "timestamp": "2025-11-11T12:00:00Z",
                "utilization_current": 0.45,
                "utilization_5m": 0.40,
                "utilization_15m": 0.42,
                "utilization_1h": 0.38,
                "rate_limit_ratio_5m": 0.02,
                "rate_limit_ratio_15m": 0.015,
                "rate_limit_ratio_1h": 0.01,
                "total_requests_5m": 50.0,
                "total_requests_15m": 140.0,
                "total_requests_1h": 500.0,
                "completed_requests_5m": 49.0,
                "completed_requests_15m": 138.0,
                "completed_requests_1h": 495.0,
                "rate_limited_requests_5m": 1.0,
                "rate_limited_requests_15m": 2.0,
                "rate_limited_requests_1h": 5.0,
                "instance_count": 3,
                "action_taken": "none",
                "target_count": 3,
                "total_instance_count": 10,
                "active_instance_count": 3,
                "scalable": True,
                "scale_allowance": 5,
                "avg_busy_ratio": 0.40,
                "total_invocations": 750.0,
                "total_rate_limit_errors": 5.0,
            },
        ]

        with aioresponses() as mocked:
            mocked.get(
                "https://api.chutes.ai/chutes/utilization",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.get_chutes_status()

            assert isinstance(result, list)
            assert len(result) == 2

            assert isinstance(result[0], ChuteStatus)
            assert result[0].chute_id == "chute-123"
            assert result[0].name == "deepseek-ai/DeepSeek-R1"
            assert result[0].utilization_current == 0.85
            assert result[0].utilization_5m == 0.75
            assert result[0].instance_count == 5
            assert result[0].action_taken == "scale_up"
            assert result[0].scalable is True

            assert isinstance(result[1], ChuteStatus)
            assert result[1].chute_id == "chute-456"
            assert result[1].name == "Qwen/Qwen3-32B"
            assert result[1].utilization_current == 0.45
            assert result[1].action_taken == "none"

    async def test_get_chutes_status_error_raised(self, client: ChutesClient):
        with aioresponses() as mocked:
            mocked.get(
                "https://api.chutes.ai/chutes/utilization",
                status=401,
                body=b"Unauthorized",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.get_chutes_status()

            assert exc.value.status == 401

    async def test_get_chutes_status_empty_list(self, client: ChutesClient):
        mock_response = []

        with aioresponses() as mocked:
            mocked.get(
                "https://api.chutes.ai/chutes/utilization",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.get_chutes_status()

            assert isinstance(result, list)
            assert len(result) == 0
