import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.openrouter import OpenRouterClient
from neurons.validator.models.openrouter import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    OpenRouterCompletion,
    OpenRouterDecision,
    ScoreAnswer,
    ScoreQuestion,
)

MOCK_RESPONSE = {
    "id": "gen-abc123",
    "object": "chat.completion",
    "created": 1709000000,
    "model": "anthropic/claude-sonnet-4-6",
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
        "cost": 0.00165,
    },
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class TestOpenRouterClient:
    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="test_api_key")

    async def test_chat_completion_success(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(
                OPENROUTER_URL,
                status=200,
                body=json.dumps(MOCK_RESPONSE).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.5,
            )

            assert isinstance(result, OpenRouterCompletion)
            assert result.id == "gen-abc123"
            assert result.model == "anthropic/claude-sonnet-4-6"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Test response"
            assert result.usage.prompt_tokens == 50
            assert result.usage.completion_tokens == 100

    async def test_chat_completion_with_tools(self, client: OpenRouterClient):
        response = {
            **MOCK_RESPONSE,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                OPENROUTER_URL,
                status=200,
                body=json.dumps(response).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Weather in NYC?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                tool_choice="auto",
            )

            assert result.choices[0].message.tool_calls is not None
            assert result.choices[0].finish_reason == "tool_calls"

    async def test_chat_completion_with_max_tokens(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(
                OPENROUTER_URL,
                status=200,
                body=json.dumps(MOCK_RESPONSE).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=500,
            )

            assert isinstance(result, OpenRouterCompletion)

    async def test_chat_completion_minimal_response(self, client: OpenRouterClient):
        minimal = {
            "id": "gen-minimal",
            "object": "chat.completion",
            "created": 1709000000,
            "model": "anthropic/claude-haiku-4-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Short"},
                    "finish_reason": "stop",
                }
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                OPENROUTER_URL,
                status=200,
                body=json.dumps(minimal).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="anthropic/claude-haiku-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert isinstance(result, OpenRouterCompletion)
            assert result.usage is None

    async def test_chat_completion_server_error(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(OPENROUTER_URL, status=500, body=b"Internal server error")

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "Test"}],
                )

            assert exc.value.status == 500

    async def test_chat_completion_authentication_error(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(OPENROUTER_URL, status=401, body=b"Unauthorized")

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "Test"}],
                )

            assert exc.value.status == 401

    async def test_chat_completion_rate_limit(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(OPENROUTER_URL, status=429, body=b"Rate limit exceeded")

            with pytest.raises(ClientResponseError) as exc:
                await client.chat_completion(
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "Test"}],
                )

            assert exc.value.status == 429

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="OpenRouter API key is not set"):
            OpenRouterClient(api_key="")

        with pytest.raises(ValueError, match="OpenRouter API key is not set"):
            OpenRouterClient(api_key=None)

    async def test_chat_completion_with_kwargs(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(
                OPENROUTER_URL,
                status=200,
                body=json.dumps(MOCK_RESPONSE).encode("utf-8"),
            )

            result = await client.chat_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Test"}],
                top_p=0.9,
                frequency_penalty=0.5,
            )

            assert isinstance(result, OpenRouterCompletion)


DECISIONS_URL = "https://openrouter.ai/api/v1/systemone"

MOCK_DECISION = {
    "id": "gen-dec-abc123",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "will_happen": {"type": "noul", "noul": 0.24},
        "direction": {
            "type": "choice",
            "choice": "toward_no",
            "probabilities": {"toward_yes": 0.05, "toward_no": 0.95},
            "confidence": 0.9,
        },
        "strength": {
            "type": "score",
            "score": 0.67,
            "legend": {"0": "weak", "1": "moderate", "2": "strong"},
            "probabilities": {"0": 0.45, "1": 0.43, "2": 0.12},
            "confidence": 0.0,
        },
    },
    "usage": {"input_tokens": 1675, "output_tokens": 79, "cost": 0.00007},
}


def _questions() -> dict:
    return {
        "will_happen": NoulQuestion(instructions="Will this resolve YES?"),
        "direction": ChoiceQuestion(
            instructions="Which way?",
            criteria={"toward_yes": "More likely", "toward_no": "Less likely"},
        ),
        "strength": ScoreQuestion(
            instructions="How strong?", criteria=["weak", "moderate", "strong"]
        ),
    }


class TestOpenRouterDecisions:
    @pytest.fixture
    def client(self):
        return OpenRouterClient(api_key="test_api_key")

    async def test_decisions_success(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(DECISIONS_URL, payload=MOCK_DECISION, status=200)

            result = await client.decisions(
                model="typesafe/jev-1.13",
                state={"question": "Will the Fed cut?"},
                questions=_questions(),
            )

        assert isinstance(result, OpenRouterDecision)
        assert result.provider == "TypeSafe"
        assert isinstance(result.answers["will_happen"], NoulAnswer)
        assert result.answers["will_happen"].noul == 0.24
        assert isinstance(result.answers["direction"], ChoiceAnswer)
        assert result.answers["direction"].choice == "toward_no"
        assert isinstance(result.answers["strength"], ScoreAnswer)
        assert result.answers["strength"].legend["2"] == "strong"

    async def test_decisions_sends_state_and_questions(self, client: OpenRouterClient):
        state = {"question": "Will the Fed cut?", "evidence": ["hot CPI"]}

        with aioresponses() as mocked:
            mocked.post(DECISIONS_URL, payload=MOCK_DECISION, status=200)

            await client.decisions(model="typesafe/jev-1.13", state=state, questions=_questions())

            request = next(iter(mocked.requests.values()))[0]
            body = json.loads(request.kwargs["json"] and json.dumps(request.kwargs["json"]))

        assert body["model"] == "typesafe/jev-1.13"
        assert body["state"] == state
        assert body["questions"]["direction"]["criteria"] == {
            "toward_yes": "More likely",
            "toward_no": "Less likely",
        }
        assert body["questions"]["strength"]["criteria"] == ["weak", "moderate", "strong"]
        assert "messages" not in body

    async def test_decisions_omits_unset_noul_criteria(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(DECISIONS_URL, payload=MOCK_DECISION, status=200)

            await client.decisions(
                model="typesafe/jev-1.13",
                state="x",
                questions={"q": NoulQuestion(instructions="Urgent?")},
            )

            request = next(iter(mocked.requests.values()))[0]
            body = request.kwargs["json"]

        assert "criteria" not in body["questions"]["q"]

    async def test_decisions_raises_on_error_status(self, client: OpenRouterClient):
        with aioresponses() as mocked:
            mocked.post(DECISIONS_URL, status=400, payload={"error": {"message": "nope"}})

            with pytest.raises(ClientResponseError):
                await client.decisions(
                    model="typesafe/jev-9.9",
                    state="x",
                    questions={"q": NoulQuestion(instructions="ok")},
                )
