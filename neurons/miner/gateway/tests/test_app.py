from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from neurons.miner.gateway.app import app
from neurons.validator.models.numinous_signals import (
    CorpusFetchResponse,
    CorpusSearchResponse,
    CorpusSearchResult,
    NewsFeedPage,
)
from neurons.validator.models.openai import OpenAIResponse
from neurons.validator.models.openrouter import OpenRouterCompletion


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestGatewayApp:
    def test_app_configuration(self):
        routes = [route.path for route in app.routes]
        assert "/api/health" in routes
        assert "/api/gateway/openai/responses/inference" in routes
        assert "/api/gateway/openrouter/chat/completions/inference" in routes
        assert "/api/gateway/lightning-rod/chat/completions" in routes
        assert "/api/gateway/numinous-indicia/x-osint" in routes
        assert "/api/gateway/numinous-indicia/liveuamap" in routes
        assert "/api/gateway/numinous-signals/signals" in routes
        assert "/api/gateway/numinous-signals/causal-drivers/drivers" in routes
        assert "/api/gateway/numinous-signals/deep-research/report" in routes
        assert "/api/gateway/numinous-signals/corpus/search" in routes
        assert "/api/gateway/numinous-signals/corpus/fetch" in routes
        assert "/api/gateway/numinous-signals/news" in routes

    def test_deprecated_main_track_routes_are_gone(self):
        routes = [route.path for route in app.routes]
        assert "/api/gateway/openai/responses" not in routes
        assert "/api/gateway/openrouter/chat/completions" not in routes
        assert not any("chutes" in route for route in routes)
        assert not any("desearch" in route for route in routes)
        assert not any("perplexity" in route for route in routes)
        assert not any("vericore" in route for route in routes)
        assert not any("lunar-crush" in route for route in routes)
        assert not any("unusual-whales" in route for route in routes)
        assert not any("public-data" in route for route in routes)


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "API Gateway"}


class TestOpenAIInferenceEndpoint:
    _INFERENCE_PATH = "/api/gateway/openai/responses/inference"

    @staticmethod
    def _mock_response() -> OpenAIResponse:
        return OpenAIResponse(
            id="resp-abc123",
            object="response",
            created_at=1709000000,
            model="gpt-5",
            status="completed",
            output=[],
            usage={"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
        )

    @patch("neurons.miner.gateway.app.OpenAIClient")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_inference_success(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.create_response = AsyncMock(return_value=self._mock_response())

        request_body = {
            "run_id": str(uuid4()),
            "model": "gpt-5",
            "input": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "resp-abc123"
        assert "cost" in result

    @patch("neurons.miner.gateway.app.OpenAIClient")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_inference_rejects_web_search(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.create_response = AsyncMock()

        request_body = {
            "run_id": str(uuid4()),
            "model": "gpt-5",
            "input": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "web_search"}],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 400
        assert "web_search" in response.json()["detail"]
        mock_instance.create_response.assert_not_called()

    @patch("neurons.miner.gateway.app.OpenAIClient")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_inference_accepts_custom_function_tool(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.create_response = AsyncMock(return_value=self._mock_response())

        request_body = {
            "run_id": str(uuid4()),
            "model": "gpt-5",
            "input": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 200
        assert "cost" in response.json()

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""})
    def test_inference_missing_api_key(self, client: TestClient):
        request_body = {
            "run_id": str(uuid4()),
            "model": "gpt-5",
            "input": [{"role": "user", "content": "Test"}],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 401
        assert "OPENAI_API_KEY not configured" in response.json()["detail"]


class TestOpenRouterInferenceEndpoint:
    _INFERENCE_PATH = "/api/gateway/openrouter/chat/completions/inference"

    @staticmethod
    def _mock_response() -> OpenRouterCompletion:
        return OpenRouterCompletion(
            id="gen-abc123",
            object="chat.completion",
            created=1709000000,
            model="anthropic/claude-sonnet-4-6",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
                "cost": 0.00165,
            },
        )

    @patch("neurons.miner.gateway.app.OpenRouterClient")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_inference_success(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock(return_value=self._mock_response())

        request_body = {
            "run_id": str(uuid4()),
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "gen-abc123"
        assert "cost" in result

    @patch("neurons.miner.gateway.app.OpenRouterClient")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_inference_accepts_custom_function_tool(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock(return_value=self._mock_response())

        request_body = {
            "run_id": str(uuid4()),
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 200
        assert "cost" in response.json()

    @pytest.mark.parametrize(
        "bypass,expected_detail",
        [
            ({"model": "anthropic/claude-sonnet-4-6:online"}, ":online"),
            ({"tools": [{"type": "openrouter:web_search"}]}, "openrouter:web_search"),
            ({"plugins": [{"id": "web"}]}, "plugins"),
        ],
    )
    @patch("neurons.miner.gateway.app.OpenRouterClient")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_inference_rejects_builtin_provider_tools(
        self, mock_client_class, bypass, expected_detail, client: TestClient
    ):
        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock()

        request_body = {
            "run_id": str(uuid4()),
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            **bypass,
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 400
        assert expected_detail in response.json()["detail"]
        mock_instance.chat_completion.assert_not_called()

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": ""})
    def test_inference_missing_api_key(self, client: TestClient):
        request_body = {
            "run_id": str(uuid4()),
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client.post(self._INFERENCE_PATH, json=request_body)

        assert response.status_code == 401
        assert "OPENROUTER_API_KEY not configured" in response.json()["detail"]


class TestRequestValidation:
    def test_openrouter_inference_invalid_request_body(self, client: TestClient):
        request_body = {"model": "anthropic/claude-sonnet-4-6"}

        response = client.post(
            "/api/gateway/openrouter/chat/completions/inference", json=request_body
        )

        assert response.status_code == 422

    def test_openai_inference_invalid_request_body(self, client: TestClient):
        response = client.post("/api/gateway/openai/responses/inference", json={})

        assert response.status_code == 422


class TestNuminousCorpusEndpoints:
    @patch("neurons.miner.gateway.app.NuminousSignalsClient")
    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": "test-key"})
    def test_corpus_search_success(self, mock_client_class, client: TestClient):
        source_id = uuid4()
        mock_response = CorpusSearchResponse(
            results=[
                CorpusSearchResult(
                    source_id=source_id,
                    url="https://example.com/article",
                    title="Carrier group repositions",
                    snippet="...the carrier strike group was observed moving...",
                )
            ]
        )

        mock_instance = mock_client_class.return_value
        mock_instance.search_corpus = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "query": "carrier group movement",
            "max_results": 10,
        }

        response = client.post("/api/gateway/numinous-signals/corpus/search", json=request_body)

        assert response.status_code == 200
        result = response.json()
        assert result["cost"] == 0.0015
        assert len(result["results"]) == 1
        assert result["results"][0]["source_id"] == str(source_id)
        assert result["results"][0]["snippet"].startswith("...the carrier")

    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": ""})
    def test_corpus_search_missing_api_key(self, client: TestClient):
        request_body = {"run_id": str(uuid4()), "query": "anything"}

        response = client.post("/api/gateway/numinous-signals/corpus/search", json=request_body)

        assert response.status_code == 401
        assert "NUMINOUS_SIGNALS_API_KEY not configured" in response.json()["detail"]

    def test_corpus_search_invalid_request_body(self, client: TestClient):
        response = client.post("/api/gateway/numinous-signals/corpus/search", json={})

        assert response.status_code == 422

    @patch("neurons.miner.gateway.app.NuminousSignalsClient")
    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": "test-key"})
    def test_corpus_fetch_success(self, mock_client_class, client: TestClient):
        source_id = uuid4()
        mock_response = CorpusFetchResponse(
            source_id=source_id,
            url="https://example.com/article",
            title="Carrier group repositions",
            content="Full scraped article text...",
        )

        mock_instance = mock_client_class.return_value
        mock_instance.fetch_corpus_source = AsyncMock(return_value=mock_response)

        request_body = {"run_id": str(uuid4()), "source_id": str(source_id)}

        response = client.post("/api/gateway/numinous-signals/corpus/fetch", json=request_body)

        assert response.status_code == 200
        result = response.json()
        assert result["cost"] == 0.0005
        assert result["source_id"] == str(source_id)
        assert result["content"] == "Full scraped article text..."

    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": ""})
    def test_corpus_fetch_missing_api_key(self, client: TestClient):
        request_body = {"run_id": str(uuid4()), "source_id": str(uuid4())}

        response = client.post("/api/gateway/numinous-signals/corpus/fetch", json=request_body)

        assert response.status_code == 401
        assert "NUMINOUS_SIGNALS_API_KEY not configured" in response.json()["detail"]

    def test_corpus_fetch_invalid_source_id(self, client: TestClient):
        request_body = {"run_id": str(uuid4()), "source_id": "not-a-uuid"}

        response = client.post("/api/gateway/numinous-signals/corpus/fetch", json=request_body)

        assert response.status_code == 422


class TestNuminousNewsFeedEndpoint:
    _PATH = "/api/gateway/numinous-signals/news"

    @staticmethod
    def _page(impacted_markets: list[dict]) -> NewsFeedPage:
        return NewsFeedPage(
            count=1,
            items=[
                {
                    "id": "news-1",
                    "headline": "Carrier strike group repositions",
                    "summary": "A summary",
                    "source": "Reuters",
                    "source_url": "https://example.com/article",
                    "source_timestamp": "2026-08-04T10:00:00Z",
                    "emitted_at": "2026-08-04T10:05:00Z",
                    "category": "geopolitics",
                    "impacted_markets": impacted_markets,
                }
            ],
        )

    @staticmethod
    def _market(condition_id: str, impact_score: float, direction: str) -> dict:
        return {
            "condition_id": condition_id,
            "question": "Will it happen?",
            "impact": True,
            "direction": direction,
            "impact_score": impact_score,
            "rationale": f"rationale for {condition_id}",
        }

    @patch("neurons.miner.gateway.app.NuminousSignalsClient")
    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": "test-key"})
    def test_news_feed_success(self, mock_client_class, client: TestClient):
        event_id = uuid4()
        mock_instance = mock_client_class.return_value
        mock_instance.get_news_feed = AsyncMock(
            return_value=self._page([self._market("0xabc", 0.9, "supports_yes")])
        )

        response = client.post(self._PATH, json={"run_id": str(uuid4()), "event_id": str(event_id)})

        assert response.status_code == 200
        result = response.json()
        assert result["event_id"] == str(event_id)
        assert result["count"] == 1
        assert result["cost"] == 0.002
        assert len(result["articles"]) == 1
        article = result["articles"][0]
        assert article["headline"] == "Carrier strike group repositions"
        assert article["direction"] == "supports_yes"
        assert article["impact_score"] == 0.9

    @patch("neurons.miner.gateway.app.NuminousSignalsClient")
    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": "test-key"})
    def test_news_feed_forwards_event_id_and_filters(self, mock_client_class, client: TestClient):
        event_id = uuid4()
        mock_instance = mock_client_class.return_value
        mock_instance.get_news_feed = AsyncMock(
            return_value=self._page([self._market("0xabc", 0.7, "supports_no")])
        )

        response = client.post(
            self._PATH,
            json={
                "run_id": str(uuid4()),
                "event_id": str(event_id),
                "min_impact_score": 0.4,
                "order": "impact",
                "published_within_hours": 12,
                "limit": 5,
                "offset": 2,
            },
        )

        assert response.status_code == 200
        kwargs = mock_instance.get_news_feed.call_args.kwargs
        assert kwargs["event_id"] == event_id
        assert kwargs["min_impact_score"] == 0.4
        assert kwargs["order"] == "impact"
        assert kwargs["published_within_hours"] == 12
        assert kwargs["limit"] == 5
        assert kwargs["offset"] == 2

    @patch("neurons.miner.gateway.app.NuminousSignalsClient")
    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": "test-key"})
    def test_news_feed_drops_items_without_impacted_markets(
        self, mock_client_class, client: TestClient
    ):
        mock_instance = mock_client_class.return_value
        mock_instance.get_news_feed = AsyncMock(return_value=self._page([]))

        response = client.post(self._PATH, json={"run_id": str(uuid4()), "event_id": str(uuid4())})

        assert response.status_code == 200
        assert response.json()["articles"] == []

    @patch.dict("os.environ", {"NUMINOUS_SIGNALS_API_KEY": ""})
    def test_news_feed_missing_api_key(self, client: TestClient):
        response = client.post(self._PATH, json={"run_id": str(uuid4()), "event_id": str(uuid4())})

        assert response.status_code == 401
        assert "NUMINOUS_SIGNALS_API_KEY not configured" in response.json()["detail"]

    def test_news_feed_requires_event_id(self, client: TestClient):
        response = client.post(self._PATH, json={"run_id": str(uuid4())})

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "overrides",
        [
            {"min_impact_score": 1.5},
            {"limit": 31},
            {"offset": -1},
            {"published_within_hours": 0},
            {"order": "not-an-order"},
        ],
    )
    def test_news_feed_rejects_out_of_range_params(self, overrides, client: TestClient):
        response = client.post(
            self._PATH,
            json={"run_id": str(uuid4()), "event_id": str(uuid4()), **overrides},
        )

        assert response.status_code == 422
