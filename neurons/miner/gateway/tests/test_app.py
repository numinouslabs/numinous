from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from neurons.miner.gateway.app import app
from neurons.validator.models.chutes import ChutesCompletion, ChuteStatus
from neurons.validator.models.desearch import (
    AISearchResponse,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestGatewayApp:
    def test_app_configuration(self):
        routes = [route.path for route in app.routes]
        assert "/api/health" in routes
        assert "/api/gateway/chutes/chat/completions" in routes
        assert "/api/gateway/chutes/status" in routes
        assert "/api/gateway/desearch/ai/search" in routes
        assert "/api/gateway/desearch/ai/links" in routes
        assert "/api/gateway/desearch/web/search" in routes
        assert "/api/gateway/desearch/web/crawl" in routes


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient):
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "API Gateway"}


class TestChutesEndpoint:
    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_chutes_completion_success(self, mock_client_class, client: TestClient):
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

        response = client.post("/api/gateway/chutes/chat/completions", json=request_body)

        assert response.status_code == 200

        result = response.json()
        assert result["id"] == "chatcmpl-123"
        assert result["model"] == "deepseek-ai/DeepSeek-R1"

    @patch.dict("os.environ", {"CHUTES_API_KEY": ""})
    def test_chutes_completion_missing_api_key(self, client: TestClient):
        request_body = {
            "run_id": str(uuid4()),
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client.post("/api/gateway/chutes/chat/completions", json=request_body)

        assert response.status_code == 401
        assert "CHUTES_API_KEY not configured" in response.json()["detail"]

    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_chutes_completion_api_error(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.chat_completion = AsyncMock(side_effect=Exception("API Error"))

        request_body = {
            "run_id": str(uuid4()),
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client.post("/api/gateway/chutes/chat/completions", json=request_body)

        assert response.status_code == 500
        assert "Chutes API error" in response.json()["detail"]

    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_get_chutes_status_success(self, mock_client_class, client: TestClient):
        mock_response = [
            ChuteStatus(
                chute_id="chute-123",
                name="deepseek-ai/DeepSeek-R1",
                timestamp="2025-11-11T12:00:00Z",
                utilization_current=0.85,
                utilization_5m=0.75,
                utilization_15m=0.70,
                utilization_1h=0.65,
                rate_limit_ratio_5m=0.1,
                rate_limit_ratio_15m=0.08,
                rate_limit_ratio_1h=0.05,
                total_requests_5m=100.0,
                total_requests_15m=280.0,
                total_requests_1h=1000.0,
                completed_requests_5m=90.0,
                completed_requests_15m=257.0,
                completed_requests_1h=950.0,
                rate_limited_requests_5m=10.0,
                rate_limited_requests_15m=23.0,
                rate_limited_requests_1h=50.0,
                instance_count=5,
                action_taken="scale_up",
                target_count=6,
                total_instance_count=10,
                active_instance_count=5,
                scalable=True,
                scale_allowance=3,
                avg_busy_ratio=0.75,
                total_invocations=1500.0,
                total_rate_limit_errors=50.0,
            ),
            ChuteStatus(
                chute_id="chute-456",
                name="Qwen/Qwen3-32B",
                timestamp="2025-11-11T12:00:00Z",
                utilization_current=0.45,
                utilization_5m=0.40,
                utilization_15m=0.42,
                utilization_1h=0.38,
                rate_limit_ratio_5m=0.02,
                rate_limit_ratio_15m=0.015,
                rate_limit_ratio_1h=0.01,
                total_requests_5m=50.0,
                total_requests_15m=140.0,
                total_requests_1h=500.0,
                completed_requests_5m=49.0,
                completed_requests_15m=138.0,
                completed_requests_1h=495.0,
                rate_limited_requests_5m=1.0,
                rate_limited_requests_15m=2.0,
                rate_limited_requests_1h=5.0,
                instance_count=3,
                action_taken="none",
                target_count=3,
                total_instance_count=10,
                active_instance_count=3,
                scalable=True,
                scale_allowance=5,
                avg_busy_ratio=0.40,
                total_invocations=750.0,
                total_rate_limit_errors=5.0,
            ),
        ]

        mock_instance = mock_client_class.return_value
        mock_instance.get_chutes_status = AsyncMock(return_value=mock_response)

        response = client.get("/api/gateway/chutes/status")

        assert response.status_code == 200

        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["chute_id"] == "chute-123"
        assert result[0]["name"] == "deepseek-ai/DeepSeek-R1"
        assert result[0]["utilization_current"] == 0.85
        assert result[0]["action_taken"] == "scale_up"
        assert result[1]["chute_id"] == "chute-456"
        assert result[1]["name"] == "Qwen/Qwen3-32B"

    @patch.dict("os.environ", {"CHUTES_API_KEY": ""})
    def test_get_chutes_status_missing_api_key(self, client: TestClient):
        response = client.get("/api/gateway/chutes/status")

        assert response.status_code == 401
        assert "CHUTES_API_KEY not configured" in response.json()["detail"]

    @patch("neurons.miner.gateway.app.ChutesClient")
    @patch.dict("os.environ", {"CHUTES_API_KEY": "test-key"})
    def test_get_chutes_status_api_error(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.get_chutes_status = AsyncMock(side_effect=Exception("API Error"))

        response = client.get("/api/gateway/chutes/status")

        assert response.status_code == 500
        assert "Chutes API error" in response.json()["detail"]


class TestDesearchEndpoints:
    @patch("neurons.miner.gateway.app.DesearchClient")
    @patch.dict("os.environ", {"DESEARCH_API_KEY": "test-key"})
    def test_ai_search_success(self, mock_client_class, client: TestClient):
        mock_response = AISearchResponse(
            text="Test text", completion="Test summary", miner_link_scores={}
        )

        mock_instance = mock_client_class.return_value
        mock_instance.ai_search = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "prompt": "What is AI?",
            "model": "NOVA",
            "tools": ["web"],
        }

        response = client.post("/api/gateway/desearch/ai/search", json=request_body)

        assert response.status_code == 200

        result = response.json()
        assert result["text"] == "Test text"
        assert result["completion"] == "Test summary"

    @patch("neurons.miner.gateway.app.DesearchClient")
    @patch.dict("os.environ", {"DESEARCH_API_KEY": "test-key"})
    def test_web_links_search_success(self, mock_client_class, client: TestClient):
        mock_response = WebLinksResponse(
            links=[
                {"url": "https://example.com/1", "title": "Example 1"},
                {"url": "https://example.com/2", "title": "Example 2"},
            ]
        )

        mock_instance = mock_client_class.return_value
        mock_instance.web_links_search = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "prompt": "AI trends",
            "model": "NOVA",
            "tools": ["web"],
        }

        response = client.post("/api/gateway/desearch/ai/links", json=request_body)

        assert response.status_code == 200

        result = response.json()
        assert "links" in result
        assert len(result["links"]) == 2

    @patch("neurons.miner.gateway.app.DesearchClient")
    @patch.dict("os.environ", {"DESEARCH_API_KEY": "test-key"})
    def test_web_search_success(self, mock_client_class, client: TestClient):
        mock_response = WebSearchResponse(
            data=[
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        )

        mock_instance = mock_client_class.return_value
        mock_instance.web_search = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "query": "python programming",
            "num": 10,
            "start": 0,
        }

        response = client.post("/api/gateway/desearch/web/search", json=request_body)

        assert response.status_code == 200

        result = response.json()
        assert "data" in result
        assert len(result["data"]) == 2

    @patch("neurons.miner.gateway.app.DesearchClient")
    @patch.dict("os.environ", {"DESEARCH_API_KEY": "test-key"})
    def test_web_crawl_success(self, mock_client_class, client: TestClient):
        mock_content = "<html><body><h1>Test Page</h1><p>This is test content.</p></body></html>"
        mock_response = WebCrawlResponse(
            url="https://example.com",
            content=mock_content,
        )

        mock_instance = mock_client_class.return_value
        mock_instance.web_crawl = AsyncMock(return_value=mock_response)

        request_body = {
            "run_id": str(uuid4()),
            "url": "https://example.com",
        }

        response = client.post("/api/gateway/desearch/web/crawl", json=request_body)

        assert response.status_code == 200

        result = response.json()
        assert result["url"] == "https://example.com"
        assert result["content"] == mock_content

    @patch.dict("os.environ", {"DESEARCH_API_KEY": ""})
    def test_missing_api_key(self, client: TestClient):
        request_body = {
            "run_id": str(uuid4()),
            "prompt": "Test query",
        }

        response = client.post("/api/gateway/desearch/ai/search", json=request_body)

        assert response.status_code == 401
        assert "DESEARCH_API_KEY not configured" in response.json()["detail"]

    @patch("neurons.miner.gateway.app.DesearchClient")
    @patch.dict("os.environ", {"DESEARCH_API_KEY": "test-key"})
    def test_api_error(self, mock_client_class, client: TestClient):
        mock_instance = mock_client_class.return_value
        mock_instance.ai_search = AsyncMock(side_effect=Exception("API Error"))

        request_body = {
            "run_id": str(uuid4()),
            "prompt": "Test query",
        }

        response = client.post("/api/gateway/desearch/ai/search", json=request_body)

        assert response.status_code == 500
        assert "Desearch API error" in response.json()["detail"]


class TestRequestValidation:
    def test_chutes_invalid_request_body(self, client: TestClient):
        request_body = {"model": "test-model"}

        response = client.post("/api/gateway/chutes/chat/completions", json=request_body)

        assert response.status_code == 422

    def test_desearch_invalid_request_body(self, client: TestClient):
        request_body = {}

        response = client.post("/api/gateway/desearch/ai/search", json=request_body)

        assert response.status_code == 422
