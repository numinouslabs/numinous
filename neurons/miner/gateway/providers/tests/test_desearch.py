import json

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.desearch import DesearchClient
from neurons.validator.models.desearch import (
    AISearchResponse,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
)


class TestDesearchClient:
    @pytest.fixture
    def client(self):
        return DesearchClient(api_key="test_api_key")

    async def test_ai_search_success(self, client: DesearchClient):
        mock_response = {
            "id": "search-123",
            "results": [
                {"title": "Test Result 1", "url": "https://example.com/1"},
                {"title": "Test Result 2", "url": "https://example.com/2"},
            ],
            "summary": "Test summary",
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.ai_search(prompt="What is AI?")

            assert isinstance(result, AISearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_ai_search_with_optional_params(self, client: DesearchClient):
        mock_response = {"id": "search-456", "results": []}

        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.ai_search(
                prompt="Latest tech news",
                model="ORBIT",
                tools=["web", "reddit"],
                date_filter="PAST_WEEK",
                result_type="LINKS_WITH_SUMMARIES",
                count=20,
            )

            assert isinstance(result, AISearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_ai_search_with_system_message(self, client: DesearchClient):
        mock_response = {"id": "search-789", "results": []}

        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.ai_search(
                prompt="Explain quantum computing",
                system_message="Be concise and technical",
            )

            assert isinstance(result, AISearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_ai_search_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search",
                status=429,
                body=b"Rate limit exceeded",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.ai_search(prompt="Test query")

            assert exc.value.status == 429

    def test_client_initialization_invalid_api_key(self):
        with pytest.raises(ValueError, match="Desearch API key is not set"):
            DesearchClient(api_key="")

        with pytest.raises(ValueError, match="Desearch API key is not set"):
            DesearchClient(api_key=None)

    async def test_web_links_search_success(self, client: DesearchClient):
        mock_response = {
            "links": [
                {"url": "https://example.com/1", "title": "Example 1"},
                {"url": "https://example.com/2", "title": "Example 2"},
            ],
        }

        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search/links/web",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.web_links_search(prompt="AI trends")

            assert isinstance(result, WebLinksResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_web_links_search_with_optional_params(self, client: DesearchClient):
        mock_response = {"links": []}

        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search/links/web",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.web_links_search(
                prompt="Machine learning",
                model="ORBIT",
                tools=["web"],
                count=15,
            )

            assert isinstance(result, WebLinksResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_web_links_search_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.post(
                "https://api.desearch.ai/desearch/ai/search/links/web",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.web_links_search(prompt="Test query")

            assert exc.value.status == 500

    async def test_web_search_success(self, client: DesearchClient):
        mock_response = {
            "data": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ],
        }

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/web?query=python&num_results=10&start=0",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.web_search(query="python")

            assert isinstance(result, WebSearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_web_search_with_optional_params(self, client: DesearchClient):
        mock_response = {"data": []}

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/web?query=machine+learning&num_results=25&start=10",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.web_search(query="machine learning", num_results=25, start=10)

            assert isinstance(result, WebSearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response

    async def test_web_search_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/web?query=test&num_results=10&start=0",
                status=403,
                body=b"Forbidden",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.web_search(query="test")

            assert exc.value.status == 403

    async def test_web_crawl_success(self, client: DesearchClient):
        mock_content = "<html><body><h1>Test Page</h1><p>This is test content.</p></body></html>"

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/web/crawl?url=https%3A%2F%2Fexample.com",
                status=200,
                body=mock_content.encode("utf-8"),
            )

            result = await client.web_crawl(url="https://example.com")

            assert isinstance(result, WebCrawlResponse)
            assert result.url == "https://example.com"
            assert result.content == mock_content

    async def test_web_crawl_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/web/crawl?url=https%3A%2F%2Fexample.com",
                status=404,
                body=b"Page not found",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.web_crawl(url="https://example.com")

            assert exc.value.status == 404
