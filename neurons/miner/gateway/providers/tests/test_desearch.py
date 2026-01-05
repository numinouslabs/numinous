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
    XPostResponse,
    XPostSummary,
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

    async def test_x_search_success(self, client: DesearchClient):
        mock_response = [
            {
                "id": "1234567890",
                "text": "This is a test tweet about AI",
                "reply_count": 5,
                "view_count": 1000,
                "retweet_count": 10,
                "like_count": 50,
                "quote_count": 2,
                "bookmark_count": 3,
                "url": "https://twitter.com/user/status/1234567890",
                "created_at": "2024-01-01T12:00:00Z",
                "media": [],
                "is_quote_tweet": False,
                "is_retweet": False,
                "lang": "en",
                "conversation_id": "1234567890",
            },
            {
                "id": "0987654321",
                "text": "Another tweet about machine learning",
                "reply_count": 3,
                "view_count": 500,
                "retweet_count": 5,
                "like_count": 25,
                "quote_count": 1,
                "bookmark_count": 2,
                "url": "https://twitter.com/user/status/0987654321",
                "created_at": "2024-01-01T11:00:00Z",
                "media": [],
                "is_quote_tweet": False,
                "is_retweet": False,
                "lang": "en",
                "conversation_id": "0987654321",
            },
        ]

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter?query=AI&sort=Top&count=20",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.x_search(query="AI")

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(post, XPostSummary) for post in result)
            assert result[0].id == "1234567890"
            assert result[0].text == "This is a test tweet about AI"
            assert result[1].id == "0987654321"

    async def test_x_search_with_optional_params(self, client: DesearchClient):
        mock_response = [
            {
                "id": "1111111111",
                "text": "Test tweet with filters",
                "reply_count": 10,
                "view_count": 2000,
                "retweet_count": 20,
                "like_count": 100,
                "quote_count": 5,
                "bookmark_count": 8,
                "url": "https://twitter.com/user/status/1111111111",
                "created_at": "2024-01-02T12:00:00Z",
                "media": [],
                "is_quote_tweet": False,
                "is_retweet": False,
                "lang": "en",
                "conversation_id": "1111111111",
            }
        ]

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter?query=blockchain&sort=Latest&user=elonmusk&lang=en&min_likes=50&count=10",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.x_search(
                query="blockchain",
                sort="Latest",
                user="elonmusk",
                lang="en",
                min_likes=50,
                count=10,
            )

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].id == "1111111111"
            assert result[0].like_count == 100

    async def test_x_search_with_count_filters(self, client: DesearchClient):
        mock_response = []

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter?query=crypto&sort=Top&min_retweets=100&min_replies=50&count=20",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.x_search(
                query="crypto",
                min_retweets=100,
                min_replies=50,
            )

            assert isinstance(result, list)
            assert len(result) == 0

    async def test_x_search_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter?query=test&sort=Top&count=20",
                status=500,
                body=b"Internal server error",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.x_search(query="test")

            assert exc.value.status == 500

    async def test_fetch_x_post_success(self, client: DesearchClient):
        mock_response = {
            "user": {
                "id": "123456",
                "url": "https://twitter.com/testuser",
                "name": "Test User",
                "username": "testuser",
                "created_at": "2020-01-01T00:00:00Z",
                "description": "Test user description",
                "favourites_count": 1000,
                "followers_count": 5000,
                "listed_count": 50,
                "media_count": 100,
                "profile_image_url": "https://example.com/image.jpg",
                "profile_banner_url": "https://example.com/banner.jpg",
                "statuses_count": 2000,
                "verified": True,
                "is_blue_verified": False,
                "can_dm": True,
                "can_media_tag": True,
                "location": "San Francisco, CA",
            },
            "id": "9876543210",
            "text": "This is a detailed tweet with full information",
            "reply_count": 15,
            "view_count": 5000,
            "retweet_count": 50,
            "like_count": 200,
            "quote_count": 10,
            "bookmark_count": 25,
            "url": "https://twitter.com/testuser/status/9876543210",
            "created_at": "2024-01-03T12:00:00Z",
            "media": [],
            "is_quote_tweet": False,
            "is_retweet": False,
            "lang": "en",
            "conversation_id": "9876543210",
        }

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter/post?id=9876543210",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.fetch_x_post(post_id="9876543210")

            assert isinstance(result, XPostResponse)
            assert result.id == "9876543210"
            assert result.text == "This is a detailed tweet with full information"
            assert result.user.username == "testuser"
            assert result.user.verified is True
            assert result.like_count == 200

    async def test_fetch_x_post_with_replies_and_quotes(self, client: DesearchClient):
        mock_response = {
            "user": {
                "id": "789012",
                "url": "https://twitter.com/anotheruser",
                "name": "Another User",
                "username": "anotheruser",
                "created_at": "2019-06-15T00:00:00Z",
                "description": "Another test user",
                "favourites_count": 500,
                "followers_count": 3000,
                "listed_count": 30,
                "media_count": 50,
                "profile_image_url": "https://example.com/image2.jpg",
                "profile_banner_url": "https://example.com/banner2.jpg",
                "statuses_count": 1500,
                "verified": False,
                "is_blue_verified": True,
                "can_dm": False,
                "can_media_tag": True,
                "location": "New York, NY",
            },
            "id": "5555555555",
            "text": "Tweet with replies and quotes",
            "reply_count": 5,
            "view_count": 1500,
            "retweet_count": 10,
            "like_count": 75,
            "quote_count": 3,
            "bookmark_count": 5,
            "url": "https://twitter.com/anotheruser/status/5555555555",
            "created_at": "2024-01-04T12:00:00Z",
            "media": [],
            "is_quote_tweet": True,
            "is_retweet": False,
            "lang": "en",
            "conversation_id": "5555555555",
            "quoted_status_id": "4444444444",
            "quote": {
                "id": "4444444444",
                "text": "Original quoted tweet",
                "reply_count": 0,
                "view_count": 500,
                "retweet_count": 5,
                "like_count": 20,
                "quote_count": 1,
                "bookmark_count": 2,
                "url": "https://twitter.com/user/status/4444444444",
                "created_at": "2024-01-03T10:00:00Z",
                "media": [],
                "is_quote_tweet": False,
                "is_retweet": False,
                "lang": "en",
                "conversation_id": "4444444444",
            },
        }

        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter/post?id=5555555555",
                status=200,
                body=json.dumps(mock_response).encode("utf-8"),
            )

            result = await client.fetch_x_post(post_id="5555555555")

            assert isinstance(result, XPostResponse)
            assert result.id == "5555555555"
            assert result.is_quote_tweet is True
            assert result.quoted_status_id == "4444444444"
            assert result.quote is not None
            assert isinstance(result.quote, XPostSummary)
            assert result.quote.id == "4444444444"
            assert result.user.is_blue_verified is True

    async def test_fetch_x_post_error_raised(self, client: DesearchClient):
        with aioresponses() as mocked:
            mocked.get(
                "https://api.desearch.ai/twitter/post?id=invalid_id",
                status=404,
                body=b"Post not found",
            )

            with pytest.raises(ClientResponseError) as exc:
                await client.fetch_x_post(post_id="invalid_id")

            assert exc.value.status == 404
