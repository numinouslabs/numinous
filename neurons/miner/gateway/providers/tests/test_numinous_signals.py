from uuid import uuid4

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.numinous_signals import DEFAULT_BASE_URL, NuminousSignalsClient
from neurons.validator.models.numinous_signals import NewsOrder

NEWS_URL = f"{DEFAULT_BASE_URL}/api/v1/signals/news"

MOCK_NEWS_RESPONSE = {
    "count": 1,
    "items": [
        {
            "id": "news-1",
            "headline": "Carrier strike group repositions",
            "summary": "A summary",
            "source": "Reuters",
            "source_url": "https://example.com/article",
            "source_timestamp": "2026-08-04T10:00:00Z",
            "emitted_at": "2026-08-04T10:05:00Z",
            "category": "geopolitics",
            "impacted_markets": [
                {
                    "condition_id": "0xabc",
                    "question": "Will it happen?",
                    "impact": True,
                    "direction": "supports_yes",
                    "impact_score": 0.82,
                    "rationale": "Direct movement toward the threshold",
                }
            ],
        }
    ],
}


class TestNuminousSignalsNewsFeed:
    @pytest.fixture
    def client(self):
        return NuminousSignalsClient(api_key="test-key")

    def test_missing_api_key_rejected(self):
        with pytest.raises(ValueError, match="Numinous Signals API key is not set"):
            NuminousSignalsClient(api_key="")

    async def test_get_news_feed_success(self, client: NuminousSignalsClient):
        event_id = uuid4()

        with aioresponses() as mocked:
            mocked.get(
                f"{NEWS_URL}?event_id={event_id}&order=recent&limit=20&offset=0",
                payload=MOCK_NEWS_RESPONSE,
            )

            result = await client.get_news_feed(event_id=event_id)

        assert result.count == 1
        assert len(result.items) == 1
        assert result.items[0].headline == "Carrier strike group repositions"
        assert result.items[0].impacted_markets[0].condition_id == "0xabc"

    async def test_get_news_feed_sends_optional_filters(self, client: NuminousSignalsClient):
        event_id = uuid4()

        with aioresponses() as mocked:
            mocked.get(
                f"{NEWS_URL}?event_id={event_id}&order=impact&limit=5&offset=2"
                "&language=en&min_impact_score=0.4&published_within_hours=12.0",
                payload=MOCK_NEWS_RESPONSE,
            )

            result = await client.get_news_feed(
                event_id=event_id,
                language="en",
                min_impact_score=0.4,
                order=NewsOrder.IMPACT,
                published_within_hours=12.0,
                limit=5,
                offset=2,
            )

        assert result.count == 1

    async def test_get_news_feed_omits_unset_filters(self, client: NuminousSignalsClient):
        event_id = uuid4()

        with aioresponses() as mocked:
            mocked.get(
                f"{NEWS_URL}?event_id={event_id}&order=recent&limit=20&offset=0",
                payload=MOCK_NEWS_RESPONSE,
            )

            await client.get_news_feed(event_id=event_id)

            request_url = str(next(iter(mocked.requests))[1])

        assert "language" not in request_url
        assert "min_impact_score" not in request_url
        assert "published_within_hours" not in request_url

    async def test_get_news_feed_error_raised(self, client: NuminousSignalsClient):
        event_id = uuid4()

        with aioresponses() as mocked:
            mocked.get(
                f"{NEWS_URL}?event_id={event_id}&order=recent&limit=20&offset=0",
                status=404,
                payload={"detail": "Event not found"},
            )

            with pytest.raises(ClientResponseError) as error:
                await client.get_news_feed(event_id=event_id)

        assert error.value.status == 404
