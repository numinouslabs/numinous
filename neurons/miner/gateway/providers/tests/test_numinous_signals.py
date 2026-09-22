from uuid import uuid4

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses

from neurons.miner.gateway.providers.numinous_signals import DEFAULT_BASE_URL, NuminousSignalsClient
from neurons.validator.models.numinous_signals import NewsOrder

NEWS_URL = f"{DEFAULT_BASE_URL}/api/v1/signals/news"
NUMINOUS1_URL = f"{DEFAULT_BASE_URL}/api/v1/numinous-1/chat/completions"

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


MOCK_NUMINOUS1_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1758480000,
    "model": "numinous/numinous-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"probability": 0.12}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 943,
        "completion_tokens": 65,
        "total_tokens": 1008,
        "cost": 0.0004,
    },
}


class TestNuminousSignalsNuminous1:
    @pytest.fixture
    def client(self):
        return NuminousSignalsClient(api_key="test-key")

    async def test_numinous1_success(self, client: NuminousSignalsClient):
        with aioresponses() as mocked:
            mocked.post(NUMINOUS1_URL, payload=MOCK_NUMINOUS1_RESPONSE)

            result = await client.numinous1_chat_completion(
                messages=[{"role": "user", "content": "Will it reopen?"}],
                model="numinous/numinous-1",
                temperature=0.0,
                max_tokens=400,
            )

        assert result.usage.cost == 0.0004
        assert result.usage.prompt_tokens == 943
        assert result.choices[0].message.content == '{"probability": 0.12}'

    async def test_numinous1_omits_unset_parameters(self, client: NuminousSignalsClient):
        with aioresponses() as mocked:
            mocked.post(NUMINOUS1_URL, payload=MOCK_NUMINOUS1_RESPONSE)

            await client.numinous1_chat_completion(
                messages=[{"role": "user", "content": "Will it reopen?"}],
                model="numinous/numinous-1",
                temperature=0.0,
                max_tokens=400,
            )

            request = next(iter(mocked.requests.values()))[0]

        body = request.kwargs["json"]
        assert set(body) == {"messages", "model", "temperature", "max_tokens"}

    async def test_numinous1_sends_optional_parameters(self, client: NuminousSignalsClient):
        with aioresponses() as mocked:
            mocked.post(NUMINOUS1_URL, payload=MOCK_NUMINOUS1_RESPONSE)

            await client.numinous1_chat_completion(
                messages=[{"role": "user", "content": "Will it reopen?"}],
                model="numinous/numinous-1",
                temperature=0.0,
                max_tokens=400,
                top_p=0.9,
                seed=7,
                response_format={"type": "json_object"},
            )

            request = next(iter(mocked.requests.values()))[0]

        body = request.kwargs["json"]
        assert body["top_p"] == 0.9
        assert body["seed"] == 7
        assert body["response_format"] == {"type": "json_object"}

    async def test_numinous1_error_raised(self, client: NuminousSignalsClient):
        with aioresponses() as mocked:
            mocked.post(NUMINOUS1_URL, status=402, payload={"detail": "Insufficient credits"})

            with pytest.raises(ClientResponseError):
                await client.numinous1_chat_completion(
                    messages=[{"role": "user", "content": "Will it reopen?"}],
                    model="numinous/numinous-1",
                    temperature=0.0,
                    max_tokens=400,
                )
