from datetime import date, datetime
from urllib.parse import quote
from uuid import UUID

import aiohttp

from neurons.validator.models.numinous_signals import (
    CausalDriversResponse,
    CorpusFetchResponse,
    CorpusSearchResponse,
    DeepResearchReportResponse,
    MarketGraphResponse,
    NewsFeedPage,
    NewsOrder,
)

DEFAULT_BASE_URL = "https://signals.numinouslabs.io"
DEFAULT_TIMEOUT = 120.0


class NuminousSignalsClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise ValueError("Numinous Signals API key is not set")
        self.__api_key = api_key
        self.__base_url = base_url
        self.__timeout = aiohttp.ClientTimeout(total=timeout)
        self.__headers = {
            "X-API-Key": self.__api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def get_causal_drivers(
        self,
        event_id: str,
        topic: str = "geopolitics",
    ) -> CausalDriversResponse:
        body: dict = {"event_id": event_id, "topic": topic}

        url = f"{self.__base_url}/api/v1/causal-drivers/drivers"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return CausalDriversResponse.model_validate(data)

    async def get_deep_research_report(
        self,
        event_id: str | None = None,
        polymarket_market_id: str | None = None,
        title: str | None = None,
        topics: list[str] | None = None,
    ) -> DeepResearchReportResponse:
        body: dict = {}
        if event_id is not None:
            body["event_id"] = event_id
        if polymarket_market_id is not None:
            body["polymarket_market_id"] = polymarket_market_id
        if title is not None:
            body["title"] = title
        if topics is not None:
            body["topics"] = topics

        url = f"{self.__base_url}/api/v1/deep-research/report"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return DeepResearchReportResponse.model_validate(data)

    async def search_corpus(
        self,
        query: str,
        max_results: int = 10,
        published_after: datetime | None = None,
        published_before: datetime | None = None,
    ) -> CorpusSearchResponse:
        body: dict = {"query": query, "max_results": max_results}
        if published_after is not None:
            body["published_after"] = published_after.isoformat()
        if published_before is not None:
            body["published_before"] = published_before.isoformat()

        url = f"{self.__base_url}/api/v1/corpus/search"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return CorpusSearchResponse.model_validate(data)

    async def get_news_feed(
        self,
        event_id: UUID,
        language: str | None = None,
        min_impact_score: float | None = None,
        order: NewsOrder = NewsOrder.RECENT,
        published_within_hours: float | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> NewsFeedPage:
        params: dict[str, str | int | float] = {
            "event_id": str(event_id),
            "order": str(order),
            "limit": limit,
            "offset": offset,
        }
        if language is not None:
            params["language"] = language
        if min_impact_score is not None:
            params["min_impact_score"] = min_impact_score
        if published_within_hours is not None:
            params["published_within_hours"] = published_within_hours

        url = f"{self.__base_url}/api/v1/signals/news"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return NewsFeedPage.model_validate(data)

    async def fetch_corpus_source(self, source_id: UUID) -> CorpusFetchResponse:
        url = f"{self.__base_url}/api/v1/corpus/fetch/{source_id}"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                return CorpusFetchResponse.model_validate(data)

    async def get_market_graph(
        self, theme: str, method: str = "INTERSECTION", as_of: date | None = None
    ) -> MarketGraphResponse:
        params: dict[str, str] = {"method": method}
        if as_of is not None:
            params["as_of"] = as_of.isoformat()

        url = f"{self.__base_url}/api/v1/market-graphs/{quote(theme, safe='')}"
        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return MarketGraphResponse.model_validate(data)
