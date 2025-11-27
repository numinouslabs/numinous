import typing
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ModelEnum(StrEnum):
    NOVA = "NOVA"
    ORBIT = "ORBIT"
    HORIZON = "HORIZON"


class ToolEnum(StrEnum):
    WEB = "web"
    HACKER_NEWS = "hackernews"
    REDDIT = "reddit"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    ARXIV = "arxiv"


class WebToolEnum(StrEnum):
    WEB = "web"
    HACKER_NEWS = "hackernews"
    REDDIT = "reddit"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    ARXIV = "arxiv"


class DateFilterEnum(StrEnum):
    PAST_24_HOURS = "PAST_24_HOURS"
    PAST_2_DAYS = "PAST_2_DAYS"
    PAST_WEEK = "PAST_WEEK"
    PAST_2_WEEKS = "PAST_2_WEEKS"
    PAST_MONTH = "PAST_MONTH"
    PAST_2_MONTHS = "PAST_2_MONTHS"
    PAST_YEAR = "PAST_YEAR"
    PAST_2_YEARS = "PAST_2_YEARS"


class ResultTypeEnum(StrEnum):
    ONLY_LINKS = "ONLY_LINKS"
    LINKS_WITH_SUMMARIES = "LINKS_WITH_SUMMARIES"
    LINKS_WITH_FINAL_SUMMARY = "LINKS_WITH_FINAL_SUMMARY"


class AISearchResponse(BaseModel):
    text: typing.Optional[str] = None
    completion: typing.Optional[str] = None
    wikipedia_search: typing.Optional[list[dict]] = None
    youtube_search: typing.Optional[list[dict]] = None
    arxiv_search: typing.Optional[list[dict]] = None
    reddit_search: typing.Optional[list[dict]] = None
    hacker_news_search: typing.Optional[list[dict]] = None
    tweets: typing.Optional[list[dict]] = None
    miner_link_scores: typing.Optional[dict] = None

    model_config = ConfigDict(extra="allow")


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: typing.Optional[str] = None
    date: typing.Optional[str] = None


class WebSearchResponse(BaseModel):
    data: list[SearchResult]

    model_config = ConfigDict(extra="allow")


class WebLinksResponse(BaseModel):
    youtube_search_results: typing.Optional[list[dict]] = None
    hacker_news_search_results: typing.Optional[list[dict]] = None
    reddit_search_results: typing.Optional[list[dict]] = None
    arxiv_search_results: typing.Optional[list[dict]] = None
    wikipedia_search_results: typing.Optional[list[dict]] = None
    search_results: typing.Optional[list[dict]] = None

    model_config = ConfigDict(extra="allow")


class WebCrawlResponse(BaseModel):
    url: str
    content: str

    model_config = ConfigDict(extra="allow")


class DesearchEndpoint(StrEnum):
    AI_SEARCH = "ai_search"
    AI_WEB_SEARCH = "ai_web_search"
    WEB_SEARCH = "web_search"
    WEB_CRAWL = "web_crawl"


# Cost per 100 searches
DESEARCH_PRICING: typing.Dict[DesearchEndpoint, typing.Any] = {
    DesearchEndpoint.AI_SEARCH: {
        ModelEnum.NOVA: 0.4,
        ModelEnum.ORBIT: 2.2,
        ModelEnum.HORIZON: 2.6,
    },
    DesearchEndpoint.AI_WEB_SEARCH: {
        ModelEnum.NOVA: 0.4,
        ModelEnum.ORBIT: 1.7,
        ModelEnum.HORIZON: 2.1,
    },
    DesearchEndpoint.WEB_SEARCH: 0.25,
    DesearchEndpoint.WEB_CRAWL: 0.05,
}


def calculate_cost(
    endpoint: DesearchEndpoint,
    model: typing.Optional[ModelEnum] = None,
) -> float:
    pricing = DESEARCH_PRICING.get(endpoint)
    if pricing is None:
        raise ValueError(f"No pricing found for endpoint: {endpoint}")

    if isinstance(pricing, dict):
        if model is None:
            raise ValueError(f"Model is required for {endpoint}")
        cost_per_100 = pricing.get(model)
        if cost_per_100 is None:
            available = ", ".join(m.value for m in pricing.keys())
            raise ValueError(
                f"Model '{model}' not available for {endpoint}. " f"Available models: {available}"
            )
    else:
        cost_per_100 = pricing

    return cost_per_100 / 100
