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
