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


class XUser(BaseModel):
    id: str
    url: str
    name: str
    username: str
    created_at: str
    description: str
    favourites_count: int
    followers_count: int
    followings_count: typing.Optional[int] = None
    listed_count: int
    media_count: int
    profile_image_url: str
    profile_banner_url: str
    statuses_count: int
    verified: bool
    is_blue_verified: bool
    can_dm: bool
    can_media_tag: bool
    location: str
    pinned_tweet_ids: typing.Optional[list[str]] = None


class XPostSummary(BaseModel):
    id: str
    text: str
    reply_count: int
    view_count: int
    retweet_count: int
    like_count: int
    quote_count: int
    bookmark_count: int
    url: str
    created_at: str
    media: list[dict[str, typing.Any]]
    is_quote_tweet: bool
    is_retweet: bool
    lang: str
    conversation_id: str
    in_reply_to_screen_name: typing.Optional[str] = None
    in_reply_to_status_id: typing.Optional[str] = None
    in_reply_to_user_id: typing.Optional[str] = None
    quoted_status_id: typing.Optional[str] = None
    replies: typing.Optional[list[dict[str, typing.Any]]] = None
    display_text_range: typing.Optional[list[int]] = None


class XPostResponse(BaseModel):
    user: XUser
    id: str
    text: str
    reply_count: int
    view_count: int
    retweet_count: int
    like_count: int
    quote_count: int
    bookmark_count: int
    url: str
    created_at: str
    media: list[dict[str, typing.Any]]
    is_quote_tweet: bool
    is_retweet: bool
    lang: str
    conversation_id: str
    in_reply_to_screen_name: typing.Optional[str] = None
    in_reply_to_status_id: typing.Optional[str] = None
    in_reply_to_user_id: typing.Optional[str] = None
    quoted_status_id: typing.Optional[str] = None
    quote: typing.Optional[XPostSummary] = None
    replies: typing.Optional[list[XPostSummary]] = None
    display_text_range: typing.Optional[list[int]] = None
    entities: typing.Optional[dict[str, typing.Any]] = None
    extended_entities: typing.Optional[dict[str, typing.Any]] = None
    retweet: typing.Optional[XPostSummary] = None


class DesearchEndpoint(StrEnum):
    AI_SEARCH = "ai_search"
    AI_WEB_SEARCH = "ai_web_search"
    WEB_SEARCH = "web_search"
    WEB_CRAWL = "web_crawl"
    X_SEARCH = "x_search"
    FETCH_X_POST = "fetch_x_post"


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
    DesearchEndpoint.X_SEARCH: 0.30,
    DesearchEndpoint.FETCH_X_POST: 0.03,
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
