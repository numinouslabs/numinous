import typing
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from neurons.validator.models.chutes import ChuteModel, ChutesCompletion, Message
from neurons.validator.models.desearch import (
    AISearchResponse,
    DateFilterEnum,
    ModelEnum,
    ResultTypeEnum,
    ToolEnum,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
    WebToolEnum,
    XPostResponse,
    XPostSummary,
)
from neurons.validator.models.lightning_rod import LightningRodCompletion
from neurons.validator.models.lunar_crush import (
    LunarCrushCoinsListResponse,
    LunarCrushNewsResponse,
    LunarCrushPostsResponse,
    LunarCrushTimeSeriesResponse,
    LunarCrushTopicResponse,
    LunarCrushWhatsupResponse,
)
from neurons.validator.models.numinous_indicia import IndiciaSignalsResponse
from neurons.validator.models.numinous_signals import (
    CausalDriversResponse,
    DeepResearchReportResponse,
    SignalsResponse,
)
from neurons.validator.models.openai import OpenAIResponse
from neurons.validator.models.openrouter import OpenRouterCompletion
from neurons.validator.models.perplexity import PerplexityCompletion
from neurons.validator.models.sources import SourceItem
from neurons.validator.models.track import TrackEnum
from neurons.validator.models.unusual_whales import NewsHeadlinesResponse
from neurons.validator.models.vericore import VericoreResponse


class NuminousEvent(BaseModel):
    event_id: str
    market_type: str
    title: str
    description: str
    event_metadata: typing.Optional[dict] = None
    created_at: datetime
    cutoff: datetime
    run_days_before_cutoff: int
    tracks: list[TrackEnum]


class GetEventsResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[NuminousEvent]
    has_more: bool = False

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class NuminousEventDeleted(BaseModel):
    event_id: str
    market_type: str
    created_at: datetime
    deleted_at: datetime


class GetEventsDeletedResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[NuminousEventDeleted]

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class NuminousEventResolved(BaseModel):
    event_id: str
    market_type: str
    created_at: datetime
    answer: int = Field(..., ge=0, le=1)
    resolved_at: datetime
    # No need to type as datetime since is converted to string to persist in the DB
    forecasts: dict[str, float]


class GetEventsResolvedResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[NuminousEventResolved]

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class MinerPrediction(BaseModel):
    unique_event_id: str
    provider_type: str
    track: TrackEnum
    prediction: float
    interval_start_minutes: int
    interval_datetime: datetime
    interval_agg_prediction: float
    interval_agg_count: int
    miner_hotkey: str
    miner_uid: int
    validator_hotkey: str
    validator_uid: int
    submitted_at: datetime
    run_id: UUID
    version_id: UUID

    # To be dropped
    title: typing.Optional[str]
    outcome: typing.Optional[float] = Field(None, ge=0, le=1)

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PostPredictionsRequestBody(BaseModel):
    submissions: typing.Optional[typing.List[MinerPrediction]]

    # To be dropped
    events: typing.Optional[None] = Field(None)


class MinerScore(BaseModel):
    event_id: str
    prediction: float
    answer: float = Field(..., json_schema_extra={"ge": 0, "le": 1})
    miner_hotkey: str
    miner_uid: int
    track: TrackEnum
    miner_score: float
    validator_hotkey: str
    validator_uid: int
    spec_version: typing.Optional[str] = "0.0.0"
    registered_date: typing.Optional[datetime]
    scored_at: typing.Optional[datetime]

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PostScoresRequestBody(BaseModel):
    results: typing.List[MinerScore] = Field(..., min_length=1)


class MinerAgentWithCode(BaseModel):
    version_id: UUID
    miner_hotkey: str
    miner_uid: int
    track: TrackEnum
    agent_name: str
    version_number: int
    created_at: datetime
    code: str

    model_config = ConfigDict(from_attributes=True)


class GetAgentsQueryParams(BaseModel):
    offset: typing.Optional[int] = Field(0, ge=0, description="Pagination offset")
    limit: typing.Optional[int] = Field(50, ge=1, le=100, description="Results per page")


class GetAgentsResponse(BaseModel):
    count: int
    items: typing.List[MinerAgentWithCode]


class PostAgentLogsRequestBody(BaseModel):
    run_id: UUID
    log_content: str = Field(..., max_length=30_000)


class MinerReasoningSubmission(BaseModel):
    event_id: str
    miner_uid: int
    miner_hotkey: str
    track: TrackEnum
    validator_uid: int
    validator_hotkey: str
    run_id: UUID
    reasoning: str = Field(..., max_length=10_000)
    submitted_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PostReasoningRequestBody(BaseModel):
    reasonings: typing.List[MinerReasoningSubmission] = Field(..., min_length=1)


class MinerSourceSubmission(BaseModel):
    event_id: str
    miner_uid: int
    miner_hotkey: str
    track: TrackEnum
    validator_uid: int
    validator_hotkey: str
    run_id: UUID
    submitted_at: datetime
    sources: typing.List[SourceItem] = Field(..., min_length=1, max_length=20)

    model_config = ConfigDict(from_attributes=True)


class PostSourcesRequestBody(BaseModel):
    submissions: typing.List[MinerSourceSubmission] = Field(..., min_length=1)


class GatewayCall(BaseModel):
    run_id: UUID


class AgentRunSubmission(BaseModel):
    run_id: UUID
    miner_uid: int
    miner_hotkey: str
    track: TrackEnum
    vali_uid: int
    vali_hotkey: str
    status: str
    event_id: str
    version_id: UUID
    is_final: bool


class PostAgentRunsRequestBody(BaseModel):
    runs: typing.List[AgentRunSubmission]


class CreateAgentRunRequest(BaseModel):
    miner_uid: int
    miner_hotkey: str
    track: TrackEnum
    vali_uid: int
    vali_hotkey: str
    event_id: str
    version_id: UUID

    model_config = ConfigDict(from_attributes=True)


class CreateAgentRunResponse(BaseModel):
    run_id: UUID


class UpdateAgentRunRequest(BaseModel):
    run_id: UUID
    status: str
    is_final: bool

    model_config = ConfigDict(from_attributes=True)


class BatchUpdateAgentRunsRequest(BaseModel):
    runs: typing.List[UpdateAgentRunRequest]


class ChutesInferenceRequest(GatewayCall):
    model: ChuteModel = Field(..., description="Model to use for inference.")
    messages: list[Message] = Field(..., description="List of chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: typing.Optional[int] = Field(default=None, description="Maximum tokens to generate")
    tools: typing.Optional[list[dict[str, typing.Any]]] = Field(
        default=None, description="Tool definitions for function calling"
    )
    tool_choice: typing.Optional[typing.Any] = Field(
        default=None,
        description="Tool choice setting ('auto', 'required', or specific tool)",
    )

    model_config = ConfigDict(extra="allow", use_enum_values=True)


class DesearchAISearchRequest(GatewayCall):
    prompt: str = Field(..., description="The search query/prompt")
    model: ModelEnum = Field(default=ModelEnum.NOVA, description="Model to use for search")
    tools: list[ToolEnum] = Field(
        default=[ToolEnum.WEB], description="List of tools to use for search"
    )
    date_filter: typing.Optional[DateFilterEnum] = Field(
        default=None, description="Filter results by date range"
    )
    result_type: typing.Optional[ResultTypeEnum] = Field(
        default=None, description="Type of results to return"
    )
    system_message: typing.Optional[str] = Field(
        default=None, description="Optional system message for AI"
    )
    count: int = Field(default=10, ge=10, le=100, description="Number of results")


class DesearchWebLinksRequest(GatewayCall):
    prompt: str = Field(..., description="The search query/prompt")
    model: ModelEnum = Field(default=ModelEnum.NOVA, description="Model to use for search")
    tools: list[WebToolEnum] = Field(
        default=[WebToolEnum.WEB], description="List of web tools to use"
    )
    count: int = Field(default=10, ge=1, le=100, description="Number of links")


class DesearchWebSearchRequest(GatewayCall):
    query: str = Field(..., description="The search query")
    num: int = Field(default=10, ge=1, le=100, description="Number of results")
    start: int = Field(default=0, ge=0, description="Pagination offset")


class DesearchWebCrawlRequest(GatewayCall):
    url: str = Field(..., description="The URL to crawl")


class DesearchXSearchRequest(GatewayCall):
    query: str = Field(..., description="The search query for X posts")
    sort: typing.Optional[typing.Literal["Top", "Latest"]] = Field(
        default="Top", description="Sort order for results"
    )
    user: typing.Optional[str] = Field(default=None, description="Filter by username")
    start_date: typing.Optional[datetime] = Field(
        default=None, description="Filter posts after this date"
    )
    end_date: typing.Optional[datetime] = Field(
        default=None, description="Filter posts before this date"
    )
    lang: typing.Optional[str] = Field(default=None, description="Filter by language code")
    verified: typing.Optional[bool] = Field(default=None, description="Filter by verified status")
    blue_verified: typing.Optional[bool] = Field(
        default=None, description="Filter by blue verified status"
    )
    is_quote: typing.Optional[bool] = Field(default=None, description="Filter for quote tweets")
    is_video: typing.Optional[bool] = Field(default=None, description="Filter for videos")
    is_image: typing.Optional[bool] = Field(default=None, description="Filter for images")
    min_retweets: typing.Optional[int] = Field(
        default=None, description="Minimum retweet count", ge=0
    )
    min_replies: typing.Optional[int] = Field(default=None, description="Minimum reply count", ge=0)
    min_likes: typing.Optional[int] = Field(default=None, description="Minimum like count", ge=0)
    count: int = Field(default=20, ge=1, le=100, description="Number of results")


class DesearchXPostRequest(GatewayCall):
    post_id: str = Field(..., description="The X post ID to fetch")


class GatewayCallResponse(BaseModel):
    cost: float


class GatewayChutesCompletion(ChutesCompletion, GatewayCallResponse):
    pass


class GatewayDesearchAISearchResponse(AISearchResponse, GatewayCallResponse):
    pass


class GatewayDesearchWebLinksResponse(WebLinksResponse, GatewayCallResponse):
    pass


class GatewayDesearchWebSearchResponse(WebSearchResponse, GatewayCallResponse):
    pass


class GatewayDesearchWebCrawlResponse(WebCrawlResponse, GatewayCallResponse):
    pass


class GatewayDesearchXSearchResponse(GatewayCallResponse):
    posts: list[XPostSummary] = Field(..., description="List of X posts")


class GatewayDesearchXPostResponse(XPostResponse, GatewayCallResponse):
    pass


class OpenAIMessage(BaseModel):
    role: str = Field(..., description="Message role: 'developer', 'user', 'assistant', or 'tool'")
    content: typing.Optional[str] = Field(None, description="Message content")
    tool_calls: typing.Optional[list[dict[str, typing.Any]]] = Field(
        None, description="Tool calls made by the model"
    )

    model_config = ConfigDict(extra="allow")


class OpenAIInferenceRequest(GatewayCall):
    model: str = Field(..., description="OpenAI model to use for inference")
    input: list[OpenAIMessage] = Field(..., description="List of input messages")
    temperature: typing.Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_output_tokens: typing.Optional[int] = Field(
        default=None, ge=16, description="Maximum tokens to generate (minimum 16)"
    )
    tools: typing.Optional[list[dict[str, typing.Any]]] = Field(
        default=None, description="Tool definitions (web_search, functions, etc.)"
    )
    tool_choice: typing.Optional[typing.Any] = Field(
        default=None,
        description="Tool choice setting ('auto', 'required', or specific tool)",
    )
    instructions: typing.Optional[str] = Field(
        default=None, description="High-level instructions for model behavior"
    )

    model_config = ConfigDict(extra="allow")


class GatewayOpenAIResponse(OpenAIResponse, GatewayCallResponse):
    pass


class PerplexityMessage(BaseModel):
    role: str
    content: str


class PerplexityInferenceRequest(GatewayCall):
    model: str = Field(..., description="Perplexity model to use")
    messages: list[PerplexityMessage] = Field(..., description="Chat messages")
    temperature: typing.Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: typing.Optional[int] = Field(default=None, description="Maximum tokens to generate")
    search_recency_filter: typing.Optional[str] = Field(
        default=None, description="Search recency filter: day, week, month, year"
    )

    model_config = ConfigDict(extra="allow")


class GatewayPerplexityCompletion(PerplexityCompletion, GatewayCallResponse):
    pass


class OpenRouterMessage(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: typing.Optional[typing.Union[str, list]] = Field(
        "", description="Message content (can be None for tool calls)"
    )

    model_config = ConfigDict(extra="allow")


class OpenRouterInferenceRequest(GatewayCall):
    model: str = Field(..., description="OpenRouter model ID (e.g. anthropic/claude-sonnet-4-6)")
    messages: list[OpenRouterMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: typing.Optional[int] = Field(default=None, description="Maximum tokens to generate")
    tools: typing.Optional[list[dict[str, typing.Any]]] = Field(
        default=None, description="Tool definitions for function calling"
    )
    tool_choice: typing.Optional[typing.Any] = Field(
        default=None,
        description="Tool choice setting ('auto', 'required', or specific tool)",
    )

    model_config = ConfigDict(extra="allow")


class GatewayOpenRouterCompletion(OpenRouterCompletion, GatewayCallResponse):
    pass


class LightningRodMessage(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: typing.Optional[typing.Union[str, list]] = Field(
        "", description="Message content (can be None for tool calls)"
    )

    model_config = ConfigDict(extra="allow")


class LightningRodInferenceRequest(GatewayCall):
    model: str = Field(
        default="LightningRodLabs/foresight-v3",
        description="Lightning Rod model ID",
    )
    messages: list[LightningRodMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: typing.Optional[int] = Field(default=None, description="Maximum tokens to generate")

    model_config = ConfigDict(extra="allow")


class GatewayLightningRodCompletion(LightningRodCompletion, GatewayCallResponse):
    pass


class VericoreCalculateRatingRequest(GatewayCall):
    statement: str = Field(..., description="Statement to verify")
    generate_preview: bool = Field(default=False, description="Generate a preview URL")


class GatewayVericoreResponse(VericoreResponse, GatewayCallResponse):
    pass


class NuminousIndiciaXOsintRequest(GatewayCall):
    account: typing.Optional[str] = None
    limit: int = Field(default=20, ge=1, le=50)


class NuminousIndiciaLiveuamapRequest(GatewayCall):
    region: typing.Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)


class GatewayNuminousIndiciaSignalsResponse(IndiciaSignalsResponse, GatewayCallResponse):
    pass


class LunarCrushTopicRequest(GatewayCall):
    topic: str


class LunarCrushTimeSeriesRequest(GatewayCall):
    topic: str
    bucket: str = "day"


class LunarCrushNewsRequest(GatewayCall):
    topic: str


class LunarCrushWhatsupRequest(GatewayCall):
    topic: str


class LunarCrushPostsRequest(GatewayCall):
    topic: str
    start: typing.Optional[int] = None
    end: typing.Optional[int] = None


class LunarCrushCoinsListRequest(GatewayCall):
    pass


class GatewayLunarCrushTopicResponse(LunarCrushTopicResponse, GatewayCallResponse):
    pass


class GatewayLunarCrushTimeSeriesResponse(LunarCrushTimeSeriesResponse, GatewayCallResponse):
    pass


class GatewayLunarCrushNewsResponse(LunarCrushNewsResponse, GatewayCallResponse):
    pass


class GatewayLunarCrushWhatsupResponse(LunarCrushWhatsupResponse, GatewayCallResponse):
    pass


class GatewayLunarCrushPostsResponse(LunarCrushPostsResponse, GatewayCallResponse):
    pass


class GatewayLunarCrushCoinsListResponse(LunarCrushCoinsListResponse, GatewayCallResponse):
    pass


class NuminousSignalsRequest(GatewayCall):
    market: typing.Optional[str] = Field(None, description="Polymarket URL, slug, or condition ID")
    question: typing.Optional[str] = Field(None, description="Free-text question")
    relevance_threshold: float = Field(0.25, ge=0.0, le=1.0)
    max_events_per_source: int = Field(25, ge=1, le=100)
    time_window_hours: int = Field(72, ge=1, le=720)


class GatewayNuminousSignalsResponse(SignalsResponse, GatewayCallResponse):
    pass


class CausalDriversRequest(GatewayCall):
    event_id: str = Field(..., description="Event ID to look up causal drivers for")
    topic: str = Field(default="geopolitics", description="Topic for causal graph lookup")


class GatewayCausalDriversResponse(CausalDriversResponse, GatewayCallResponse):
    pass


class DeepResearchReportRequest(GatewayCall):
    event_id: typing.Optional[str] = Field(None, description="Event ID to match report")
    polymarket_market_id: typing.Optional[str] = Field(
        None, description="Polymarket market/condition ID"
    )
    title: typing.Optional[str] = Field(None, description="Market title for fuzzy matching")
    topics: typing.Optional[list[str]] = Field(None, description="Topics to narrow title matching")


class GatewayDeepResearchReportResponse(DeepResearchReportResponse, GatewayCallResponse):
    pass


class UnusualWhalesHeadlinesRequest(GatewayCall):
    sources: typing.Optional[str] = Field(None, description="Comma-separated news sources")
    search_term: typing.Optional[str] = Field(None, description="Search term to filter headlines")
    ticker: typing.Optional[str] = Field(None, description="Ticker symbol to filter headlines")
    major_only: typing.Optional[bool] = Field(None, description="Only return major headlines")
    limit: int = Field(50, ge=1, le=200, description="Number of headlines per page")
    page: int = Field(0, ge=0, description="Page number for pagination")


class GatewayUnusualWhalesHeadlinesResponse(NewsHeadlinesResponse, GatewayCallResponse):
    pass


class PublicDataSourceListItem(BaseModel):
    name: str
    domain: str
    base_url: str | None = None
    category: str
    requires_auth: bool


class PublicDataSourceListResponse(BaseModel):
    sources: list[PublicDataSourceListItem]


class PublicDataProxyRequest(GatewayCall):
    url: str
    method: typing.Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    timeout: float = Field(default=30.0, ge=1.0, le=60.0)


class GatewayPublicDataProxyResponse(GatewayCallResponse):
    status_code: int
    response_headers: dict[str, str]
    response_body: str
    content_type: str | None = None
    source_name: str
    source_category: str


class MinerWeight(BaseModel):
    miner_uid: int
    miner_hotkey: str
    aggregated_weight: float


class GetWeightsResponse(BaseModel):
    aggregated_at: datetime
    weights: typing.List[MinerWeight]
    count: int
