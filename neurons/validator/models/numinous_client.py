import typing
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from neurons.validator.models.lightning_rod import LightningRodCompletion
from neurons.validator.models.numinous_indicia import IndiciaSignalsResponse
from neurons.validator.models.numinous_signals import (
    CausalDriversResponse,
    CorpusFetchResponse,
    CorpusSearchResponse,
    DeepResearchReportResponse,
    FeedLanguage,
    NewsFeedArticle,
    NewsOrder,
)
from neurons.validator.models.openai import OpenAIResponse
from neurons.validator.models.openrouter import OpenRouterCompletion
from neurons.validator.models.sources import SourceItem
from neurons.validator.models.track import TrackEnum


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


class MemoryPairKey(BaseModel):
    miner_uid: int
    miner_hotkey: str
    event_id: str


class MemoryPullRequestBody(BaseModel):
    pairs: typing.List[MemoryPairKey] = Field(..., min_length=1)
    interval_datetime: datetime


class MemoryEntry(BaseModel):
    miner_uid: int
    miner_hotkey: str
    event_id: str
    interval_datetime: datetime
    memory: str = Field(..., max_length=32_768)

    model_config = ConfigDict(from_attributes=True)


class MemoryPullResponse(BaseModel):
    items: typing.List[MemoryEntry]
    count: int

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class MemoryCommitRequestBody(BaseModel):
    entries: typing.List[MemoryEntry] = Field(..., min_length=1)


class MemoryCommitResponse(BaseModel):
    inserted: int

    model_config = ConfigDict(from_attributes=True, extra="ignore")


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
    interval_datetime: datetime

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


class GatewayCallResponse(BaseModel):
    cost: float


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


class NuminousIndiciaXOsintRequest(GatewayCall):
    account: typing.Optional[str] = None
    limit: int = Field(default=20, ge=1, le=50)


class NuminousIndiciaLiveuamapRequest(GatewayCall):
    region: typing.Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)


class GatewayNuminousIndiciaSignalsResponse(IndiciaSignalsResponse, GatewayCallResponse):
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


class CorpusSearchRequest(GatewayCall):
    query: str = Field(..., description="Free-text search query over the corpus")
    max_results: int = Field(10, ge=5, le=25, description="Number of results to return")
    published_after: typing.Optional[datetime] = Field(
        None, description="Only return sources published at or after this time"
    )
    published_before: typing.Optional[datetime] = Field(
        None, description="Only return sources published at or before this time"
    )


class GatewayCorpusSearchResponse(CorpusSearchResponse, GatewayCallResponse):
    pass


class CorpusFetchRequest(GatewayCall):
    source_id: UUID = Field(..., description="Corpus source ID to fetch the full snapshot for")


class GatewayCorpusFetchResponse(CorpusFetchResponse, GatewayCallResponse):
    pass


class NewsFeedRequest(GatewayCall):
    event_id: UUID = Field(..., description="Event to pull the news feed for")
    language: FeedLanguage | None = Field(default=None, description="Filter by article language")
    min_impact_score: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Drop articles scoring below this impact on the event",
    )
    order: NewsOrder = Field(default=NewsOrder.RECENT, description="Feed ordering")
    published_within_hours: float | None = Field(
        default=None, gt=0, description="Only articles published within this many hours"
    )
    limit: int = Field(default=20, ge=1, le=30, description="Number of articles to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class GatewayNewsFeedResponse(GatewayCallResponse):
    event_id: UUID
    count: int
    articles: list[NewsFeedArticle]


class MinerWeight(BaseModel):
    miner_uid: int
    miner_hotkey: str
    aggregated_weight: float


class GetWeightsResponse(BaseModel):
    aggregated_at: datetime
    weights: typing.List[MinerWeight]
    count: int
