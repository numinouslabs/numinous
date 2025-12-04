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
)


class NuminousEvent(BaseModel):
    event_id: str
    market_type: str
    title: str
    description: str
    event_metadata: typing.Optional[dict] = None
    created_at: datetime
    cutoff: datetime


class GetEventsResponse(BaseModel):
    count: typing.Optional[int] = None
    items: typing.List[NuminousEvent]

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
    miner_score: float
    miner_effective_score: float
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


class GatewayCall(BaseModel):
    run_id: UUID


class AgentRunSubmission(BaseModel):
    run_id: UUID
    miner_uid: int
    miner_hotkey: str
    vali_uid: int
    vali_hotkey: str
    status: str
    event_id: str
    version_id: UUID
    is_final: bool


class PostAgentRunsRequestBody(BaseModel):
    runs: typing.List[AgentRunSubmission]


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
