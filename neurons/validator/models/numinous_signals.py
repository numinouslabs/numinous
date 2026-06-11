from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WeightedSignal(BaseModel):
    headline: str
    source: str
    timestamp: datetime
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    impact_score: float = Field(..., ge=0.0, le=1.0)
    direction: Literal["supports_yes", "supports_no", "neutral"]
    rationale: str
    source_url: str | None = None

    model_config = ConfigDict(extra="allow")


class SourceWeight(BaseModel):
    source_name: str
    event_count: int
    avg_relevance_score: float = Field(..., ge=0.0, le=1.0)
    avg_impact_score: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")


class ProcessingMetadata(BaseModel):
    duration_seconds: float
    llm_scored_count: int
    total_ingested_events: int
    question_text: str
    market_yes_price: float | None = None

    model_config = ConfigDict(extra="allow")


class SignalsResponse(BaseModel):
    signals: list[WeightedSignal]
    source_weights: list[SourceWeight]
    total_event_count: int
    filtered_count: int
    failed_sources: list[str] = Field(default_factory=list)
    question_context: str
    processing_metadata: ProcessingMetadata

    model_config = ConfigDict(extra="allow")


class DriverMarket(BaseModel):
    question: str
    yes_price: float
    condition_id: str

    model_config = ConfigDict(extra="allow")


class CausalDriver(BaseModel):
    event_id: str
    title: str
    direction: str
    strength: str
    reasoning: str
    markets: list[DriverMarket] = []
    cluster_source: str | None = None

    model_config = ConfigDict(extra="allow")


class CausalDriveEntry(BaseModel):
    event_id: str
    title: str
    direction: str
    strength: str
    reasoning: str

    model_config = ConfigDict(extra="allow")


class CausalDriversResponse(BaseModel):
    event_id: str
    title: str | None = None
    is_target: bool | None = None
    drivers: list[CausalDriver] | None = None
    drives: list[CausalDriveEntry] | None = None
    found: bool = False

    model_config = ConfigDict(extra="allow")


class DeepResearchReportResponse(BaseModel):
    report: str | None = None
    storyline_name: str | None = None
    research_focus: str | None = None
    topic: str | None = None
    run_date: str | None = None
    matched_via: str | None = None
    market_mappings: list[dict] | None = None

    model_config = ConfigDict(extra="allow")


class CorpusSearchResult(BaseModel):
    source_id: UUID
    url: str
    title: str | None = None
    published_at: datetime | None = None
    snapshot_at: datetime | None = None
    snippet: str

    model_config = ConfigDict(extra="allow")


class CorpusSearchResponse(BaseModel):
    results: list[CorpusSearchResult]

    model_config = ConfigDict(extra="allow")


class CorpusFetchResponse(BaseModel):
    source_id: UUID
    url: str
    title: str | None = None
    content: str
    published_at: datetime | None = None
    snapshot_at: datetime | None = None

    model_config = ConfigDict(extra="allow")


COST_PER_CALL = Decimal("0.001")
CAUSAL_DRIVERS_COST = Decimal("0.0")
DEEP_RESEARCH_COST = Decimal("0.0")
CORPUS_SEARCH_COST = Decimal("0.0")
CORPUS_FETCH_COST = Decimal("0.0")


def calculate_cost() -> Decimal:
    return COST_PER_CALL


def calculate_causal_drivers_cost() -> Decimal:
    return CAUSAL_DRIVERS_COST


def calculate_deep_research_cost() -> Decimal:
    return DEEP_RESEARCH_COST


def calculate_corpus_search_cost() -> Decimal:
    return CORPUS_SEARCH_COST


def calculate_corpus_fetch_cost() -> Decimal:
    return CORPUS_FETCH_COST
