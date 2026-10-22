from datetime import datetime

from pydantic import BaseModel, JsonValue


class DkgQuestionRelationship(BaseModel):
    head: str
    relation: str
    tail: str


class DkgQuestionResponse(BaseModel):
    id: str
    entity_id: str
    entity: str
    counterparty_id: str | None
    counterparty: str | None
    relationship: DkgQuestionRelationship | None


class DkgRoundResponse(BaseModel):
    id: str
    deadline_at: datetime
    horizon_end_at: datetime
    questions: list[DkgQuestionResponse]


class ClosedRoundsPage(BaseModel):
    rounds: list[DkgRoundResponse]
    next_offset: int | None


class ForecastHash(BaseModel):
    miner_uid: int
    miner_hotkey: str
    question_id: str
    sealed_sha256: str
    received_at: datetime


class RoundHashesPage(BaseModel):
    forecasts: list[ForecastHash]
    next_offset: int | None


class StatementResult(BaseModel):
    statement_position: int
    outcome: bool
    reason: str
    event_id: str | None
    reported_at: datetime | None
    judged_at: datetime


class ReleasedForecastResponse(BaseModel):
    miner_uid: int
    miner_hotkey: str
    question_id: str
    deadline_at: datetime
    sealed: str
    signature: str
    opened: dict[str, JsonValue] | None
    invalid_reason: str | None
    results: list[StatementResult]


class ResultsPage(BaseModel):
    forecasts: list[ReleasedForecastResponse]
    next_offset: int | None


class MinerScore(BaseModel):
    miner_uid: int
    miner_hotkey: str
    score: float


class ScoringRun(BaseModel):
    scored_at: datetime
    spec_version: int
    scores: list[MinerScore]


class StoredScores(BaseModel):
    stored: int
