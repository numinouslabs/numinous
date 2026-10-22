from datetime import datetime

from pydantic import BaseModel, ConfigDict, JsonValue


class KnownRelationship(BaseModel):
    model_config = ConfigDict(extra="allow")

    head: str
    relation: str
    tail: str


class QuestionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    entity_id: str
    entity: str
    counterparty_id: str | None
    counterparty: str | None
    relationship: KnownRelationship | None


class RoundResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    deadline_at: datetime
    horizon_end_at: datetime
    reveal_round: int
    questions: list[QuestionResponse]


class ReportedEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    description: str
    reported_at: datetime


class StatementStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    statement_position: int
    status: str
    reason: str | None
    event: ReportedEvent | None
    judged_at: datetime | None


class MinerForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    round_id: str
    deadline_at: datetime
    horizon_end_at: datetime
    question: QuestionResponse
    received_at: datetime
    sealed_sha256: str
    state: str
    opened: dict[str, JsonValue] | None
    invalid_reason: str | None
    statements: list[StatementStatus]


class MinerForecastsPage(BaseModel):
    forecasts: list[MinerForecastResponse]
    next_offset: int | None


class ForecastReceipt(BaseModel):
    miner_hotkey: str
    question_id: str
    sealed_sha256: str
    received_at: datetime
    signer_hotkey: str
    signature: str


class Submission(BaseModel):
    round_id: str
    reveal_round: int
    sealed: str
    signature: str
    receipt: ForecastReceipt


class ForecastError(BaseModel):
    path: str
    message: str
