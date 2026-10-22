from datetime import datetime

from pydantic import BaseModel


class DkgRoundModel(BaseModel):
    round_id: str
    deadline_at: datetime
    horizon_end_at: datetime


class DkgQuestionModel(BaseModel):
    question_id: str
    round_id: str
    entity_id: str
    entity: str
    counterparty_id: str | None
    counterparty: str | None
    relationship: str | None


class DkgForecastHashModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    question_id: str
    sealed_sha256: str
    received_at: datetime


class DkgForecastModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    question_id: str
    opened: str | None
    invalid_reason: str | None
    failed_check: str | None


class DkgStatementResultModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    question_id: str
    statement_position: int
    outcome: bool
    reason: str
    event_id: str | None
    reported_at: datetime | None
    judged_at: datetime


class DkgMinerModel(BaseModel):
    miner_uid: int
    miner_hotkey: str


class DkgForecastRightsModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    rights: int
    statements: int


class DkgScoreModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    scored_at: datetime
    score: float
    spec_version: int
