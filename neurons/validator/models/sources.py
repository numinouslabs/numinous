import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

MAX_SOURCE_URL_CHARS = 2048
MAX_SOURCE_REASONING_CHARS = 2500
MAX_SOURCES_PER_RUN = 15


class SourceDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class ImpactBucket(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class PersistenceBucket(StrEnum):
    FLASH = "flash"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SourceItem(BaseModel):
    url: str = Field(..., max_length=MAX_SOURCE_URL_CHARS)
    source_type: str
    direction: SourceDirection
    source_timestamp: datetime
    impact_bucket: ImpactBucket
    persistence_bucket: PersistenceBucket
    reasoning: str = Field(..., max_length=MAX_SOURCE_REASONING_CHARS)


class SourcesForExport(BaseModel):
    run_id: str
    sources: list[SourceItem]
    created_at: Optional[datetime] = None
    event_id: str
    miner_uid: int
    miner_hotkey: str
    track: str

    @field_validator("sources", mode="before")
    def parse_sources_json(cls, v: Any) -> Any:
        if isinstance(v, str):
            return json.loads(v)
        return v


class SourcesModel(BaseModel):
    run_id: str
    sources: str
    exported: Optional[bool] = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def primary_key(self):
        return ["run_id"]

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        if isinstance(v, int):
            return bool(v)
        return v


SOURCES_FIELDS = SourcesModel.model_fields.keys()
