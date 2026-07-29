from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, field_validator

MAX_MEMORY_CHARS = 32_768


class ReforecastMemoryForExport(BaseModel):
    run_id: str
    memory: str
    interval_start_minutes: int
    created_at: Optional[datetime] = None
    event_id: str
    miner_uid: int
    miner_hotkey: str


class ReforecastMemoryModel(BaseModel):
    run_id: str
    memory: str
    interval_start_minutes: int
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


REFORECAST_MEMORY_FIELDS = ReforecastMemoryModel.model_fields.keys()
