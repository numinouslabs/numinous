from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class AgentRunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INTERNAL_AGENT_ERROR = "INTERNAL_AGENT_ERROR"
    INVALID_SANDBOX_OUTPUT = "INVALID_SANDBOX_OUTPUT"
    SANDBOX_TIMEOUT = "SANDBOX_TIMEOUT"


class AgentRunExportedStatus(IntEnum):
    NOT_EXPORTED = 0
    EXPORTED = 1


class IsFinalStatus(IntEnum):
    NOT_FINAL = 0
    IS_FINAL = 1


class AgentRunsModel(BaseModel):
    run_id: str
    unique_event_id: str
    agent_version_id: str
    miner_uid: int
    miner_hotkey: str
    status: AgentRunStatus
    exported: Optional[bool] = False
    is_final: Optional[bool] = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    model_config = {"arbitrary_types_allowed": True}

    @property
    def primary_key(self):
        return ["run_id"]

    @field_validator("exported", mode="before")
    def parse_exported_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v

    @field_validator("is_final", mode="before")
    def parse_is_final_as_bool(cls, v: Any) -> bool:
        # If the DB returns an integer, convert it to boolean
        if isinstance(v, int):
            return bool(v)
        return v


AGENT_RUNS_FIELDS = AgentRunsModel.model_fields.keys()
