from datetime import datetime
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, field_validator


class AgentRunLogExportedStatus(IntEnum):
    NOT_EXPORTED = 0
    EXPORTED = 1


class AgentRunLogsModel(BaseModel):
    run_id: str
    log_content: str
    exported: Optional[bool] = False
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


AGENT_RUN_LOGS_FIELDS = AgentRunLogsModel.model_fields.keys()
