from enum import StrEnum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from neurons.validator.models.reforecast_memory import MAX_MEMORY_CHARS
from neurons.validator.models.sources import SourceItem


class AgentInput(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Forecasting question")
    description: Optional[str] = Field(None, description="Additional context")
    cutoff: Optional[str] = Field(None, description="Event cutoff date (ISO 8601)")


class AgentOutput(BaseModel):
    event_id: str = Field(..., description="Event ID this prediction is for")
    prediction: float = Field(..., description="Probability prediction (0.0 to 1.0)", ge=0, le=1)
    reasoning: Optional[str] = Field(None, description="Explanation of prediction")
    sources: Optional[list[SourceItem]] = Field(
        None, description="Optional list of structured sources supporting the prediction"
    )
    memory: Optional[str] = Field(
        None, description="Updated per-(miner, event) memory blob for re-forecasting"
    )

    @field_validator("memory")
    @classmethod
    def _cap_memory(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and len(value) > MAX_MEMORY_CHARS:
            return value[:MAX_MEMORY_CHARS]
        return value


class RunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class AgentRunnerOutput(BaseModel):
    status: RunStatus = Field(..., description="'success' or 'error'")
    output: Optional[Dict[str, Any]] = Field(None, description="Agent output if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    traceback: Optional[str] = Field(None, description="Python traceback if failed")
