from enum import StrEnum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Forecasting question")
    description: Optional[str] = Field(None, description="Additional context")
    cutoff: Optional[str] = Field(None, description="Event cutoff date (ISO 8601)")


class AgentOutput(BaseModel):
    event_id: str = Field(..., description="Event ID this prediction is for")
    prediction: float = Field(..., description="Probability prediction (0.0 to 1.0)", ge=0, le=1)
    reasoning: Optional[str] = Field(None, description="Explanation of prediction")


class RunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class AgentRunnerOutput(BaseModel):
    status: RunStatus = Field(..., description="'success' or 'error'")
    output: Optional[Dict[str, Any]] = Field(None, description="Agent output if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    traceback: Optional[str] = Field(None, description="Python traceback if failed")
