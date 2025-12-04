from enum import Enum
from typing import Any, Callable, Dict, Optional

from docker.models.containers import Container
from pydantic import BaseModel, ConfigDict, Field


class SandboxErrorType(str, Enum):
    TIMEOUT = "timeout"
    CONTAINER_ERROR = "container_error"
    INVALID_OUTPUT = "invalid_output"
    AGENT_ERROR = "agent_error"


class SandboxState(BaseModel):
    temp_dir: str = Field(..., description="Path to temporary directory")
    run_id: str = Field(..., description="Unique run ID for this sandbox")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    on_finish: Callable[[Dict[str, Any]], None] = Field(..., description="Callback when done")
    timeout: int = Field(..., description="Timeout in seconds")
    start_time: float = Field(..., description="Unix timestamp when sandbox started")
    container: Optional[Container] = Field(default=None, description="Docker container instance")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SandboxResult(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Agent output if successful")
    logs: str = Field(default="", description="Container logs")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    traceback: Optional[str] = Field(default=None, description="Python traceback if failed")
    error_type: Optional[SandboxErrorType] = Field(
        default=None,
        description="Error category from SandboxErrorType enum",
    )
