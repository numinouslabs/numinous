import typing
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ChuteModel(StrEnum):
    # DeepSeek models
    DEEPSEEK_R1_SGTEST = "deepseek-ai/DeepSeek-R1-sgtest"
    DEEPSEEK_R1_0528 = "deepseek-ai/DeepSeek-R1-0528"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
    DEEPSEEK_V3_0324 = "deepseek-ai/DeepSeek-V3-0324"
    DEEPSEEK_V3_1_TERMINUS = "deepseek-ai/DeepSeek-V3.1-Terminus"
    DEEPSEEK_V3_1 = "deepseek-ai/DeepSeek-V3.1"
    DEEPSEEK_TNG_R1T2_CHIMERA = "tngtech/DeepSeek-TNG-R1T2-Chimera"
    DEEPSEEK_V3_2_EXP = "deepseek-ai/DeepSeek-V3.2-Exp"

    # Gemma models
    GEMMA_3_4B_IT = "unsloth/gemma-3-4b-it"
    GEMMA_3_27B_IT = "unsloth/gemma-3-27b-it"
    GEMMA_3_12B_IT = "unsloth/gemma-3-12b-it"

    # Zai models
    GLM_4_6 = "zai-org/GLM-4.6"
    GLM_4_5 = "zai-org/GLM-4.5"
    GLM_4_5_AIR = "zai-org/GLM-4.5-Air"

    # Qwen models
    QWEN3_32B = "Qwen/Qwen3-32B"
    QWEN2_5_VL_32B_INSTRUCT = "Qwen/Qwen2.5-VL-32B-Instruct"
    QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"
    QWEN3_235B_A22B_INSTRUCT_2507 = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    QWEN3_VL_235B_A22B_THINKING = "Qwen/Qwen3-VL-235B-A22B-Thinking"

    # Mistral models
    MISTRAL_SMALL_24B_INSTRUCT_2501 = "unsloth/Mistral-Small-24B-Instruct-2501"

    # OpenAI models
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/gpt-oss-120b"


class Message(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: typing.Optional[typing.Union[str, list]] = Field(
        "", description="Message content (can be None for tool calls)"
    )

    model_config = ConfigDict(extra="allow")


class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Role of the message author")
    content: typing.Optional[str] = Field(None, description="Message content")
    tool_calls: typing.Optional[list[dict[str, typing.Any]]] = Field(
        None, description="Tool calls made by the model"
    )

    model_config = ConfigDict(extra="allow")


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    message: ChatCompletionMessage = Field(..., description="Chat completion message")
    finish_reason: typing.Optional[str] = Field(None, description="Reason for completion stop")

    model_config = ConfigDict(extra="allow")


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int = Field(..., description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")

    model_config = ConfigDict(extra="allow")


class ChutesCompletion(BaseModel):
    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChoice] = Field(..., description="List of choices")
    usage: typing.Optional[ChatCompletionUsage] = Field(None, description="Token usage stats")

    model_config = ConfigDict(extra="allow")


class ChuteStatus(BaseModel):
    chute_id: str
    name: str
    timestamp: datetime
    utilization_current: float
    utilization_5m: float
    utilization_15m: float
    utilization_1h: float
    rate_limit_ratio_5m: float
    rate_limit_ratio_15m: float
    rate_limit_ratio_1h: float
    total_requests_5m: float
    total_requests_15m: float
    total_requests_1h: float
    completed_requests_5m: float
    completed_requests_15m: float
    completed_requests_1h: float
    rate_limited_requests_5m: float
    rate_limited_requests_15m: float
    rate_limited_requests_1h: float
    instance_count: int
    action_taken: str
    target_count: int
    total_instance_count: int
    active_instance_count: int
    scalable: bool
    scale_allowance: int
    avg_busy_ratio: float
    total_invocations: float
    total_rate_limit_errors: float
