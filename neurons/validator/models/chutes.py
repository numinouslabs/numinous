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


class Chute(BaseModel):
    name: str
    input_cost: float
    output_cost: float

    def calculate_cost(self, completion: ChutesCompletion) -> float:
        return (self.input_cost / 1_000_000) * completion.usage.prompt_tokens + (
            self.output_cost / 1_000_000
        ) * completion.usage.completion_tokens


CHUTES_REGISTRY: dict[ChuteModel, Chute] = {
    ChuteModel.DEEPSEEK_R1_SGTEST: Chute(
        name=ChuteModel.DEEPSEEK_R1_SGTEST,
        input_cost=0.3,
        output_cost=1.2,
    ),
    ChuteModel.DEEPSEEK_R1_0528: Chute(
        name=ChuteModel.DEEPSEEK_R1_0528,
        input_cost=0.4,
        output_cost=1.75,
    ),
    ChuteModel.DEEPSEEK_R1: Chute(
        name=ChuteModel.DEEPSEEK_R1,
        input_cost=0.3,
        output_cost=1.2,
    ),
    ChuteModel.DEEPSEEK_V3_0324: Chute(
        name=ChuteModel.DEEPSEEK_V3_0324,
        input_cost=0.24,
        output_cost=0.84,
    ),
    ChuteModel.DEEPSEEK_V3_1_TERMINUS: Chute(
        name=ChuteModel.DEEPSEEK_V3_1_TERMINUS,
        input_cost=0.23,
        output_cost=0.9,
    ),
    ChuteModel.DEEPSEEK_V3_1: Chute(
        name=ChuteModel.DEEPSEEK_V3_1,
        input_cost=0.20,
        output_cost=0.8,
    ),
    ChuteModel.DEEPSEEK_TNG_R1T2_CHIMERA: Chute(
        name=ChuteModel.DEEPSEEK_TNG_R1T2_CHIMERA,
        input_cost=0.3,
        output_cost=1.2,
    ),
    ChuteModel.DEEPSEEK_V3_2_EXP: Chute(
        name=ChuteModel.DEEPSEEK_V3_2_EXP,
        input_cost=0.25,
        output_cost=0.35,
    ),
    ChuteModel.GLM_4_6: Chute(
        name=ChuteModel.GLM_4_6,
        input_cost=0.4,
        output_cost=1.75,
    ),
    ChuteModel.GLM_4_5: Chute(
        name=ChuteModel.GLM_4_5,
        input_cost=0.35,
        output_cost=1.55,
    ),
    ChuteModel.GLM_4_5_AIR: Chute(
        name=ChuteModel.GLM_4_5_AIR,
        input_cost=0,
        output_cost=0,
    ),
    ChuteModel.GEMMA_3_4B_IT: Chute(
        name=ChuteModel.GEMMA_3_4B_IT,
        input_cost=0,
        output_cost=0,
    ),
    ChuteModel.GEMMA_3_27B_IT: Chute(
        name=ChuteModel.GEMMA_3_27B_IT,
        input_cost=0.13,
        output_cost=0.52,
    ),
    ChuteModel.GEMMA_3_12B_IT: Chute(
        name=ChuteModel.GEMMA_3_12B_IT,
        input_cost=0.03,
        output_cost=0.1,
    ),
    ChuteModel.QWEN3_32B: Chute(
        name=ChuteModel.QWEN3_32B,
        input_cost=0.05,
        output_cost=0.2,
    ),
    ChuteModel.QWEN3_235B_A22B: Chute(
        name=ChuteModel.QWEN3_235B_A22B,
        input_cost=0.3,
        output_cost=1.2,
    ),
    ChuteModel.QWEN2_5_VL_32B_INSTRUCT: Chute(
        name=ChuteModel.QWEN2_5_VL_32B_INSTRUCT,
        input_cost=0.05,
        output_cost=0.22,
    ),
    ChuteModel.QWEN3_235B_A22B_INSTRUCT_2507: Chute(
        name=ChuteModel.QWEN3_235B_A22B_INSTRUCT_2507,
        input_cost=0.08,
        output_cost=0.55,
    ),
    ChuteModel.QWEN3_VL_235B_A22B_THINKING: Chute(
        name=ChuteModel.QWEN3_VL_235B_A22B_THINKING,
        input_cost=0.3,
        output_cost=1.2,
    ),
    ChuteModel.MISTRAL_SMALL_24B_INSTRUCT_2501: Chute(
        name=ChuteModel.MISTRAL_SMALL_24B_INSTRUCT_2501,
        input_cost=0.05,
        output_cost=0.22,
    ),
    ChuteModel.GPT_OSS_20B: Chute(
        name=ChuteModel.GPT_OSS_20B,
        input_cost=0,
        output_cost=0,
    ),
    ChuteModel.GPT_OSS_120B: Chute(
        name=ChuteModel.GPT_OSS_120B,
        input_cost=0.04,
        output_cost=0.4,
    ),
}


def get_chute(model: typing.Union[ChuteModel, str]) -> Chute:
    if isinstance(model, str):
        try:
            model = ChuteModel(model)
        except ValueError:
            available = ", ".join(m.value for m in ChuteModel)
            raise ValueError(f"Model '{model}' is not available. Available models: {available}")

    return CHUTES_REGISTRY[model]


def list_available_models() -> list[str]:
    return [model.value for model in ChuteModel]


def calculate_cost(model: typing.Union[ChuteModel, str], completion: ChutesCompletion) -> float:
    chute = get_chute(model)
    return chute.calculate_cost(completion)
