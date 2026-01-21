from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class OpenAITokensDetails(BaseModel):
    cached_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)


class OpenAIUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[OpenAITokensDetails] = None
    output_tokens_details: Optional[OpenAITokensDetails] = None


class OpenAIOutputContent(BaseModel):
    type: str
    text: Optional[str] = None
    logprobs: Optional[list] = None
    annotations: Optional[list] = None


class OpenAISearchAction(BaseModel):
    type: str
    query: Optional[str] = None
    queries: Optional[list[str]] = None
    url: Optional[str] = None
    pattern: Optional[str] = None


class OpenAIOutputItem(BaseModel):
    id: str
    type: str
    role: Optional[str] = None
    content: Optional[list[OpenAIOutputContent]] = None
    summary: Optional[list] = None
    status: Optional[str] = None
    action: Optional[OpenAISearchAction] = None


class OpenAIResponse(BaseModel):
    id: str
    object: str = Field(default="response")
    created_at: int
    model: str
    output: list[OpenAIOutputItem]
    usage: Optional[OpenAIUsage] = None
    status: Optional[str] = None
    completed_at: Optional[int] = None
    error: Optional[dict] = None


class OpenAIModelName(StrEnum):
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_CHAT_LATEST = "gpt-5.2-chat-latest"
    GPT_5_2_PRO = "gpt-5.2-pro"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"


class OpenAIModel(BaseModel):
    name: str
    input_cost: float
    output_cost: float

    def calculate_cost_from_tokens(self, input_tokens: int, output_tokens: int) -> float:
        return ((self.input_cost * input_tokens) + (self.output_cost * output_tokens)) / 1_000_000


OPENAI_REGISTRY: dict[OpenAIModelName, OpenAIModel] = {
    OpenAIModelName.GPT_5_2: OpenAIModel(
        name=OpenAIModelName.GPT_5_2,
        input_cost=1.75,
        output_cost=14.00,
    ),
    OpenAIModelName.GPT_5_2_CHAT_LATEST: OpenAIModel(
        name=OpenAIModelName.GPT_5_2_CHAT_LATEST,
        input_cost=1.75,
        output_cost=14.00,
    ),
    OpenAIModelName.GPT_5_2_PRO: OpenAIModel(
        name=OpenAIModelName.GPT_5_2_PRO,
        input_cost=21.00,
        output_cost=168.00,
    ),
    OpenAIModelName.GPT_5: OpenAIModel(
        name=OpenAIModelName.GPT_5,
        input_cost=1.25,
        output_cost=10.00,
    ),
    OpenAIModelName.GPT_5_MINI: OpenAIModel(
        name=OpenAIModelName.GPT_5_MINI,
        input_cost=0.25,
        output_cost=2.00,
    ),
    OpenAIModelName.GPT_5_NANO: OpenAIModel(
        name=OpenAIModelName.GPT_5_NANO,
        input_cost=0.05,
        output_cost=0.40,
    ),
}


def get_openai_model(model: str) -> OpenAIModel:
    model_base = model.split("-202")[0] if "-202" in model else model

    try:
        model_enum = OpenAIModelName(model_base)
    except ValueError:
        available = ", ".join(m.value for m in OpenAIModelName)
        raise ValueError(f"Model '{model}' is not available. Available models: {available}")

    return OPENAI_REGISTRY[model_enum]


def count_web_search_calls(response: OpenAIResponse) -> int:
    return sum(
        1
        for item in response.output
        if item.type == "web_search_call" and item.action and item.action.type == "search"
    )


def calculate_cost(model: str, response: OpenAIResponse) -> float:
    if not response.usage:
        return 0.0

    openai_model = get_openai_model(model)
    token_cost = openai_model.calculate_cost_from_tokens(
        response.usage.input_tokens, response.usage.output_tokens
    )
    web_search_count = count_web_search_calls(response)
    web_search_fee = web_search_count * 0.01
    return token_cost + web_search_fee
