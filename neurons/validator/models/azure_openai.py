from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class AzureOpenAITokensDetails(BaseModel):
    cached_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)


class AzureOpenAIUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[AzureOpenAITokensDetails] = None
    output_tokens_details: Optional[AzureOpenAITokensDetails] = None


class AzureOpenAIOutputContent(BaseModel):
    type: str
    text: Optional[str] = None
    logprobs: Optional[list] = None
    annotations: Optional[list] = None


class AzureOpenAISearchAction(BaseModel):
    type: str
    query: Optional[str] = None
    queries: Optional[list[str]] = None
    url: Optional[str] = None
    pattern: Optional[str] = None


class AzureOpenAIOutputItem(BaseModel):
    id: str
    type: str
    role: Optional[str] = None
    content: Optional[list[AzureOpenAIOutputContent]] = None
    summary: Optional[list] = None
    status: Optional[str] = None
    action: Optional[AzureOpenAISearchAction] = None


class AzureOpenAIResponse(BaseModel):
    id: str
    object: str = Field(default="response")
    created_at: int
    model: str
    output: list[AzureOpenAIOutputItem]
    usage: Optional[AzureOpenAIUsage] = None
    status: Optional[str] = None
    completed_at: Optional[int] = None
    error: Optional[dict] = None


class AzureOpenAIModelName(StrEnum):
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_CHAT_LATEST = "gpt-5.2-chat-latest"
    GPT_5_2_PRO = "gpt-5.2-pro"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"


class AzureOpenAIModel(BaseModel):
    name: str
    input_cost: float
    output_cost: float

    def calculate_cost_from_tokens(self, input_tokens: int, output_tokens: int) -> float:
        return ((self.input_cost * input_tokens) + (self.output_cost * output_tokens)) / 1_000_000


AZURE_OPENAI_REGISTRY: dict[AzureOpenAIModelName, AzureOpenAIModel] = {
    AzureOpenAIModelName.GPT_5_2: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5_2,
        input_cost=1.75,
        output_cost=14.00,
    ),
    AzureOpenAIModelName.GPT_5_2_CHAT_LATEST: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5_2_CHAT_LATEST,
        input_cost=1.75,
        output_cost=14.00,
    ),
    AzureOpenAIModelName.GPT_5_2_PRO: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5_2_PRO,
        input_cost=21.00,
        output_cost=168.00,
    ),
    AzureOpenAIModelName.GPT_5: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5,
        input_cost=1.25,
        output_cost=10.00,
    ),
    AzureOpenAIModelName.GPT_5_MINI: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5_MINI,
        input_cost=0.25,
        output_cost=2.00,
    ),
    AzureOpenAIModelName.GPT_5_NANO: AzureOpenAIModel(
        name=AzureOpenAIModelName.GPT_5_NANO,
        input_cost=0.05,
        output_cost=0.40,
    ),
}


def get_azure_openai_model(model: str) -> AzureOpenAIModel:
    model_base = model.split("-202")[0] if "-202" in model else model

    try:
        model_enum = AzureOpenAIModelName(model_base)
    except ValueError:
        available = ", ".join(m.value for m in AzureOpenAIModelName)
        raise ValueError(f"Model '{model}' is not available. Available models: {available}")

    return AZURE_OPENAI_REGISTRY[model_enum]


def count_web_search_calls(response: AzureOpenAIResponse) -> int:
    return sum(
        1
        for item in response.output
        if item.type == "web_search_call" and item.action and item.action.type == "search"
    )


def calculate_cost(model: str, response: AzureOpenAIResponse) -> float:
    if not response.usage:
        return 0.0

    azure_model = get_azure_openai_model(model)
    token_cost = azure_model.calculate_cost_from_tokens(
        response.usage.input_tokens, response.usage.output_tokens
    )
    web_search_count = count_web_search_calls(response)
    web_search_fee = web_search_count * 0.01
    return token_cost + web_search_fee