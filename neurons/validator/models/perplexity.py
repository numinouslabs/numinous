from decimal import Decimal
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class PerplexityMessage(BaseModel):
    role: str
    content: Optional[str] = None


class PerplexityChoice(BaseModel):
    index: int
    message: PerplexityMessage
    finish_reason: Optional[str] = None
    delta: Optional[dict] = None


class PerplexitySearchResult(BaseModel):
    title: str
    url: str
    date: Optional[str] = None
    last_updated: Optional[str] = None
    snippet: str
    source: str


class PerplexityUsageCost(BaseModel):
    input_tokens_cost: Decimal = Decimal("0")
    output_tokens_cost: Decimal = Decimal("0")
    request_cost: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")


class PerplexityUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    search_context_size: Optional[str] = None
    cost: Optional[PerplexityUsageCost] = None


class PerplexityCompletion(BaseModel):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: list[PerplexityChoice]
    usage: Optional[PerplexityUsage] = None
    citations: Optional[list[str]] = None
    search_results: Optional[list[PerplexitySearchResult]] = None


class PerplexityModelName(StrEnum):
    SONAR = "sonar"
    SONAR_PRO = "sonar-pro"
    SONAR_REASONING_PRO = "sonar-reasoning-pro"


class PerplexityModel(BaseModel):
    name: str
    input_cost: Decimal
    output_cost: Decimal

    def calculate_cost_from_tokens(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            (self.input_cost * Decimal(str(input_tokens)))
            + (self.output_cost * Decimal(str(output_tokens)))
        ) / Decimal("1000000")


REQUEST_COSTS: dict[str, Decimal] = {
    "low": Decimal("5.00"),
    "medium": Decimal("8.00"),
    "high": Decimal("12.00"),
}


PERPLEXITY_REGISTRY: dict[PerplexityModelName, PerplexityModel] = {
    PerplexityModelName.SONAR: PerplexityModel(
        name=PerplexityModelName.SONAR,
        input_cost=Decimal("1.00"),
        output_cost=Decimal("1.00"),
    ),
    PerplexityModelName.SONAR_PRO: PerplexityModel(
        name=PerplexityModelName.SONAR_PRO,
        input_cost=Decimal("3.00"),
        output_cost=Decimal("15.00"),
    ),
    PerplexityModelName.SONAR_REASONING_PRO: PerplexityModel(
        name=PerplexityModelName.SONAR_REASONING_PRO,
        input_cost=Decimal("2.00"),
        output_cost=Decimal("8.00"),
    ),
}


def get_perplexity_model(model: str) -> PerplexityModel:
    try:
        model_enum = PerplexityModelName(model)
    except ValueError:
        available = ", ".join(m.value for m in PerplexityModelName)
        raise ValueError(f"Model '{model}' is not available. Available models: {available}")

    return PERPLEXITY_REGISTRY[model_enum]


def calculate_cost(model: str, completion: PerplexityCompletion) -> Decimal:
    if not completion.usage:
        return Decimal("0")

    perplexity_model = get_perplexity_model(model)
    token_cost = perplexity_model.calculate_cost_from_tokens(
        completion.usage.prompt_tokens, completion.usage.completion_tokens
    )

    request_cost = Decimal("0")
    if completion.usage.search_context_size:
        request_cost = REQUEST_COSTS.get(
            completion.usage.search_context_size.lower(), Decimal("0")
        ) / Decimal("1000")

    return token_cost + request_cost
