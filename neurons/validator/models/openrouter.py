from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from neurons.validator.models.chat_completion import ChatCompletionChoice


class OpenRouterUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[Decimal] = None

    model_config = ConfigDict(extra="allow")


class OpenRouterCompletion(BaseModel):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[OpenRouterUsage] = None

    model_config = ConfigDict(extra="allow")


def calculate_cost(completion: OpenRouterCompletion) -> Decimal:
    if completion.usage and completion.usage.cost is not None:
        return completion.usage.cost
    return Decimal("0")
