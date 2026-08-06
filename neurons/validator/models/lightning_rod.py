from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict

from neurons.validator.models.chat_completion import ChatCompletionChoice

INPUT_COST_PER_1M = Decimal("1.0")
OUTPUT_COST_PER_1M = Decimal("6.0")


class LightningRodUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = ConfigDict(extra="allow")


class LightningRodCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[LightningRodUsage] = None

    model_config = ConfigDict(extra="allow")


def calculate_cost(completion: LightningRodCompletion) -> Decimal:
    if not completion.usage:
        return Decimal("0")
    return (
        INPUT_COST_PER_1M * completion.usage.prompt_tokens
        + OUTPUT_COST_PER_1M * completion.usage.completion_tokens
    ) / 1_000_000
