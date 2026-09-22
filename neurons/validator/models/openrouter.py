from decimal import Decimal
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue

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


MAX_CHOICE_OPTIONS = 255


class _Question(BaseModel):
    instructions: Union[str, dict[str, JsonValue], list[JsonValue]] = Field(
        ..., description="What to decide, as text, a mapping, or a list of parts"
    )


class NoulQuestion(_Question):
    type: Literal["noul"] = "noul"
    criteria: Optional[dict[Literal["true", "false"], str]] = Field(
        default=None, description="What each outcome means"
    )


class ChoiceQuestion(_Question):
    type: Literal["choice"] = "choice"
    criteria: dict[str, str] = Field(
        ...,
        max_length=MAX_CHOICE_OPTIONS,
        description="Allowed options, keyed by the value returned in `choice`",
    )


class ScoreQuestion(_Question):
    type: Literal["score"] = "score"
    criteria: list[str] = Field(..., description="Ordered level labels, lowest first")


DecisionQuestion = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


class _Upstream(BaseModel):
    model_config = ConfigDict(extra="allow")


class _DistributionAnswer(_Upstream):
    probabilities: dict[str, float] = Field(
        ..., description="Probability per option key or criteria index"
    )
    confidence: float = Field(..., description="Confidence in the selected value")


class NoulAnswer(_Upstream):
    type: Literal["noul"]
    noul: float = Field(..., description="Probability the answer is true")


class ChoiceAnswer(_DistributionAnswer):
    type: Literal["choice"]
    choice: str = Field(..., description="Selected option key")


class ScoreAnswer(_DistributionAnswer):
    type: Literal["score"]
    score: float = Field(..., description="Score on the criteria index scale")
    legend: dict[str, str] = Field(..., description="Criteria label per index, as returned")


DecisionAnswer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")
]


class DecisionUsage(_Upstream):
    input_tokens: int = Field(..., description="Tokens in the state and questions")
    output_tokens: int = Field(..., description="Tokens in the answers")
    cost: Decimal = Field(..., description="Cost in USD from OpenRouter")


class OpenRouterDecision(_Upstream):
    id: str = Field(..., description="Unique decision ID")
    model: str = Field(..., description="Resolved model, including its dated revision")
    provider: str = Field(..., description="Upstream provider that served the request")
    answers: dict[str, DecisionAnswer] = Field(..., description="One answer per question key")
    usage: DecisionUsage = Field(..., description="Token usage stats with cost")


def calculate_decision_cost(decision: OpenRouterDecision) -> Decimal:
    return decision.usage.cost
