from decimal import Decimal

from neurons.validator.models.chat_completion import ChatCompletionChoice, ChatCompletionMessage
from neurons.validator.models.openrouter import (
    OpenRouterCompletion,
    OpenRouterUsage,
    calculate_cost,
)


class TestOpenRouterUsage:
    def test_usage_with_cost(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            cost=Decimal("0.00165"),
        )
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 150
        assert usage.cost == Decimal("0.00165")

    def test_usage_without_cost(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )
        assert usage.cost is None

    def test_usage_allows_extra_fields(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            some_extra_field="value",
        )
        assert usage.model_extra["some_extra_field"] == "value"


class TestOpenRouterCompletion:
    def _make_choice(self, content: str = "Response") -> ChatCompletionChoice:
        return ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
            finish_reason="stop",
        )

    def test_completion_minimal(self):
        completion = OpenRouterCompletion(
            id="gen-123",
            created=1709000000,
            model="anthropic/claude-sonnet-4-6",
            choices=[self._make_choice()],
        )
        assert completion.id == "gen-123"
        assert completion.model == "anthropic/claude-sonnet-4-6"
        assert len(completion.choices) == 1
        assert completion.usage is None

    def test_completion_with_usage(self):
        completion = OpenRouterCompletion(
            id="gen-456",
            created=1709000000,
            model="google/gemini-2.5-flash",
            choices=[self._make_choice()],
            usage=OpenRouterUsage(
                prompt_tokens=30,
                completion_tokens=80,
                total_tokens=110,
                cost=Decimal("0.00053"),
            ),
        )
        assert completion.usage.prompt_tokens == 30
        assert completion.usage.cost == Decimal("0.00053")

    def test_completion_allows_extra_fields(self):
        completion = OpenRouterCompletion(
            id="gen-789",
            created=1709000000,
            model="test",
            choices=[self._make_choice()],
            system_fingerprint="abc",
        )
        assert completion.model_extra["system_fingerprint"] == "abc"


class TestCalculateCost:
    def _make_completion(self, usage=None) -> OpenRouterCompletion:
        return OpenRouterCompletion(
            id="gen-test",
            created=1709000000,
            model="test",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def test_cost_from_usage(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost=Decimal("0.0045"),
            )
        )
        assert calculate_cost(completion) == Decimal("0.0045")

    def test_cost_zero_when_usage_cost_is_none(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
            )
        )
        assert calculate_cost(completion) == Decimal("0")

    def test_cost_zero_when_no_usage(self):
        completion = self._make_completion()
        assert calculate_cost(completion) == Decimal("0")

    def test_cost_zero_value(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=Decimal("0"),
            )
        )
        assert calculate_cost(completion) == Decimal("0")
