from decimal import Decimal

import pytest

from neurons.validator.models.perplexity import (
    PERPLEXITY_REGISTRY,
    REQUEST_COSTS,
    PerplexityChoice,
    PerplexityCompletion,
    PerplexityMessage,
    PerplexityModelName,
    PerplexityUsage,
    calculate_cost,
    get_perplexity_model,
)


class TestPerplexityModels:
    def test_perplexity_message(self):
        message = PerplexityMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"

    def test_perplexity_choice(self):
        message = PerplexityMessage(role="assistant", content="Response")
        choice = PerplexityChoice(index=0, message=message, finish_reason="stop")
        assert choice.index == 0
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"

    def test_perplexity_usage(self):
        usage = PerplexityUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            search_context_size="medium",
        )
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 150
        assert usage.search_context_size == "medium"

    def test_perplexity_completion_minimal(self):
        completion = PerplexityCompletion(
            id="cmpl-123",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
        )
        assert completion.id == "cmpl-123"
        assert completion.model == "sonar"
        assert len(completion.choices) == 1

    def test_perplexity_completion_with_usage_and_citations(self):
        completion = PerplexityCompletion(
            id="cmpl-456",
            created=1677652288,
            model="sonar-pro",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=30,
                completion_tokens=80,
                total_tokens=110,
                search_context_size="high",
            ),
            citations=["https://example.com", "https://example2.com"],
        )
        assert completion.usage.prompt_tokens == 30
        assert completion.usage.completion_tokens == 80
        assert len(completion.citations) == 2


class TestPerplexityModelRegistry:
    def test_model_registry_exists(self):
        assert len(PERPLEXITY_REGISTRY) == 3
        assert PerplexityModelName.SONAR in PERPLEXITY_REGISTRY
        assert PerplexityModelName.SONAR_PRO in PERPLEXITY_REGISTRY
        assert PerplexityModelName.SONAR_REASONING_PRO in PERPLEXITY_REGISTRY

    def test_sonar_model_costs(self):
        model = PERPLEXITY_REGISTRY[PerplexityModelName.SONAR]
        assert model.name == "sonar"
        assert model.input_cost == Decimal("1.00")
        assert model.output_cost == Decimal("1.00")

    def test_sonar_pro_model_costs(self):
        model = PERPLEXITY_REGISTRY[PerplexityModelName.SONAR_PRO]
        assert model.name == "sonar-pro"
        assert model.input_cost == Decimal("3.00")
        assert model.output_cost == Decimal("15.00")

    def test_sonar_reasoning_pro_model_costs(self):
        model = PERPLEXITY_REGISTRY[PerplexityModelName.SONAR_REASONING_PRO]
        assert model.name == "sonar-reasoning-pro"
        assert model.input_cost == Decimal("2.00")
        assert model.output_cost == Decimal("8.00")

    def test_get_perplexity_model_valid(self):
        model = get_perplexity_model("sonar")
        assert model.name == "sonar"

        model = get_perplexity_model("sonar-pro")
        assert model.name == "sonar-pro"

    def test_get_perplexity_model_invalid(self):
        with pytest.raises(ValueError, match="not available"):
            get_perplexity_model("invalid-model")


class TestRequestCosts:
    def test_request_costs_exist(self):
        assert "low" in REQUEST_COSTS
        assert "medium" in REQUEST_COSTS
        assert "high" in REQUEST_COSTS

    def test_request_cost_values(self):
        assert REQUEST_COSTS["low"] == Decimal("5.00")
        assert REQUEST_COSTS["medium"] == Decimal("8.00")
        assert REQUEST_COSTS["high"] == Decimal("12.00")


class TestCostCalculation:
    def test_calculate_cost_token_only(self):
        completion = PerplexityCompletion(
            id="cmpl-123",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=1000,
                completion_tokens=2000,
                total_tokens=3000,
            ),
        )

        cost = calculate_cost("sonar", completion)
        expected = (Decimal("1.00") * 1000 + Decimal("1.00") * 2000) / Decimal("1000000")
        assert cost == expected
        assert cost == Decimal("0.003")

    def test_calculate_cost_with_request_cost_low(self):
        completion = PerplexityCompletion(
            id="cmpl-123",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=1000,
                completion_tokens=2000,
                total_tokens=3000,
                search_context_size="low",
            ),
        )

        cost = calculate_cost("sonar", completion)
        token_cost = Decimal("0.003")
        request_cost = Decimal("5.00") / Decimal("1000")
        assert cost == token_cost + request_cost
        assert cost == Decimal("0.008")

    def test_calculate_cost_with_request_cost_medium(self):
        completion = PerplexityCompletion(
            id="cmpl-123",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=1000,
                completion_tokens=2000,
                total_tokens=3000,
                search_context_size="medium",
            ),
        )

        cost = calculate_cost("sonar", completion)
        token_cost = Decimal("0.003")
        request_cost = Decimal("8.00") / Decimal("1000")
        assert cost == token_cost + request_cost
        assert cost == Decimal("0.011")

    def test_calculate_cost_with_request_cost_high(self):
        completion = PerplexityCompletion(
            id="cmpl-123",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=1000,
                completion_tokens=2000,
                total_tokens=3000,
                search_context_size="high",
            ),
        )

        cost = calculate_cost("sonar", completion)
        token_cost = Decimal("0.003")
        request_cost = Decimal("12.00") / Decimal("1000")
        assert cost == token_cost + request_cost
        assert cost == Decimal("0.015")

    def test_calculate_cost_sonar_pro(self):
        completion = PerplexityCompletion(
            id="cmpl-456",
            created=1677652288,
            model="sonar-pro",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=500,
                completion_tokens=1000,
                total_tokens=1500,
                search_context_size="medium",
            ),
        )

        cost = calculate_cost("sonar-pro", completion)
        token_cost = (Decimal("3.00") * 500 + Decimal("15.00") * 1000) / Decimal("1000000")
        request_cost = Decimal("8.00") / Decimal("1000")
        assert cost == token_cost + request_cost
        assert cost == Decimal("0.0245")

    def test_calculate_cost_no_usage(self):
        completion = PerplexityCompletion(
            id="cmpl-789",
            created=1677652288,
            model="sonar",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
        )

        cost = calculate_cost("sonar", completion)
        assert cost == Decimal("0")

    def test_calculate_cost_invalid_model(self):
        completion = PerplexityCompletion(
            id="cmpl-999",
            created=1677652288,
            model="invalid",
            choices=[
                PerplexityChoice(
                    index=0,
                    message=PerplexityMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=PerplexityUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
            ),
        )

        with pytest.raises(ValueError, match="not available"):
            calculate_cost("invalid-model", completion)


class TestModelCostCalculation:
    def test_model_calculate_cost_from_tokens(self):
        model = get_perplexity_model("sonar")
        cost = model.calculate_cost_from_tokens(1000, 2000)
        expected = (Decimal("1.00") * 1000 + Decimal("1.00") * 2000) / Decimal("1000000")
        assert cost == expected
        assert cost == Decimal("0.003")

    def test_model_calculate_cost_from_tokens_sonar_pro(self):
        model = get_perplexity_model("sonar-pro")
        cost = model.calculate_cost_from_tokens(500, 1000)
        expected = (Decimal("3.00") * 500 + Decimal("15.00") * 1000) / Decimal("1000000")
        assert cost == expected
        assert cost == Decimal("0.0165")
