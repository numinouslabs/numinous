import pytest
from pydantic import ValidationError

from neurons.validator.sandbox.agent_models import AgentOutput


def _valid_source() -> dict:
    return {
        "url": "https://example.com",
        "source_type": "news",
        "direction": "up",
        "source_timestamp": "2024-01-01T12:00:00+00:00",
        "impact_bucket": "high",
        "persistence_bucket": "medium",
        "reasoning": "supports outcome",
    }


class TestAgentOutputSources:
    def test_accepts_none_sources(self):
        output = AgentOutput(event_id="e1", prediction=0.5)
        assert output.sources is None

    def test_accepts_empty_list(self):
        output = AgentOutput(event_id="e1", prediction=0.5, sources=[])
        assert output.sources == []

    def test_accepts_valid_sources(self):
        output = AgentOutput(
            event_id="e1",
            prediction=0.5,
            sources=[_valid_source(), _valid_source()],
        )
        assert len(output.sources) == 2
        assert output.sources[0].url == "https://example.com"

    def test_rejects_invalid_enum(self):
        bad = _valid_source()
        bad["impact_bucket"] = "ultra"

        with pytest.raises(ValidationError):
            AgentOutput(event_id="e1", prediction=0.5, sources=[bad])

    def test_rejects_missing_required_field(self):
        bad = _valid_source()
        del bad["url"]

        with pytest.raises(ValidationError):
            AgentOutput(event_id="e1", prediction=0.5, sources=[bad])

    def test_rejects_non_list_sources(self):
        with pytest.raises(ValidationError):
            AgentOutput(event_id="e1", prediction=0.5, sources="not-a-list")

    def test_rejects_non_dict_item(self):
        with pytest.raises(ValidationError):
            AgentOutput(event_id="e1", prediction=0.5, sources=["just-a-string"])
