from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from neurons.validator.models.sources import (
    MAX_SOURCE_REASONING_CHARS,
    MAX_SOURCE_URL_CHARS,
    MAX_SOURCES_PER_RUN,
    ImpactBucket,
    PersistenceBucket,
    SourceDirection,
    SourceItem,
    SourcesModel,
)


def _valid_source_kwargs() -> dict:
    return {
        "url": "https://example.com/article",
        "source_type": "news",
        "direction": SourceDirection.UP,
        "source_timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        "impact_bucket": ImpactBucket.HIGH,
        "persistence_bucket": PersistenceBucket.MEDIUM,
        "reasoning": "supports outcome",
    }


class TestSourceItem:
    def test_create_valid(self):
        source = SourceItem(**_valid_source_kwargs())
        assert source.url == "https://example.com/article"
        assert source.direction is SourceDirection.UP
        assert source.impact_bucket is ImpactBucket.HIGH
        assert source.persistence_bucket is PersistenceBucket.MEDIUM

    def test_url_too_long_rejected(self):
        kwargs = _valid_source_kwargs()
        kwargs["url"] = "h" * (MAX_SOURCE_URL_CHARS + 1)

        with pytest.raises(ValidationError):
            SourceItem(**kwargs)

    def test_reasoning_too_long_rejected(self):
        kwargs = _valid_source_kwargs()
        kwargs["reasoning"] = "x" * (MAX_SOURCE_REASONING_CHARS + 1)

        with pytest.raises(ValidationError):
            SourceItem(**kwargs)

    def test_invalid_direction_rejected(self):
        kwargs = _valid_source_kwargs()
        kwargs["direction"] = "sideways"

        with pytest.raises(ValidationError):
            SourceItem(**kwargs)

    def test_invalid_impact_bucket_rejected(self):
        kwargs = _valid_source_kwargs()
        kwargs["impact_bucket"] = "ultra"

        with pytest.raises(ValidationError):
            SourceItem(**kwargs)

    def test_invalid_persistence_bucket_rejected(self):
        kwargs = _valid_source_kwargs()
        kwargs["persistence_bucket"] = "forever"

        with pytest.raises(ValidationError):
            SourceItem(**kwargs)


class TestSourcesModel:
    def test_create_minimal(self):
        model = SourcesModel(run_id="run001", sources="[]")
        assert model.run_id == "run001"
        assert model.sources == "[]"
        assert model.created_at is None
        assert model.updated_at is None
        assert model.exported is False

    def test_create_full(self):
        dt = datetime(2024, 1, 1, 12, 0)
        model = SourcesModel(
            run_id="run002",
            sources='[{"url": "x"}]',
            created_at=dt,
            updated_at=dt,
            exported=True,
        )
        assert model.created_at == dt
        assert model.exported is True

    @pytest.mark.parametrize("exported_input, expected", [(1, True), (0, False)])
    def test_exported_int_to_bool(self, exported_input, expected):
        model = SourcesModel(run_id="run003", sources="[]", exported=exported_input)
        assert model.exported is expected

    def test_primary_key_property(self):
        model = SourcesModel(run_id="run004", sources="[]")
        assert model.primary_key == ["run_id"]


def test_constants():
    assert MAX_SOURCE_URL_CHARS == 2048
    assert MAX_SOURCE_REASONING_CHARS == 2500
    assert MAX_SOURCES_PER_RUN == 15
