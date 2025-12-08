from datetime import datetime, timezone

import pytest

from neurons.validator.models.weights import WeightsModel


class TestWeightsModel:
    def test_weights_model_creation(self):
        weights = WeightsModel(
            miner_uid=1,
            miner_hotkey="5C4hrfjw9nL7fKRAn3RRKqUWewVE5bGVU7VKF5Ut3N7TkPJJ",
            metagraph_score=0.835,
            aggregated_at=datetime(2025, 1, 30, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert weights.miner_uid == 1
        assert weights.miner_hotkey == "5C4hrfjw9nL7fKRAn3RRKqUWewVE5bGVU7VKF5Ut3N7TkPJJ"
        assert weights.metagraph_score == 0.835
        assert weights.aggregated_at == datetime(2025, 1, 30, 12, 0, 0, tzinfo=timezone.utc)

    def test_weights_model_without_aggregated_at(self):
        weights = WeightsModel(
            miner_uid=2,
            miner_hotkey="5Dpqn...",
            metagraph_score=0.165,
        )

        assert weights.aggregated_at is None

    def test_weights_model_primary_key(self):
        weights = WeightsModel(
            miner_uid=3,
            miner_hotkey="test_hotkey",
            metagraph_score=0.5,
        )

        assert weights.primary_key == ["miner_uid", "miner_hotkey"]

    def test_weights_model_validation(self):
        with pytest.raises(ValueError):
            WeightsModel(miner_uid=1, miner_hotkey="test")
