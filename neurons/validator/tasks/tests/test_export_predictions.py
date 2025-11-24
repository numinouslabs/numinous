from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.numinous_client import PostPredictionsRequestBody
from neurons.validator.models.prediction import PredictionExportedStatus, PredictionsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_predictions import ExportPredictions
from neurons.validator.utils.common.interval import (
    get_interval_iso_datetime,
    get_interval_start_minutes,
)
from neurons.validator.utils.logger.logger import NuminousLogger


class TestExportPredictions:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "ss58_address"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)

        return bt_wallet

    @pytest.fixture
    def export_predictions_task(self, db_operations: DatabaseOperations, bt_wallet: Wallet):
        mocked_logger = MagicMock(spec=NuminousLogger)

        api_client = NuminousClient(env="test", logger=mocked_logger, bt_wallet=bt_wallet)

        return ExportPredictions(
            interval_seconds=60.0,
            db_operations=db_operations,
            api_client=api_client,
            batch_size=1,
            validator_uid=0,
            validator_hotkey="validator_hotkey_test",
            logger=mocked_logger,
        )

    def test_parse_predictions_for_exporting(self, export_predictions_task: ExportPredictions):
        predictions = [
            (
                1,  # ROWID (unused in function)
                "event123",  # unique_event_id
                11,  # miner_uid
                "miner_key_1",  # miner_hotkey
                "weather",  # event_type
                0.75,  # prediction
                120,  # interval_start_minutes
                0.8,  # interval_agg_prediction
                5,  # interval_count
                "2024-01-01 02:00:00",  # submitted
                "123e4567-e89b-12d3-a456-426614174000",  # run_id
                "223e4567-e89b-12d3-a456-426614174001",  # version_id
            )
        ]

        result = export_predictions_task.parse_predictions_for_exporting(
            predictions_db_data=predictions
        )

        assert len(result.submissions) == 1

        prediction = result.submissions[0]

        assert prediction.unique_event_id == "event123"
        assert prediction.provider_type == "weather"
        assert prediction.prediction == 0.75
        assert prediction.interval_start_minutes == 120
        assert prediction.interval_agg_prediction == 0.8
        assert prediction.interval_agg_count == 5
        assert prediction.miner_hotkey == "miner_key_1"
        assert prediction.miner_uid == 11
        assert prediction.validator_hotkey == "validator_hotkey_test"
        assert prediction.validator_uid == 0
        assert prediction.title is None
        assert prediction.outcome is None

        # Verify datetime calculation
        expected_datetime = datetime(2024, 1, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert prediction.interval_datetime == expected_datetime

    def test_parse_predictions_for_exporting_multiple_predictions(
        self, export_predictions_task: ExportPredictions
    ):
        # Test with multiple predictions
        predictions = [
            (
                0,
                "event1",
                1,
                "miner_hotkey_1",
                "weather",
                0.75,
                120,
                0.8,
                5,
                "2024-01-01 02:00:00",
                "323e4567-e89b-12d3-a456-426614174002",
                "423e4567-e89b-12d3-a456-426614174003",
            ),
            (
                0,
                "event2",
                2,
                "miner_hotkey_2",
                "sports",
                0.25,
                240,
                0.3,
                3,
                "2024-01-01 04:00:00",
                "523e4567-e89b-12d3-a456-426614174004",
                "623e4567-e89b-12d3-a456-426614174005",
            ),
        ]

        result = export_predictions_task.parse_predictions_for_exporting(predictions)

        assert len(result.submissions) == 2
        assert result.submissions[0].unique_event_id == "event1"
        assert result.submissions[1].unique_event_id == "event2"

    async def test_run(
        self,
        db_client: DatabaseClient,
        db_operations: DatabaseOperations,
        export_predictions_task: ExportPredictions,
    ):
        """Test the run method when there are predictions to export."""

        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=NuminousClient)

        events = [
            EventsModel(
                unique_event_id="unique_event_id_1",
                event_id="event_1",
                market_type="truncated_market1",
                event_type="market_1",
                description="desc1",
                outcome="outcome1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_2",
                event_id="event_2",
                market_type="truncated_market2",
                event_type="market_2",
                description="desc2",
                outcome="outcome2",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique_event_id_3",
                event_id="event_3",
                market_type="truncated_market3",
                event_type="market_3",
                description="desc3",
                outcome="outcome3",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-02T14:30:00+00:00",
            ),
        ]

        current_interval_minutes = get_interval_start_minutes()
        previous_interval_minutes = current_interval_minutes - 1

        predictions = [
            PredictionsModel(
                unique_event_id="unique_event_id_1",
                miner_hotkey="neuronHotkey_1",
                miner_uid=1,
                latest_prediction=1.0,
                interval_start_minutes=previous_interval_minutes,
                interval_agg_prediction=1.0,
                run_id="723e4567-e89b-12d3-a456-426614174006",
                version_id="823e4567-e89b-12d3-a456-426614174007",
            ),
            PredictionsModel(
                unique_event_id="unique_event_id_2",
                miner_hotkey="neuronHotkey_2",
                miner_uid=2,
                latest_prediction=1.0,
                interval_start_minutes=previous_interval_minutes,
                interval_agg_prediction=1.0,
                run_id="923e4567-e89b-12d3-a456-426614174008",
                version_id="a23e4567-e89b-12d3-a456-426614174009",
            ),
            PredictionsModel(
                unique_event_id="unique_event_id_3",
                miner_hotkey="neuronHotkey_3",
                miner_uid=3,
                latest_prediction=1.0,
                interval_start_minutes=current_interval_minutes,
                interval_agg_prediction=1.0,
                run_id="b23e4567-e89b-12d3-a456-42661417400a",
                version_id="c23e4567-e89b-12d3-a456-42661417400b",
            ),
        ]

        await db_operations.upsert_events(events=events)
        await db_operations.upsert_predictions(predictions=predictions)

        # Act
        await export_predictions_task.run()

        # Assert
        assert export_predictions_task.api_client.post_predictions.call_count == 2

        mock_calls = export_predictions_task.api_client.post_predictions.mock_calls
        first_call = mock_calls[0]
        second_call = mock_calls[1]

        # fetching it early to get submitted = CURRENT_TIMESTAMP from the database
        result = await db_client.many(
            """
                SELECT exported, submitted FROM predictions
            """
        )

        assert first_call == (
            "__call__",
            {
                "body": PostPredictionsRequestBody(
                    submissions=[
                        {
                            "unique_event_id": "unique_event_id_1",
                            "provider_type": "market_1",
                            "prediction": 1.0,
                            "interval_start_minutes": previous_interval_minutes,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": get_interval_iso_datetime(
                                previous_interval_minutes
                            ),
                            "miner_hotkey": "neuronHotkey_1",
                            "miner_uid": 1,
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 0,
                            "title": None,
                            "outcome": None,
                            "submitted_at": result[0][1],
                            "run_id": "723e4567-e89b-12d3-a456-426614174006",
                            "version_id": "823e4567-e89b-12d3-a456-426614174007",
                        }
                    ]
                )
            },
        )
        assert second_call == (
            "__call__",
            {
                "body": PostPredictionsRequestBody(
                    submissions=[
                        {
                            "unique_event_id": "unique_event_id_2",
                            "provider_type": "market_2",
                            "prediction": 1.0,
                            "interval_start_minutes": previous_interval_minutes,
                            "interval_agg_prediction": 1.0,
                            "interval_agg_count": 1,
                            "interval_datetime": get_interval_iso_datetime(
                                previous_interval_minutes
                            ),
                            "miner_hotkey": "neuronHotkey_2",
                            "miner_uid": 2,
                            "validator_hotkey": "validator_hotkey_test",
                            "validator_uid": 0,
                            "title": None,
                            "outcome": None,
                            "submitted_at": result[1][1],
                            "run_id": "923e4567-e89b-12d3-a456-426614174008",
                            "version_id": "a23e4567-e89b-12d3-a456-426614174009",
                        }
                    ]
                )
            },
        )

        assert len(result) == 3
        assert result[0][0] == PredictionExportedStatus.EXPORTED
        assert result[1][0] == PredictionExportedStatus.EXPORTED
        assert result[2][0] == PredictionExportedStatus.NOT_EXPORTED

    async def test_run_no_predictions(self, export_predictions_task: ExportPredictions):
        # Mock API client
        export_predictions_task.api_client = AsyncMock(spec=NuminousClient)

        # Act
        await export_predictions_task.run()

        # Assert
        export_predictions_task.api_client.post_predictions.assert_not_called()
