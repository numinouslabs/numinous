import copy
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from bittensor import AsyncSubtensor
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner import MinersModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.tasks.scoring import (
    CLIP_EPS,
    DEFAULT_POWER_DECAY_WEIGHT_EXPONENT,
    ScoreNames,
    Scoring,
)
from neurons.validator.utils.common.interval import (
    AGGREGATION_INTERVAL_LENGTH_MINUTES,
    SCORING_WINDOW_INTERVALS,
    align_to_interval,
    minutes_since_epoch,
    to_utc,
)
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import NuminousLogger


class TestScoring:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def scoring_task(
        self,
        db_operations: DatabaseOperations,
    ):
        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.sync = AsyncMock()

        # Mock metagraph attributes
        metagraph.uids = np.array([1, 2, 3], dtype=np.int64)
        metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]

        logger = MagicMock(spec=NuminousLogger)

        # Mock subtensor methods
        subtensor_cm = AsyncMock(spec=AsyncSubtensor)

        subtensor_cm.metagraph = AsyncMock(return_value=metagraph)

        subtensor_cm.__aenter__ = AsyncMock(return_value=subtensor_cm)
        subtensor_cm.__aexit__ = AsyncMock(return_value=False)

        with freeze_time("2024-12-27 07:00:00"):
            return Scoring(
                interval_seconds=60.0,
                db_operations=db_operations,
                netuid=99,
                subtensor=subtensor_cm,
                logger=logger,
            )

    def test_copy_metagraph_state(self, scoring_task: Scoring):
        unit = scoring_task

        unit.metagraph = AsyncMock()

        unit.metagraph.uids = np.array([1, 2, 3, 4], dtype=np.int64)
        unit.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]

        unit.copy_metagraph_state()
        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3, 4]
        assert unit.current_miners_df.miner_hotkey.tolist() == [
            "hotkey1",
            "hotkey2",
            "hotkey3",
            "hotkey4",
        ]
        assert unit.current_miners_df.index.size == 4
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]
        assert unit.current_uids.tolist() == [1, 2, 3, 4]
        assert unit.n_hotkeys == 4

    @pytest.mark.parametrize(
        "db_rows, current_miners_df, expected_result, expected_log",
        [
            # Test case 1: No DB rows
            # – expect False with a "no miners" error.
            (
                [],  # Empty list for DB rows.
                pd.DataFrame({ScoreNames.miner_uid: [1], ScoreNames.miner_hotkey: ["hot1"]}),
                False,
                "No miners found in the DB, skipping scoring!",
            ),
            # Test case 2: DB rows exist but no overlap with current_miners_df
            # – expect False with an overlap error.
            (
                [
                    {
                        ScoreNames.miner_uid: "1",  # Note: stored as a string in the DB.
                        ScoreNames.miner_hotkey: "hot1",
                        ScoreNames.registered_date: pd.Timestamp("2025-01-01"),
                        "is_validating": False,
                        "validator_permit": False,
                    }
                ],
                # current_miners_df has different values so the merge will be empty.
                pd.DataFrame({ScoreNames.miner_uid: [2], ScoreNames.miner_hotkey: ["hot2"]}),
                False,
                "No overlap in miners between DB and metagraph, skipping scoring!",
            ),
            # Test case 3: DB rows exist and there is overlap
            # – expect True and computed registered minutes.
            (
                [
                    {
                        ScoreNames.miner_uid: "1",
                        ScoreNames.miner_hotkey: "hot1",
                        ScoreNames.registered_date: pd.Timestamp("2025-01-01"),
                        "is_validating": False,
                        "validator_permit": False,
                    }
                ],
                pd.DataFrame(
                    {
                        ScoreNames.miner_uid: [1],
                        ScoreNames.miner_hotkey: ["hot1"],
                        "other": ["data"],
                    }
                ),
                True,
                None,  # No error is expected.
            ),
        ],
    )
    async def test_miners_last_reg_sync(
        self,
        scoring_task: Scoring,
        db_rows,
        current_miners_df,
        expected_result,
        expected_log,
    ):
        unit = scoring_task
        unit.current_miners_df = current_miners_df

        miner_db_rows = [MinersModel(**dict(row)) for row in db_rows]
        with patch.object(
            unit.db_operations,
            "get_miners_last_registration",
            new=AsyncMock(return_value=miner_db_rows),
        ):
            result = await unit.miners_last_reg_sync()

        assert result == expected_result

        if expected_log is not None:
            assert unit.logger.error.call_args_list[0].args[0] == expected_log
            assert unit.logger.error.call_count == 1
        else:
            # When the merge is successful, verify that the computed column exists.
            assert unit.miners_last_reg is not None
            assert ScoreNames.miner_registered_minutes in unit.miners_last_reg.columns

            expected_minutes = minutes_since_epoch(to_utc(pd.Timestamp("2025-01-01")))
            computed_minutes = unit.miners_last_reg[ScoreNames.miner_registered_minutes].iloc[0]
            assert computed_minutes == expected_minutes

    def test_set_right_cutoff(self):
        input_event = EventsModel(
            unique_event_id="1",
            event_id="e1",
            market_type="dummy_market",
            event_type="dummy_type",
            description="dummy description",
            status=2,
            metadata="{}",
            registered_date=datetime(2025, 1, 1, 9, 0, 0),
            cutoff=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            resolved_at=datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
        )

        original_event = copy.deepcopy(input_event)

        with freeze_time("2025-01-01 12:00:00"):
            # Compute the effective cutoff.
            effective_cutoff = min(
                to_utc(input_event.cutoff),
                to_utc(input_event.resolved_at),
                datetime.now(timezone.utc),
            )
            result_event = Scoring.set_right_cutoff(input_event)

        assert result_event.cutoff == effective_cutoff
        assert result_event.resolved_at == effective_cutoff
        # Assert that registered_date has been converted using to_utc.
        assert result_event.registered_date == to_utc(original_event.registered_date)

        # Verify that the input event was not modified.
        assert input_event.cutoff == original_event.cutoff
        assert input_event.resolved_at == original_event.resolved_at
        assert input_event.registered_date == original_event.registered_date

    def test_get_intervals_df_no_intervals(self, scoring_task: Scoring):
        unit = scoring_task
        event_registered_start_minutes = 100
        event_cutoff_start_minutes = 100

        # Call the method.
        intervals_df = unit.get_intervals_df(
            event_registered_start_minutes, event_cutoff_start_minutes
        )

        # Verify that an error was logged.
        assert unit.logger.error.call_count == 1
        logged_args = unit.logger.error.call_args_list[0].args
        assert "n_intervals computed to be <= 0" in logged_args[0]

        # Verify that the returned DataFrame has the expected columns and is empty.
        expected_columns = [
            ScoreNames.interval_idx,
            ScoreNames.interval_start,
            ScoreNames.interval_end,
            ScoreNames.weight,
        ]
        assert list(intervals_df.columns) == expected_columns
        assert intervals_df.empty

    def test_power_decay_weight(self, scoring_task: Scoring):
        n_intervals = 1
        weight = scoring_task.power_decay_weight(0, n_intervals)

        assert weight == 1

        n_intervals = 10

        for idx in range(0, n_intervals):
            expected_weight = 1 - (idx / (n_intervals - 1)) ** DEFAULT_POWER_DECAY_WEIGHT_EXPONENT
            weight = scoring_task.power_decay_weight(idx, n_intervals)

            assert weight == expected_weight

    def test_get_intervals_df_success(self, scoring_task: Scoring):
        unit = scoring_task
        event_registered_start_minutes = 0
        event_cutoff_start_minutes = 4 * AGGREGATION_INTERVAL_LENGTH_MINUTES

        unit.errors_count = 0
        unit.logger.error.reset_mock()

        intervals_df = unit.get_intervals_df(
            event_registered_start_minutes, event_cutoff_start_minutes
        )

        assert unit.errors_count == 0
        assert unit.logger.error.call_count == 0

        n_intervals = (
            event_cutoff_start_minutes - event_registered_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES

        assert n_intervals == 4
        assert intervals_df.shape[0] == n_intervals

        expected_columns = [
            ScoreNames.interval_idx,
            ScoreNames.interval_start,
            ScoreNames.interval_end,
            ScoreNames.weight,
        ]
        assert list(intervals_df.columns) == expected_columns

        for idx, row in intervals_df.iterrows():
            assert row[ScoreNames.interval_idx] == idx

            expected_interval_start = (
                event_registered_start_minutes + idx * AGGREGATION_INTERVAL_LENGTH_MINUTES
            )
            expected_interval_end = expected_interval_start + AGGREGATION_INTERVAL_LENGTH_MINUTES
            assert row[ScoreNames.interval_start] == expected_interval_start
            assert row[ScoreNames.interval_end] == expected_interval_end

            expected_weight = 1 - (idx / (n_intervals - 1)) ** DEFAULT_POWER_DECAY_WEIGHT_EXPONENT
            assert row[ScoreNames.weight] == expected_weight

    def test_prepare_predictions_df_no_valid_miner(self, scoring_task: Scoring):
        prediction = PredictionsModel(
            unique_event_id="ev1",
            miner_hotkey="hotkey3",
            miner_uid=3,
            latest_prediction=1,
            interval_start_minutes=100,
            interval_agg_prediction=0.5,
            interval_count=1,
            submitted=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            exported=False,
        )
        predictions = [prediction]

        miners = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
            }
        )

        result_df = scoring_task.prepare_predictions_df(predictions, miners)
        assert result_df.shape[0] == miners.shape[0]
        assert ScoreNames.interval_agg_prediction in result_df.columns
        assert result_df[ScoreNames.interval_agg_prediction].isna().all()

    def test_prepare_predictions_df_with_valid_miners(self, scoring_task: Scoring):
        prediction = PredictionsModel(
            unique_event_id="ev1",
            miner_hotkey="hotkey1",
            miner_uid=1,
            latest_prediction=1,
            interval_start_minutes=100,
            # Use a value that may require clipping;
            interval_agg_prediction=0.005,
            interval_count=1,
            submitted=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            exported=False,
        )
        predictions = [prediction]

        miners = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
            }
        )
        result_df = scoring_task.prepare_predictions_df(predictions, miners)
        assert result_df.shape[0] == 2

        row1 = result_df[result_df[ScoreNames.miner_uid] == 1].iloc[0]
        assert row1[ScoreNames.interval_start] == 100
        assert row1[ScoreNames.miner_hotkey] == "hotkey1"

        expected_clipped = max(CLIP_EPS, min(0.005, 1 - CLIP_EPS))
        np.testing.assert_almost_equal(row1[ScoreNames.interval_agg_prediction], expected_clipped)

        row2 = result_df[result_df[ScoreNames.miner_uid] == 2].iloc[0]
        assert pd.isna(row2[ScoreNames.interval_agg_prediction])

    @pytest.mark.parametrize(
        "predictions_data, expected_value",
        [
            # Case 1: No predictions for any miner.
            (
                [
                    {
                        ScoreNames.miner_uid: 1,
                        ScoreNames.miner_hotkey: "hotkey1",
                        ScoreNames.interval_start: 100240,
                        ScoreNames.interval_agg_prediction: pd.NA,
                    }
                ],
                None,
            ),
            # Case 2: Predictions list with a valid prediction for miner 1 at interval_start 100.
            (
                [
                    {
                        ScoreNames.miner_uid: 1,
                        ScoreNames.miner_hotkey: "hotkey1",
                        ScoreNames.interval_start: 100480,
                        ScoreNames.interval_agg_prediction: 0.8,
                    }
                ],
                0.8,
            ),
        ],
    )
    def test_get_interval_scores_base_parametrized(
        self, scoring_task: Scoring, predictions_data, expected_value
    ):
        miners = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
                ScoreNames.miner_registered_minutes: [100, 200],
            }
        )

        intervals = pd.DataFrame(
            {
                ScoreNames.interval_idx: [0, 1],
                ScoreNames.interval_start: [100240, 100480],
                ScoreNames.interval_end: [100480, 100720],
                ScoreNames.weight: [0.8, 0.5],
            }
        )
        predictions_df = pd.DataFrame(predictions_data)

        result_df = scoring_task.get_interval_scores_base(predictions_df, miners, intervals)
        assert result_df.shape[0] == 4
        assert result_df.columns.to_list() == [
            ScoreNames.miner_uid,
            ScoreNames.miner_hotkey,
            ScoreNames.miner_registered_minutes,
            ScoreNames.interval_idx,
            ScoreNames.interval_start,
            ScoreNames.interval_end,
            ScoreNames.weight,
            ScoreNames.interval_agg_prediction,
        ]

        if expected_value is None:
            assert result_df[ScoreNames.interval_agg_prediction].isna().all()
        else:
            row = result_df[
                (result_df[ScoreNames.miner_uid] == 1)
                & (result_df[ScoreNames.miner_hotkey] == "hotkey1")
                & (result_df[ScoreNames.interval_start] == 100480)
            ]
            assert not row.empty
            np.testing.assert_almost_equal(
                row.iloc[0][ScoreNames.interval_agg_prediction], expected_value
            )

            row_other = result_df[
                (result_df[ScoreNames.miner_uid] == 2)
                & (result_df[ScoreNames.miner_hotkey] == "hotkey2")
                & (result_df[ScoreNames.interval_start] == 100240)
            ]
            assert not row_other.empty
            assert pd.isna(row_other.iloc[0][ScoreNames.interval_agg_prediction])

    def test_return_empty_scores_df(self, scoring_task: Scoring):
        unit = scoring_task
        df = unit.return_empty_scores_df("test", "event_id")

        assert unit.logger.error.call_count == 1
        assert unit.logger.error.call_args_list[0].args[0] == "test"
        assert df.shape[0] == 0
        assert ScoreNames.miner_uid in df.columns
        assert ScoreNames.miner_hotkey in df.columns
        assert ScoreNames.rema_prediction in df.columns
        assert ScoreNames.rema_peer_score in df.columns

    def test_fill_unresponsive_miners_with_failed_run(self, scoring_task: Scoring):
        """Test that miners with failed agent runs get 0.5 imputed."""
        df = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
                ScoreNames.miner_registered_minutes: [50, 50],
                ScoreNames.interval_start: [100, 100],
                ScoreNames.interval_agg_prediction: [pd.NA, 0.8],
                ScoreNames.interval_idx: [0, 0],
                ScoreNames.weight: [1, 1],
            }
        )

        # Create a failed run for miner 1
        failed_runs = [
            AgentRunsModel(
                run_id="run_1",
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                status=AgentRunStatus.SANDBOX_TIMEOUT,
                is_final=True,
            )
        ]

        result_df = scoring_task.fill_unresponsive_miners(
            df, failed_runs=failed_runs, imputed_prediction=0.5
        )

        assert isinstance(result_df, pd.DataFrame)
        # Miner 1 has failed run, so missing prediction should be filled with 0.5
        np.testing.assert_allclose(
            result_df.loc[
                result_df[ScoreNames.miner_uid] == 1, ScoreNames.interval_agg_prediction
            ].iloc[0],
            0.5,
            rtol=1e-5,
        )
        # Miner 2 has real prediction, should be unchanged
        np.testing.assert_allclose(
            result_df.loc[
                result_df[ScoreNames.miner_uid] == 2, ScoreNames.interval_agg_prediction
            ].iloc[0],
            0.8,
            rtol=1e-5,
        )

    def test_fill_unresponsive_miners_without_failed_run_stays_null(self, scoring_task: Scoring):
        """Test that miners without failed runs do NOT get 0.5 imputed - they stay null."""
        df = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
                ScoreNames.miner_registered_minutes: [50, 150],  # miner 2 registered after interval
                ScoreNames.interval_start: [100, 100],
                ScoreNames.interval_agg_prediction: [pd.NA, pd.NA],
                ScoreNames.interval_idx: [0, 0],
                ScoreNames.weight: [1, 1],
            }
        )

        # Only miner 1 has a failed run
        failed_runs = [
            AgentRunsModel(
                run_id="run_1",
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                status=AgentRunStatus.INTERNAL_AGENT_ERROR,
                is_final=True,
            )
        ]

        result_df = scoring_task.fill_unresponsive_miners(
            df, failed_runs=failed_runs, imputed_prediction=0.5
        )
        assert isinstance(result_df, pd.DataFrame)

        # Miner 1: has failed run, missing prediction -> filled with 0.5
        np.testing.assert_allclose(
            result_df.loc[
                result_df[ScoreNames.miner_uid] == 1, ScoreNames.interval_agg_prediction
            ].iloc[0],
            0.5,
            rtol=1e-5,
        )
        # Miner 2: no failed run, missing prediction -> gets dropped entirely
        assert len(result_df[result_df[ScoreNames.miner_uid] == 2]) == 0

    def test_aggregate_predictions_by_miner(self, scoring_task: Scoring):
        # Test REMA weighted aggregation: sum(pred × weight) / sum(weight)
        interval_scores_df = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 1, 1, 2, 2],
                ScoreNames.miner_hotkey: ["hk1", "hk1", "hk1", "hk2", "hk2"],
                ScoreNames.interval_idx: [0, 1, 2, 0, 1],
                ScoreNames.interval_agg_prediction: [0.7, 0.8, 0.9, 0.5, 0.6],
                ScoreNames.weight: [0.11, 0.33, 1.0, 0.5, 1.0],
            }
        )

        result_df = scoring_task.aggregate_predictions_by_miner(interval_scores_df)

        assert result_df.shape[0] == 2  # 2 miners
        assert list(result_df.columns) == [
            ScoreNames.miner_uid,
            ScoreNames.miner_hotkey,
            ScoreNames.rema_prediction,
        ]

        # Miner 1: (0.7×0.11 + 0.8×0.33 + 0.9×1.0) / (0.11 + 0.33 + 1.0) ≈ 0.8312
        miner1_row = result_df[result_df[ScoreNames.miner_uid] == 1].iloc[0]
        expected_miner1 = (0.7 * 0.11 + 0.8 * 0.33 + 0.9 * 1.0) / (0.11 + 0.33 + 1.0)
        np.testing.assert_allclose(
            miner1_row[ScoreNames.rema_prediction], expected_miner1, rtol=1e-5
        )

        # Miner 2: (0.5×0.5 + 0.6×1.0) / (0.5 + 1.0) ≈ 0.5667
        miner2_row = result_df[result_df[ScoreNames.miner_uid] == 2].iloc[0]
        expected_miner2 = (0.5 * 0.5 + 0.6 * 1.0) / (0.5 + 1.0)
        np.testing.assert_allclose(
            miner2_row[ScoreNames.rema_prediction], expected_miner2, rtol=1e-5
        )

    async def test_score_event_no_intervals(self, scoring_task: Scoring, db_operations, db_client):
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        event = EventsModel(
            unique_event_id="evt_no_intervals",
            event_id="e1",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=EventStatus.SETTLED,
            outcome="1",
            cutoff=base_time + timedelta(minutes=AGGREGATION_INTERVAL_LENGTH_MINUTES),
            registered_date=base_time,
        )
        await db_operations.upsert_events([event])
        predictions = []

        unit = scoring_task
        # Set up miners_last_reg
        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1],
                ScoreNames.miner_hotkey: ["hotkey1"],
                ScoreNames.miner_registered_minutes: [0],
            }
        )

        # Mock get_intervals_df to return empty DataFrame (simulating interval generation failure)
        with patch.object(unit, "get_intervals_df", return_value=pd.DataFrame()):
            result = await unit.score_event(event, predictions)

        assert result.empty
        assert ScoreNames.rema_prediction in result.columns
        assert unit.errors_count == 1
        assert unit.logger.error.call_count == 1
        assert (
            "No intervals to score - event discarded."
            == unit.logger.error.call_args_list[0].args[0]
        )

        updated_events = await db_client.many("""SELECT * FROM events""", use_row_factory=True)
        assert len(updated_events) == 1
        assert updated_events[0]["status"] == str(EventStatus.DISCARDED.value)

    async def test_score_event_miner_with_failed_run_gets_imputed(self, scoring_task: Scoring):
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        duration_minutes = 3 * AGGREGATION_INTERVAL_LENGTH_MINUTES

        event = EventsModel(
            unique_event_id="evt_imputation_test",
            event_id="e2",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",
            cutoff=base_time + timedelta(minutes=duration_minutes),
            registered_date=base_time,
        )

        unit = scoring_task

        # Calculate scoring window start
        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        # Miner 1: has predictions for all intervals
        # Miner 2: registered before window, has failed agent run, no prediction - should get 0.5
        predictions = [
            # Miner 1: prediction in interval 0 (older, gets weight 1.0)
            PredictionsModel(
                unique_event_id="evt_imputation_test",
                miner_hotkey="hotkey1",
                miner_uid=1,
                latest_prediction=0.8,
                interval_start_minutes=scoring_window_start_minutes,
                interval_agg_prediction=0.8,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
            # Miner 1: prediction in interval 1 (newer, gets weight 0.0 with 2 intervals)
            PredictionsModel(
                unique_event_id="evt_imputation_test",
                miner_hotkey="hotkey1",
                miner_uid=1,
                latest_prediction=0.8,
                interval_start_minutes=scoring_window_start_minutes
                + AGGREGATION_INTERVAL_LENGTH_MINUTES,
                interval_agg_prediction=0.8,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
        ]

        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["hotkey1", "hotkey2"],
                ScoreNames.miner_registered_minutes: [
                    scoring_window_start_minutes - 1,  # miner 1: registered before
                    scoring_window_start_minutes - 1,  # miner 2: also registered before
                ],
            }
        )

        # Mock failed agent run for miner 2
        unit.db_operations.get_failed_agent_runs_for_event = AsyncMock(
            return_value=[
                AgentRunsModel(
                    run_id="run_failed_2",
                    unique_event_id="evt_imputation_test",
                    agent_version_id="agent_v2",
                    miner_uid=2,
                    miner_hotkey="hotkey2",
                    status=AgentRunStatus.SANDBOX_TIMEOUT,
                    is_final=True,
                )
            ]
        )

        result = await unit.score_event(event, predictions)

        # Both miners should be scored
        assert not result.empty
        assert result.shape[0] == 2

        # Miner 1: has prediction 0.8 in all intervals
        miner1 = result[result[ScoreNames.miner_uid] == 1].iloc[0]
        np.testing.assert_allclose(miner1[ScoreNames.rema_prediction], 0.8, rtol=1e-5)
        # Brier for 0.8 with outcome=1: (0.8 - 1)² = 0.04
        np.testing.assert_allclose(miner1[ScoreNames.rema_peer_score], 0.04, rtol=1e-5)

        # Miner 2: failed run, no prediction -> should be imputed with 0.5
        miner2 = result[result[ScoreNames.miner_uid] == 2].iloc[0]
        np.testing.assert_allclose(miner2[ScoreNames.rema_prediction], 0.5, rtol=1e-5)
        # Brier for 0.5 with outcome=1: (0.5 - 1)² = 0.25
        np.testing.assert_allclose(miner2[ScoreNames.rema_peer_score], 0.25, rtol=1e-5)

    async def test_score_event_no_predictions(self, scoring_task: Scoring):
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        duration_minutes = 3 * AGGREGATION_INTERVAL_LENGTH_MINUTES

        event = EventsModel(
            unique_event_id="evt_no_predictions",
            event_id="e3",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",
            cutoff=base_time + timedelta(minutes=duration_minutes),
            registered_date=base_time,
        )
        predictions = []  # No predictions provided.

        unit = scoring_task

        # Calculate scoring window start to ensure miner is registered before it
        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        # Miner registered BEFORE scoring window start should be included
        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1],
                ScoreNames.miner_hotkey: ["hotkey1"],
                ScoreNames.miner_registered_minutes: [
                    scoring_window_start_minutes - AGGREGATION_INTERVAL_LENGTH_MINUTES
                ],
            }
        )

        # Patch prepare_predictions_df to return an empty DataFrame.
        unit.prepare_predictions_df = lambda predictions, miners: pd.DataFrame()

        result = await unit.score_event(event, predictions)
        assert result.empty
        assert ScoreNames.rema_prediction in result.columns
        # Expect an error message indicating no predictions.
        assert unit.logger.error.call_count >= 1
        assert unit.logger.error.call_args_list[-1].args[0] == "No predictions to score."

    # Test the normal scenario where all required data is present.
    async def test_score_event_normal(self, scoring_task: Scoring):
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        duration_minutes = 2 * AGGREGATION_INTERVAL_LENGTH_MINUTES

        event = EventsModel(
            unique_event_id="evt_normal",
            event_id="e4",
            market_type="dummy",
            event_type="dummy",
            description="dummy event",
            metadata="{}",
            status=1,
            outcome="1",  # outcome text "1" -> float 1.0 -> int 1
            cutoff=base_time + timedelta(minutes=duration_minutes),
            registered_date=base_time,
        )

        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)

        # Calculate scoring window start
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        # Provide predictions for all intervals (like run_agents does via replication)
        predictions = [
            PredictionsModel(
                unique_event_id="evt_normal",
                miner_hotkey="hotkey1",
                miner_uid=1,
                latest_prediction=0.8,
                interval_start_minutes=scoring_window_start_minutes,  # interval 0 (older, higher weight)
                interval_agg_prediction=0.8,
                interval_count=1,
                submitted=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                exported=False,
            ),
            PredictionsModel(
                unique_event_id="evt_normal",
                miner_hotkey="hotkey1",
                miner_uid=1,
                latest_prediction=0.8,
                interval_start_minutes=event_cutoff_start_minutes
                - AGGREGATION_INTERVAL_LENGTH_MINUTES,  # interval 1 (newer, lower weight)
                interval_agg_prediction=0.8,
                interval_count=1,
                submitted=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                exported=False,
            ),
        ]

        unit = scoring_task

        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1],
                ScoreNames.miner_hotkey: ["hotkey1"],
                ScoreNames.miner_registered_minutes: [
                    scoring_window_start_minutes - AGGREGATION_INTERVAL_LENGTH_MINUTES
                ],
            }
        )

        # Mock no failed runs
        unit.db_operations.get_failed_agent_runs_for_event = AsyncMock(return_value=[])

        result = await unit.score_event(event, predictions)

        assert not result.empty
        assert result.shape[0] == 1
        for col in [
            ScoreNames.miner_uid,
            ScoreNames.miner_hotkey,
            ScoreNames.rema_prediction,
            ScoreNames.rema_peer_score,
        ]:
            assert col in result.columns

        # Scoring window generates SCORING_WINDOW_INTERVALS intervals (default 2)
        # Miner has prediction of 0.8 in all intervals
        # No imputation happens (no failed runs)
        row = result.iloc[0]
        assert row[ScoreNames.miner_uid] == 1
        assert row[ScoreNames.miner_hotkey] == "hotkey1"

        # REMA prediction should be 0.8 (all intervals have 0.8)
        np.testing.assert_allclose(row[ScoreNames.rema_prediction], 0.8, rtol=1e-5)

        # Brier score should be calculated from the REMA prediction
        expected_brier = (row[ScoreNames.rema_prediction] - 1.0) ** 2
        np.testing.assert_allclose(row[ScoreNames.rema_peer_score], expected_brier, rtol=1e-5)

        assert unit.errors_count == 0
        assert unit.logger.error.call_count == 0

    async def test_score_event_only_miners_registered_before_window(self, scoring_task: Scoring):
        """Test that only miners registered before scoring window are scored."""
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        duration_minutes = 5 * AGGREGATION_INTERVAL_LENGTH_MINUTES

        event = EventsModel(
            unique_event_id="evt_window_exclusion",
            event_id="e_window_1",
            market_type="dummy",
            event_type="dummy",
            description="scoring window exclusion test",
            metadata="{}",
            status=EventStatus.SETTLED,
            outcome="1",
            cutoff=base_time + timedelta(minutes=duration_minutes),
            registered_date=base_time,
        )

        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        # Predictions for both intervals (like run_agents does via replication)
        predictions = [
            # Miner 1: interval 0
            PredictionsModel(
                unique_event_id="evt_window_exclusion",
                miner_hotkey="miner_early",
                miner_uid=1,
                latest_prediction=0.7,
                interval_start_minutes=scoring_window_start_minutes,
                interval_agg_prediction=0.7,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
            # Miner 1: interval 1
            PredictionsModel(
                unique_event_id="evt_window_exclusion",
                miner_hotkey="miner_early",
                miner_uid=1,
                latest_prediction=0.7,
                interval_start_minutes=scoring_window_start_minutes
                + AGGREGATION_INTERVAL_LENGTH_MINUTES,
                interval_agg_prediction=0.7,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
            # Miner 2: interval 0 (won't be scored - registered after window)
            PredictionsModel(
                unique_event_id="evt_window_exclusion",
                miner_hotkey="miner_late",
                miner_uid=2,
                latest_prediction=0.9,
                interval_start_minutes=scoring_window_start_minutes,
                interval_agg_prediction=0.9,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
            # Miner 2: interval 1 (won't be scored - registered after window)
            PredictionsModel(
                unique_event_id="evt_window_exclusion",
                miner_hotkey="miner_late",
                miner_uid=2,
                latest_prediction=0.9,
                interval_start_minutes=scoring_window_start_minutes
                + AGGREGATION_INTERVAL_LENGTH_MINUTES,
                interval_agg_prediction=0.9,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
        ]

        unit = scoring_task

        # Miner 1: registered before scoring window
        # Miner 2: registered after scoring window start
        # Only miner 1 should be scored
        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1, 2],
                ScoreNames.miner_hotkey: ["miner_early", "miner_late"],
                ScoreNames.miner_registered_minutes: [
                    scoring_window_start_minutes
                    - AGGREGATION_INTERVAL_LENGTH_MINUTES,  # before window
                    scoring_window_start_minutes
                    + AGGREGATION_INTERVAL_LENGTH_MINUTES,  # after window
                ],
            }
        )

        # Mock no failed runs
        unit.db_operations.get_failed_agent_runs_for_event = AsyncMock(return_value=[])

        result = await unit.score_event(event, predictions)

        # Only miner 1 should be scored (registered before window)
        assert not result.empty
        assert result.shape[0] == 1

        # Miner 1 with prediction 0.7
        miner1_row = result[result[ScoreNames.miner_uid] == 1].iloc[0]
        np.testing.assert_allclose(miner1_row[ScoreNames.rema_prediction], 0.7, rtol=1e-5)

        assert unit.errors_count == 0

    async def test_score_event_scoring_window_intervals_generation(self, scoring_task: Scoring):
        base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

        event = EventsModel(
            unique_event_id="evt_window_intervals",
            event_id="e_window_2",
            market_type="dummy",
            event_type="dummy",
            description="scoring window intervals test",
            metadata="{}",
            status=EventStatus.SETTLED,
            outcome="1",
            cutoff=base_time + timedelta(minutes=10 * AGGREGATION_INTERVAL_LENGTH_MINUTES),
            registered_date=base_time,
        )

        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        unit = scoring_task
        unit.miners_last_reg = pd.DataFrame(
            {
                ScoreNames.miner_uid: [1],
                ScoreNames.miner_hotkey: ["miner1"],
                ScoreNames.miner_registered_minutes: [
                    scoring_window_start_minutes - AGGREGATION_INTERVAL_LENGTH_MINUTES
                ],
            }
        )

        predictions = [
            PredictionsModel(
                unique_event_id="evt_window_intervals",
                miner_hotkey="miner1",
                miner_uid=1,
                latest_prediction=0.8,
                interval_start_minutes=event_cutoff_start_minutes
                - AGGREGATION_INTERVAL_LENGTH_MINUTES,
                interval_agg_prediction=0.8,
                interval_count=1,
                submitted=base_time,
                exported=False,
            ),
        ]

        # Mock no failed runs
        unit.db_operations.get_failed_agent_runs_for_event = AsyncMock(return_value=[])

        result = await unit.score_event(event, predictions)

        assert not result.empty
        assert result.shape[0] == 1
        assert unit.errors_count == 0

    @pytest.mark.parametrize(
        "input_data, event_id, expected_valid_count, expected_error_messages",
        [
            # Case 1: All valid records.
            (
                {
                    ScoreNames.miner_uid: [1, 2],
                    ScoreNames.miner_hotkey: ["hk1", "hk2"],
                    ScoreNames.rema_prediction: [0.5, 0.6],
                    ScoreNames.rema_peer_score: [0.1, 0.2],
                },
                "event1",
                2,
                [],  # No errors expected.
            ),
            # Case 2: One record invalid (second record's rema_peer_score is non-numeric).
            (
                {
                    ScoreNames.miner_uid: [1, 2],
                    ScoreNames.miner_hotkey: ["hk1", "hk2"],
                    ScoreNames.rema_prediction: [0.5, 0.6],
                    ScoreNames.rema_peer_score: [0.1, "bad"],
                },
                "event2",
                1,  # Only the first record is valid.
                ["Error while creating a score record."],  # An error should be logged.
            ),
            # Case 3: All records invalid.
            (
                {
                    ScoreNames.miner_uid: [1],
                    ScoreNames.miner_hotkey: ["hk1"],
                    ScoreNames.rema_prediction: ["bad"],
                    ScoreNames.rema_peer_score: ["bad"],
                },
                "event3",
                0,  # No valid records.
                [
                    "Error while creating a score record.",
                    "No scores to export.",
                ],  # Two errors expected.
            ),
        ],
    )
    async def test_export_scores_to_db(
        self,
        db_client,
        scoring_task: Scoring,
        input_data,
        event_id,
        expected_valid_count,
        expected_error_messages,
    ):
        unit = scoring_task
        df = pd.DataFrame(input_data)
        unit.spec_version = 1037

        # Reset errors_count.
        unit.errors_count = 0

        await unit.export_scores_to_db(df, event_id)

        inserted_scores = await db_client.many("SELECT * FROM scores")
        if expected_valid_count > 0:
            assert len(inserted_scores) == expected_valid_count
            assert inserted_scores[0][0] == event_id
        else:
            assert len(inserted_scores) == 0

        error_calls = unit.logger.error.call_args_list
        if expected_error_messages:
            assert len(error_calls) == len(expected_error_messages)
            for i, msg in enumerate(expected_error_messages):
                assert msg == error_calls[i].args[0]
            assert unit.errors_count == len(expected_error_messages)
        else:
            assert unit.errors_count == 0
            assert len(error_calls) == 0

    async def test_e2e_run(
        self,
        scoring_task: Scoring,
        db_client: DatabaseClient,
    ):
        base_date = datetime(2024, 12, 1, 0, 0, tzinfo=timezone.utc)

        # Miner 1: early registration
        miner1_reg_date = base_date
        miner1_reg_minutes = minutes_since_epoch(miner1_reg_date)

        # Events: both registered at +3 intervals
        events_registered_date = base_date + timedelta(
            minutes=3 * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        # Event2: cutoff at +5 intervals (only miner 1 eligible)
        event2_cutoff = base_date + timedelta(minutes=5 * AGGREGATION_INTERVAL_LENGTH_MINUTES)

        # Miner 2: registers after event2 cutoff, eligible for event1
        miner2_reg_date = event2_cutoff + timedelta(minutes=1 * AGGREGATION_INTERVAL_LENGTH_MINUTES)
        miner2_reg_minutes = minutes_since_epoch(miner2_reg_date)

        # Event1: cutoff at +10 intervals (both miners eligible)
        event1_cutoff = base_date + timedelta(minutes=10 * AGGREGATION_INTERVAL_LENGTH_MINUTES)

        # Predictions: replicate across all intervals (like run_agents does)
        event1_cutoff_minutes = minutes_since_epoch(event1_cutoff)
        event1_cutoff_aligned = align_to_interval(event1_cutoff_minutes)
        event1_scoring_window_start = event1_cutoff_aligned - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        event2_cutoff_minutes = minutes_since_epoch(event2_cutoff)
        event2_cutoff_aligned = align_to_interval(event2_cutoff_minutes)
        event2_scoring_window_start = event2_cutoff_aligned - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        event3_cutoff = event1_cutoff

        # Mock dependencies
        unit = scoring_task
        unit.export_scores = AsyncMock()

        # real db client
        db_ops = unit.db_operations

        await unit.run()

        # Assert sync metagraph loads the data
        np.testing.assert_array_equal(unit.current_uids, np.array([1, 2, 3], dtype=np.int64))
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3"]
        assert unit.n_hotkeys == 3
        assert unit.interval_seconds == 60.0
        assert unit.current_miners_df.index.size == 3
        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3]

        # expect no miners found in the DB
        assert unit.errors_count == 1
        assert unit.logger.error.call_count == 1
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "No miners found in the DB, skipping scoring!"
        )

        # reset unit
        unit.errors_count = 0
        unit.logger.error.reset_mock()

        # insert miners
        miners = [
            (
                "1",
                "hotkey1",
                "0.0.0.0",
                miner1_reg_date.isoformat(),
                "100",
                False,
                False,
                "0.0.0.0",
                "100",
            ),
            (
                "2",
                "hotkey2",
                "0.0.0.0",
                miner2_reg_date.isoformat(),
                "100",
                False,
                False,
                "0.0.0.0",
                "100",
            ),
        ]
        await db_ops.upsert_miners(miners)

        # no events
        await unit.run()
        assert unit.errors_count == 0
        assert unit.logger.debug.call_count == 2
        assert unit.logger.debug.call_args_list[0].args[0] == "No events to calculate scores."

        assert unit.miners_last_reg.index.size == 2
        assert unit.miners_last_reg.miner_uid.tolist() == [1, 2]
        assert unit.miners_last_reg[ScoreNames.miner_registered_minutes].tolist() == [
            miner1_reg_minutes,
            miner2_reg_minutes,
        ]

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()

        # insert events
        expected_event_id = "event1"
        resolved_date = event1_cutoff + timedelta(days=1)

        events = [
            EventsModel(
                unique_event_id=expected_event_id,
                event_id=expected_event_id,
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=3,
                metadata='{"key": "value"}',
                created_at=base_date.isoformat(),
                cutoff=event1_cutoff.isoformat(),
                resolved_at=resolved_date.isoformat(),
            ),
            # Event2: Cutoff before miner 2 registration
            EventsModel(
                unique_event_id="event2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc2",
                outcome="0",
                status=3,
                metadata='{"key": "value"}',
                created_at=base_date.isoformat(),
                cutoff=event2_cutoff.isoformat(),
                resolved_at=resolved_date.isoformat(),
            ),
            EventsModel(
                unique_event_id="event3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market3",
                description="desc3",
                outcome=None,
                status=2,
                metadata='{"key": "value"}',
                created_at=base_date.isoformat(),
                cutoff=event3_cutoff.isoformat(),
                resolved_at=resolved_date.isoformat(),
            ),
        ]
        await db_ops.upsert_events(events)
        # registered_date is set by insertion - update it
        await db_client.update(
            f"UPDATE events SET registered_date = '{events_registered_date.isoformat()}'"
        )

        # check correct events are inserted
        events_for_scoring = await db_ops.get_events_for_scoring()
        assert len(events_for_scoring) == 2
        assert events_for_scoring[0].event_id == expected_event_id

        # no predictions, 2 events
        await unit.run()
        assert unit.logger.debug.call_count == 4
        assert unit.logger.debug.call_args_list[0].args[0] == "Found events to calculate scores."
        assert unit.logger.debug.call_args_list[1].args[0] == "Calculating scores for an event."
        assert (
            unit.logger.debug.call_args_list[3].args[0]
            == "Scoring run finished. Resetting errors count."
        )
        assert unit.logger.debug.call_args_list[3].kwargs["extra"]["errors_count_in_logs"] == 2

        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 2
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "There are no predictions for a settled event - discarding."
        )
        updated_events = await db_client.many("""SELECT * FROM events""", use_row_factory=True)
        assert len(updated_events) == 3
        assert updated_events[0]["status"] == str(EventStatus.DISCARDED.value)
        assert updated_events[1]["status"] == str(EventStatus.DISCARDED.value)
        assert updated_events[2]["status"] == str(EventStatus.PENDING.value)

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()
        unit.logger.warning.reset_mock()
        await db_client.update(
            "UPDATE events SET status = ? WHERE status = ? ",
            parameters=[EventStatus.SETTLED, EventStatus.DISCARDED],
        )

        # insert predictions - replicate across all intervals like run_agents does
        predictions = []
        # Event2 predictions for both intervals
        for interval_offset in range(SCORING_WINDOW_INTERVALS):
            predictions.extend(
                [
                    PredictionsModel(
                        unique_event_id="event2",
                        miner_hotkey="hotkey1",
                        miner_uid=1,
                        latest_prediction=1.0,
                        interval_start_minutes=event2_scoring_window_start
                        + (interval_offset * AGGREGATION_INTERVAL_LENGTH_MINUTES),
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id="event2",
                        miner_hotkey="hotkey2",
                        miner_uid=2,
                        latest_prediction=1.0,
                        interval_start_minutes=event2_scoring_window_start
                        + (interval_offset * AGGREGATION_INTERVAL_LENGTH_MINUTES),
                        interval_agg_prediction=1.0,
                    ),
                ]
            )

        # Event1 predictions for both intervals
        for interval_offset in range(SCORING_WINDOW_INTERVALS):
            predictions.extend(
                [
                    PredictionsModel(
                        unique_event_id=expected_event_id,
                        miner_hotkey="hotkey2",
                        miner_uid=2,
                        latest_prediction=1.0,
                        interval_start_minutes=event1_scoring_window_start
                        + (interval_offset * AGGREGATION_INTERVAL_LENGTH_MINUTES),
                        interval_agg_prediction=1.0,
                    ),
                    PredictionsModel(
                        unique_event_id=expected_event_id,
                        miner_hotkey="hotkey3",
                        miner_uid=3,
                        latest_prediction=1.0,
                        interval_start_minutes=event1_scoring_window_start
                        + (interval_offset * AGGREGATION_INTERVAL_LENGTH_MINUTES),
                        interval_agg_prediction=1.0,
                    ),
                ]
            )

        await db_ops.upsert_predictions(predictions)
        exp_predictions = await db_ops.get_predictions_for_scoring(
            unique_event_id=expected_event_id
        )
        exp_predictions += await db_ops.get_predictions_for_scoring(unique_event_id="event2")
        # 2 miners * 2 intervals * 2 events = 8 predictions total
        assert len(exp_predictions) == 8

        # test return empty scores df
        with patch.object(scoring_task, "prepare_predictions_df", return_value=pd.DataFrame()):
            await unit.run()

        assert unit.logger.debug.call_args_list[3].kwargs["extra"]["errors_count_in_logs"] == 2
        assert unit.logger.error.call_count == 6
        assert (
            unit.logger.error.call_args_list[0].args[0]
            == "There are no predictions for a settled event - discarding."
        )
        assert unit.logger.error.call_args_list[2].args[0] == "No predictions to score."
        assert unit.logger.error.call_args_list[0].kwargs["extra"]["event_id"] == expected_event_id
        assert (
            unit.logger.error.call_args_list[3].args[0]
            == "Scores could not be calculated for an event."
        )
        assert unit.logger.error.call_args_list[3].kwargs["extra"]["event_id"] == expected_event_id

        # reset unit
        unit.errors_count = 0
        unit.logger.debug.reset_mock()
        unit.logger.error.reset_mock()

        await unit.run()

        assert unit.logger.debug.call_count == 6
        assert unit.logger.debug.call_args_list[0].args[0] == "Found events to calculate scores."
        assert unit.logger.debug.call_args_list[1].args[0] == "Calculating scores for an event."
        assert unit.logger.debug.call_args_list[1].kwargs["extra"]["event_id"] == expected_event_id
        assert unit.logger.debug.call_args_list[2].args[0] == "Scores calculated, sample below."
        assert unit.logger.debug.call_args_list[2].kwargs["extra"]["event_id"] == expected_event_id
        assert (
            unit.logger.debug.call_args_list[5].args[0]
            == "Scoring run finished. Resetting errors count."
        )
        assert unit.logger.debug.call_args_list[5].kwargs["extra"]["errors_count_in_logs"] == 0

        # Verify actual score values match expectations
        # event1: outcome=1, miners 1 and 2 should be scored (miner 3 not in DB)
        # event2: outcome=0, only miner 1 should be scored (miner 2 registered after event2 cutoff)
        actual_scores_ev_1 = unit.logger.debug.call_args_list[2].kwargs["extra"]["scores"]
        actual_scores_ev_2 = unit.logger.debug.call_args_list[4].kwargs["extra"]["scores"]

        df_actual_ev_1 = pd.DataFrame.from_dict(actual_scores_ev_1, orient="index").reset_index(
            drop=True
        )
        # Only miner 2 should be scored (has prediction, miner 1 has no prediction and no failed run)
        assert len(df_actual_ev_1) == 1, "event1 should have 1 miner scored"

        # Check only miner 2 is present
        assert set(df_actual_ev_1[ScoreNames.miner_uid].tolist()) == {2}

        # For each miner, verify bier score
        for idx, row in df_actual_ev_1.iterrows():
            rema_pred = row[ScoreNames.rema_prediction]
            assert 0 <= rema_pred <= 1, f"event1 prediction {rema_pred} out of range"

            # Brier score for outcome=1: (pred - 1)²
            expected_brier = (rema_pred - 1.0) ** 2
            actual_brier = row[ScoreNames.rema_peer_score]
            np.testing.assert_allclose(
                actual_brier,
                expected_brier,
                rtol=1e-5,
                err_msg=f"Miner {row[ScoreNames.miner_uid]} Brier mismatch",
            )

        # Validate event2 scores (outcome=0)
        # Only miner 1 should be scored (miner 2 registered after event2 cutoff)
        df_actual_ev_2 = pd.DataFrame.from_dict(actual_scores_ev_2, orient="index").reset_index(
            drop=True
        )
        assert len(df_actual_ev_2) == 1, "event2 should have 1 miner scored"
        assert set(df_actual_ev_2[ScoreNames.miner_uid].tolist()) == {1}

        # Miner 1: has prediction 1.0 (clipped to 0.99)
        miner1_ev2 = df_actual_ev_2[df_actual_ev_2[ScoreNames.miner_uid] == 1].iloc[0]
        rema_pred_ev2_m1 = miner1_ev2[ScoreNames.rema_prediction]
        assert (
            0 <= rema_pred_ev2_m1 <= 1
        ), f"event2 miner1 prediction {rema_pred_ev2_m1} out of range"
        expected_brier_ev2_m1 = (rema_pred_ev2_m1 - 0.0) ** 2
        np.testing.assert_allclose(
            miner1_ev2[ScoreNames.rema_peer_score], expected_brier_ev2_m1, rtol=1e-5
        )

        # Check events are marked as processed
        events_for_scoring = await db_ops.get_events_for_scoring()
        assert len(events_for_scoring) == 0

    async def test_run_context_managers(self, scoring_task: Scoring):
        scoring_task.subtensor_cm.__aenter__.assert_not_awaited()
        scoring_task.subtensor_cm.__aexit__.assert_not_awaited()
        scoring_task.subtensor_cm.metagraph.assert_not_awaited()
        assert not hasattr(scoring_task, "metagraph")

        await scoring_task.run()

        scoring_task.subtensor_cm.__aenter__.assert_awaited_once()
        scoring_task.subtensor_cm.__aexit__.assert_awaited_once()
        scoring_task.subtensor_cm.metagraph.assert_awaited_once_with(
            netuid=scoring_task.netuid, lite=True
        )

        assert isinstance(scoring_task.metagraph, IfMetagraph)
