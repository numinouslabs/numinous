import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.score import SCORE_FIELDS, ScoresModel
from neurons.validator.tasks.metagraph_scoring import MetagraphScoring
from neurons.validator.utils.logger.logger import NuminousLogger


class TestMetagraphScoring:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def metagraph_scoring_task(
        self,
        db_operations: DatabaseOperations,
    ):
        logger = MagicMock(spec=NuminousLogger)
        metagraph = MagicMock()
        metagraph.owner_hotkey = "test_owner_hotkey"
        metagraph.uids = np.array([239, 10, 20, 30])
        metagraph.hotkeys = ["test_owner_hotkey", "hk10", "hk20", "hk30"]

        with freeze_time("2025-01-02 03:00:00"):
            return MetagraphScoring(
                interval_seconds=60.0,
                page_size=100,
                db_operations=db_operations,
                logger=logger,
                metagraph=metagraph,
            )

    def test_init(self, metagraph_scoring_task: MetagraphScoring):
        unit = metagraph_scoring_task

        assert isinstance(unit, MetagraphScoring)
        assert unit.interval_seconds == 60.0
        assert unit.page_size == 100
        assert unit.errors_count == 0

    @pytest.mark.parametrize(
        "scores_list, expected_result, log_calls",
        [
            # Case 0: No scores.
            (
                [],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 1: All scores are processed.
            (
                [
                    ScoresModel(
                        event_id="processed_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="processed_event_id_2",
                        miner_uid=1,
                        miner_hotkey="hk1",
                        prediction=0.85,
                        event_score=0.90,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                ],
                [],
                {
                    "debug": [
                        ("No events to calculate metagraph scores.", {}),
                    ]
                },
            ),
            # Case 2: Single miner single event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.80,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 2,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 3: Two miners single event, one miner processed -> reprocess.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,  # Higher Brier (worse)
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=True,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,  # Lower Brier (better) → Winner
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.002,  # Rank 2, only non-winner, gets all (1-0.80)*(1-0.99) = 0.002
                        "other_data": {
                            "average_brier_score": 0.80,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.40,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 3,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 4: Exception during set metagraph peer scores.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                    "exception": [
                        (
                            "Error calculating metagraph scores.",
                            {"event_id": "expected_event_id"},
                        ),
                    ],
                },
            ),
            # Case 5: Three miners single event, different Brier scores.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,  # Rank 3 (worst non-burn)
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,  # Rank 2
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=0.10,  # Rank 1 (best)
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.0004,  # Rank 3: (1-0.80)*(1-0.99)*(1/3)/(1/2+1/3) ≈ 0.0004
                        "other_data": {
                            "average_brier_score": 0.80,
                            "rank": 3,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.0006,  # Rank 2: (1-0.80)*(1-0.99)*(1/2)/(1/2+1/3) ≈ 0.0006
                        "other_data": {
                            "average_brier_score": 0.40,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Rank 1: (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.10,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 4,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 1, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 6: One miner for 2 events, one miner for 1 event.
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Miner 3: rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.80,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.002,  # Miner 3: avg=(0.80+0.40)/2=0.60, rank 2, only non-winner, gets all (1-0.80)*(1-0.99) = 0.002
                        "other_data": {
                            "average_brier_score": 0.60,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Miner 4: avg=0.40, rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.40,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 3,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 2, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
            # Case 7: 1 miner 3 events, 1 miner 2 events, 1 miner 1 event with very low Brier score.
            # Event 3 only considers miners IN event 3 (not miner 5).
            # This means miner 4 becomes rank 1 for event 3 (not miner 5 who isn't present).
            (
                [
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.80,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_1",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=3,
                        miner_hotkey="hk3",
                        prediction=0.75,
                        event_score=0.60,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=4,
                        miner_hotkey="hk4",
                        prediction=0.75,
                        event_score=0.40,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=5,
                        miner_hotkey="hk5",
                        prediction=0.75,
                        event_score=0.10,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_2",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                    ScoresModel(
                        event_id="expected_event_id_3",
                        miner_uid=239,
                        miner_hotkey="hk239",
                        prediction=0.0,
                        event_score=1.00,
                        created_at="2025-01-02 03:00:00",
                        spec_version=1,
                        processed=False,
                    ),
                ],
                [
                    # Event 1: Only miners 3 and 239
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Miner 3: rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.80,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id_1",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 2,
                        },
                    },
                    # Event 2: Miners 3, 4, 5, 239 - miner 5 is present and wins
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.0008,  # Miner 3: avg=0.60, rank 3, gets (1-0.80)*(1-0.99)*(1/3)/(1/2+1/3)
                        "other_data": {
                            "average_brier_score": 0.60,
                            "rank": 3,
                        },
                    },
                    # Event 3: Only miners 3, 4, 239 - miner 5 NOT present!
                    # Miner 4 (avg=0.40) becomes rank 1, miner 3 (avg=0.60) becomes rank 2
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.002,  # Miner 3: avg=0.60, rank 2, only non-winner non-burn, gets all 0.002
                        "other_data": {
                            "average_brier_score": 0.60,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.0012,  # Miner 4: avg=0.40, rank 2, gets (1-0.80)*(1-0.99)*(1/2)/(1/2+1/3)
                        "other_data": {
                            "average_brier_score": 0.40,
                            "rank": 2,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Miner 4: avg=0.40, rank 1 (winner since miner 5 not in event 3)
                        "other_data": {
                            "average_brier_score": 0.40,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.198,  # Miner 5: avg=0.10, rank 1, gets (1-0.80)*0.99 = 0.198
                        "other_data": {
                            "average_brier_score": 0.10,
                            "rank": 1,
                        },
                    },
                    {
                        "event_id": "expected_event_id_2",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 4,
                        },
                    },
                    {
                        "event_id": "expected_event_id_3",
                        "processed": 1,
                        "metagraph_score": 0.8,  # UID 239 gets 80%, rank 3 (only 3 miners in event)
                        "other_data": {
                            "average_brier_score": 1.00,
                            "rank": 3,
                        },
                    },
                ],
                {
                    "debug": [
                        (
                            "Found events to calculate metagraph scores.",
                            {"n_events": 3, "burn_uid": 239},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_1"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_2"},
                        ),
                        (
                            "Processing event for metagraph scoring.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        (
                            "Metagraph scores calculated successfully.",
                            {"event_id": "expected_event_id_3"},
                        ),
                        ("Metagraph scoring task completed.", {"errors_count": 0}),
                    ]
                },
            ),
        ],
    )
    async def test_run(
        self,
        metagraph_scoring_task: MetagraphScoring,
        scores_list,
        expected_result,
        log_calls,
        db_client,
        db_operations,
    ):
        # insert EVENTS that match the score rows
        if scores_list:
            uniq_event_ids = {s.event_id for s in scores_list}

            stub_events = []
            for eid in uniq_event_ids:
                # simple heuristic: prediction > 0.5 -> outcome "1", else "0"
                some_score = next(s for s in scores_list if s.event_id == eid)
                outcome = "1" if some_score.prediction >= 0.5 else "0"

                stub_events.append(
                    EventsModel(
                        unique_event_id=f"stub_{eid}",
                        event_id=eid,
                        market_type="unit_test",
                        event_type="unit_test",
                        description=f"stub for {eid}",
                        outcome=outcome,
                        status=EventStatus.SETTLED,
                        metadata="{}",
                        created_at="2100-01-01T00:00:00+00:00",
                        cutoff="2100-01-01T00:00:00+00:00",
                    )
                )

            await db_operations.upsert_events(stub_events)

        # insert scores
        sql = f"""
            INSERT INTO scores ({', '.join(SCORE_FIELDS)})
            VALUES ({', '.join(['?'] * len(SCORE_FIELDS))})
        """
        if scores_list:
            score_tuples = [
                tuple(getattr(score, field) for field in SCORE_FIELDS) for score in scores_list
            ]
            await db_client.insert_many(sql, score_tuples)

            # confirm setup
            inserted_scores = await db_client.many("SELECT * FROM scores")
            assert len(inserted_scores) == len(scores_list)

        # run the task
        if "exception" in log_calls:
            db_operations.set_metagraph_scores = AsyncMock(return_value=[100])

        await metagraph_scoring_task.run()
        updated_scores = await db_client.many("SELECT * FROM scores", use_row_factory=True)
        debug_calls = metagraph_scoring_task.logger.debug.call_args_list
        exception_calls = metagraph_scoring_task.logger.exception.call_args_list

        # confirm logs
        for log_type, calls in log_calls.items():
            for i, (args, kwargs) in enumerate(calls):
                if log_type == "debug":
                    assert debug_calls[i][0][0] == args
                    if kwargs:
                        assert debug_calls[i][1]["extra"] == kwargs
                elif log_type == "exception":
                    assert exception_calls[i][0][0] == args
                    assert exception_calls[i][1]["extra"] == kwargs

        # confirm results
        if expected_result:
            assert len(updated_scores) == len(expected_result)
            for i, expected in enumerate(expected_result):
                updated = updated_scores[i]
                assert updated["event_id"] == expected["event_id"]
                assert updated["processed"] == expected["processed"]
                assert updated["metagraph_score"] == pytest.approx(
                    expected["metagraph_score"], abs=1e-3
                )

                other_data = json.loads(updated["other_data"])
                # Validate Brier scoring fields
                assert other_data["average_brier_score"] == pytest.approx(
                    expected["other_data"]["average_brier_score"], abs=1e-3
                )
                assert other_data["rank"] == expected["other_data"]["rank"]
        else:
            assert len(updated_scores) == len(scores_list)
            for i, updated in enumerate(updated_scores):
                assert updated["metagraph_score"] is None
                assert updated["other_data"] is None
