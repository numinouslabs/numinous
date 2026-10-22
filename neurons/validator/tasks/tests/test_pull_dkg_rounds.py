import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.models.dkg_client import (
    ClosedRoundsPage,
    DkgQuestionRelationship,
    DkgQuestionResponse,
    DkgRoundResponse,
    ForecastHash,
    RoundHashesPage,
)
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.tasks.pull_dkg_rounds import PullDkgRounds
from neurons.validator.utils.logger.logger import NuminousLogger

MINER = "5MinerHotkey"
MINER_UID = 7
NOW = datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)
HISTORY_START = NOW - DKG_HISTORY


def make_round(round_id: str, deadline_day: int) -> DkgRoundResponse:
    return DkgRoundResponse(
        id=round_id,
        deadline_at=datetime(2026, 10, deadline_day, 0, 0, tzinfo=timezone.utc),
        horizon_end_at=datetime(2026, 11, deadline_day, 0, 0, tzinfo=timezone.utc),
        questions=[
            DkgQuestionResponse(
                id=f"{round_id}-question-1",
                entity_id="entity-apple",
                entity="Apple",
                counterparty_id="entity-tsmc",
                counterparty="TSMC",
                relationship=DkgQuestionRelationship(
                    head="TSMC", relation="SUPPLIES", tail="Apple"
                ),
            ),
            DkgQuestionResponse(
                id=f"{round_id}-question-2",
                entity_id="entity-nvidia",
                entity="NVIDIA",
                counterparty_id=None,
                counterparty=None,
                relationship=None,
            ),
        ],
    )


def make_hash(question_id: str) -> ForecastHash:
    return ForecastHash(
        miner_uid=MINER_UID,
        miner_hotkey=MINER,
        question_id=question_id,
        sealed_sha256=f"sha256-of-{question_id}",
        received_at=datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc),
    )


def no_hashes(round_id: str, offset: int) -> RoundHashesPage:
    return RoundHashesPage(forecasts=[], next_offset=None)


class TestPullDkgRounds:
    @pytest.fixture(autouse=True)
    def frozen_now(self):
        with freeze_time(NOW):
            yield

    @pytest.fixture
    def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        return DkgDatabaseOperations(db_client=db_client)

    @pytest.fixture
    def dkg_client(self) -> AsyncMock:
        dkg_client = AsyncMock(spec=DkgClient)
        dkg_client.get_closed_rounds.return_value = ClosedRoundsPage(rounds=[], next_offset=None)
        dkg_client.get_round_hashes.side_effect = no_hashes
        return dkg_client

    @pytest.fixture
    def logger(self) -> MagicMock:
        return MagicMock(spec=NuminousLogger)

    @pytest.fixture
    def task(
        self, dkg_operations: DkgDatabaseOperations, dkg_client: AsyncMock, logger: MagicMock
    ) -> PullDkgRounds:
        return PullDkgRounds(
            interval_seconds=600.0,
            dkg_operations=dkg_operations,
            dkg_client=dkg_client,
            logger=logger,
        )

    @pytest.mark.parametrize(
        "changed, error",
        [
            ({"interval_seconds": 600}, ValueError),
            ({"interval_seconds": 0.0}, ValueError),
            ({"dkg_operations": "not operations"}, TypeError),
            ({"dkg_client": "not a client"}, TypeError),
            ({"logger": "not a logger"}, TypeError),
        ],
    )
    def test_init_rejects_bad_arguments(
        self,
        dkg_operations: DkgDatabaseOperations,
        dkg_client: AsyncMock,
        logger: MagicMock,
        changed: dict,
        error: type[Exception],
    ):
        arguments = {
            "interval_seconds": 600.0,
            "dkg_operations": dkg_operations,
            "dkg_client": dkg_client,
            "logger": logger,
        }

        with pytest.raises(error):
            PullDkgRounds(**(arguments | changed))

    def test_name_and_interval(self, task: PullDkgRounds):
        assert task.name == "pull-dkg-rounds"
        assert task.interval_seconds == 600.0

    async def test_first_run_asks_from_the_history_start(
        self, task: PullDkgRounds, dkg_client: AsyncMock
    ):
        await task.run()

        dkg_client.get_closed_rounds.assert_called_once_with(HISTORY_START, 0)

    async def test_saves_the_round_its_questions_and_every_page_of_hashes(
        self,
        task: PullDkgRounds,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
        logger: MagicMock,
    ):
        closed_round = make_round("round-1", 13)
        dkg_client.get_closed_rounds.return_value = ClosedRoundsPage(
            rounds=[closed_round], next_offset=None
        )
        dkg_client.get_round_hashes.side_effect = [
            RoundHashesPage(forecasts=[make_hash("round-1-question-1")], next_offset=1000),
            RoundHashesPage(forecasts=[make_hash("round-1-question-2")], next_offset=None),
        ]

        await task.run()

        assert dkg_client.get_round_hashes.call_args_list == [
            call("round-1", 0),
            call("round-1", 1000),
        ]
        questions = await db_client.many(
            "SELECT question_id, counterparty, relationship FROM dkg_questions ORDER BY question_id"
        )
        assert [tuple(row) for row in questions] == [
            (
                "round-1-question-1",
                "TSMC",
                json.dumps({"head": "TSMC", "relation": "SUPPLIES", "tail": "Apple"}),
            ),
            ("round-1-question-2", None, None),
        ]
        stored_hashes = await dkg_operations.get_forecast_hashes(
            ["round-1-question-1", "round-1-question-2"]
        )
        assert sorted(
            (stored.miner_uid, stored.miner_hotkey, stored.question_id) for stored in stored_hashes
        ) == [
            (MINER_UID, MINER, "round-1-question-1"),
            (MINER_UID, MINER, "round-1-question-2"),
        ]
        assert await dkg_operations.get_last_complete_round_deadline() == closed_round.deadline_at
        logger.info.assert_called_once_with(
            "DKG round pulled",
            extra={
                "round_id": "round-1",
                "deadline_at": "2026-10-13T00:00:00+00:00",
                "questions": 2,
                "hashes": 2,
            },
        )

    async def test_walks_every_page_of_rounds(
        self, task: PullDkgRounds, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        dkg_client.get_closed_rounds.side_effect = [
            ClosedRoundsPage(rounds=[make_round("round-1", 12)], next_offset=30),
            ClosedRoundsPage(rounds=[make_round("round-2", 13)], next_offset=None),
        ]

        await task.run()

        assert dkg_client.get_closed_rounds.call_args_list == [
            call(HISTORY_START, 0),
            call(HISTORY_START, 30),
        ]
        assert await dkg_operations.get_last_complete_round_deadline() == datetime(
            2026, 10, 13, 0, 0, tzinfo=timezone.utc
        )

    async def test_next_run_asks_from_the_last_complete_round(
        self, task: PullDkgRounds, dkg_client: AsyncMock
    ):
        dkg_client.get_closed_rounds.return_value = ClosedRoundsPage(
            rounds=[make_round("round-1", 13)], next_offset=None
        )
        await task.run()

        dkg_client.get_closed_rounds.return_value = ClosedRoundsPage(rounds=[], next_offset=None)
        await task.run()

        assert dkg_client.get_closed_rounds.call_args_list[-1] == call(
            datetime(2026, 10, 13, 0, 0, tzinfo=timezone.utc), 0
        )

    async def test_a_round_cut_short_is_pulled_again(
        self,
        task: PullDkgRounds,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
    ):
        dkg_client.get_closed_rounds.return_value = ClosedRoundsPage(
            rounds=[make_round("round-1", 13)], next_offset=None
        )
        dkg_client.get_round_hashes.side_effect = RuntimeError("backend unreachable")

        with pytest.raises(RuntimeError):
            await task.run()

        assert await dkg_operations.get_last_complete_round_deadline() is None

        dkg_client.get_round_hashes.side_effect = no_hashes
        await task.run()

        assert dkg_client.get_closed_rounds.call_args_list[-1] == call(HISTORY_START, 0)
        assert await dkg_operations.get_last_complete_round_deadline() == datetime(
            2026, 10, 13, 0, 0, tzinfo=timezone.utc
        )
        assert (await db_client.one("SELECT COUNT(*) FROM dkg_questions"))[0] == 2

    async def test_after_a_long_downtime_it_asks_from_the_history_start(
        self, task: PullDkgRounds, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        expired_round = make_round("round-1", 13).model_copy(
            update={"deadline_at": NOW - timedelta(days=200)}
        )
        await task.pull_round(expired_round)

        await task.run()

        assert dkg_client.get_closed_rounds.call_args_list[-1] == call(HISTORY_START, 0)
