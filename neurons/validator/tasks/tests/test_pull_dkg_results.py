import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from bittensor_wallet import Keypair
from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.models.dkg import DkgForecastHashModel, DkgQuestionModel, DkgRoundModel
from neurons.validator.models.dkg_client import (
    ReleasedForecastResponse,
    ResultsPage,
    StatementResult,
)
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.tasks.pull_dkg_results import (
    HASH_DOES_NOT_MATCH,
    RESULTS_SINCE_MARGIN,
    SIGNATURE_DOES_NOT_VERIFY,
    PullDkgResults,
)
from neurons.validator.utils.logger.logger import NuminousLogger

MINER = Keypair.create_from_uri("//Alice")
OTHER_MINER = Keypair.create_from_uri("//Bob")
UID_OF = {MINER.ss58_address: 1, OTHER_MINER.ss58_address: 2}
REGISTERED_AGAIN = 99
NOW = datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)
HISTORY_START = NOW - DKG_HISTORY
DEADLINE = datetime(2026, 10, 13, tzinfo=timezone.utc)
OPENED = {
    "version": 2,
    "statements": [
        {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.1},
        {"head": "Apple", "relation": "ACQUIRES", "tail": "TSMC", "probability": 0.01},
    ],
}


def sealed_bytes(question_id: str) -> bytes:
    return f"sealed forecast for {question_id}".encode()


def make_released(
    miner: Keypair, question_id: str, opened: dict | None, results: list[StatementResult]
) -> ReleasedForecastResponse:
    sealed = sealed_bytes(question_id)
    return ReleasedForecastResponse(
        miner_uid=UID_OF[miner.ss58_address],
        miner_hotkey=miner.ss58_address,
        question_id=question_id,
        deadline_at=DEADLINE,
        sealed=sealed.hex(),
        signature=miner.sign(question_id.encode() + sealed).hex(),
        opened=opened,
        invalid_reason=None if opened else "the seal does not open",
        results=results,
    )


def make_result(statement_position: int, outcome: bool) -> StatementResult:
    return StatementResult(
        statement_position=statement_position,
        outcome=outcome,
        reason="a reported development is the act the forecast states",
        event_id="event-1" if outcome else None,
        reported_at=datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc) if outcome else None,
        judged_at=datetime(2026, 10, 21, 7, 30, tzinfo=timezone.utc),
    )


class TestPullDkgResults:
    @pytest.fixture(autouse=True)
    def frozen_now(self):
        with freeze_time(NOW) as frozen_now:
            yield frozen_now

    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        dkg_operations = DkgDatabaseOperations(db_client=db_client)
        await dkg_operations.save_round(
            DkgRoundModel(
                round_id="round-1",
                deadline_at=DEADLINE,
                horizon_end_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
            ),
            [
                DkgQuestionModel(
                    question_id=question_id,
                    round_id="round-1",
                    entity_id="entity-apple",
                    entity="Apple",
                    counterparty_id=None,
                    counterparty=None,
                    relationship=None,
                )
                for question_id in ["question-1", "question-2"]
            ],
        )
        await dkg_operations.save_forecast_hashes(
            [
                DkgForecastHashModel(
                    miner_uid=UID_OF[MINER.ss58_address],
                    miner_hotkey=MINER.ss58_address,
                    question_id=question_id,
                    sealed_sha256=hashlib.sha256(sealed_bytes(question_id)).hexdigest(),
                    received_at=datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc),
                )
                for question_id in ["question-1", "question-2"]
            ]
        )
        await dkg_operations.mark_round_hashes_pulled("round-1")
        return dkg_operations

    @pytest.fixture
    def dkg_client(self) -> AsyncMock:
        dkg_client = AsyncMock(spec=DkgClient)
        dkg_client.get_results.return_value = ResultsPage(forecasts=[], next_offset=None)
        return dkg_client

    @pytest.fixture
    def logger(self) -> MagicMock:
        return MagicMock(spec=NuminousLogger)

    @pytest.fixture
    def task(
        self, dkg_operations: DkgDatabaseOperations, dkg_client: AsyncMock, logger: MagicMock
    ) -> PullDkgResults:
        return PullDkgResults(
            interval_seconds=900.0,
            dkg_operations=dkg_operations,
            dkg_client=dkg_client,
            logger=logger,
        )

    @pytest.mark.parametrize(
        "changed, error",
        [
            ({"interval_seconds": 900}, ValueError),
            ({"interval_seconds": -1.0}, ValueError),
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
            "interval_seconds": 900.0,
            "dkg_operations": dkg_operations,
            "dkg_client": dkg_client,
            "logger": logger,
        }

        with pytest.raises(error):
            PullDkgResults(**(arguments | changed))

    def test_name_and_interval(self, task: PullDkgResults):
        assert task.name == "pull-dkg-results"
        assert task.interval_seconds == 900.0

    async def test_first_run_asks_from_the_history_start_and_records_the_pull(
        self, task: PullDkgResults, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        before = datetime.now(timezone.utc)

        await task.run()

        dkg_client.get_results.assert_called_once_with(HISTORY_START, 0)
        assert await dkg_operations.get_last_results_pull() == before

    async def test_next_run_asks_from_the_last_pull_minus_the_margin(
        self, task: PullDkgResults, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        last_pull = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
        await dkg_operations.record_results_pull(last_pull)

        await task.run()

        dkg_client.get_results.assert_called_once_with(last_pull - RESULTS_SINCE_MARGIN, 0)

    async def test_stores_opened_and_invalid_forecasts_with_their_results(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        db_client: DatabaseClient,
        logger: MagicMock,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(
                    MINER, "question-1", OPENED, [make_result(0, True), make_result(1, False)]
                ),
                make_released(MINER, "question-2", None, []),
            ],
            next_offset=None,
        )

        await task.run()

        forecasts = await db_client.many(
            "SELECT miner_uid, question_id, opened, invalid_reason, failed_check FROM dkg_forecasts ORDER BY question_id"
        )
        results = await db_client.many(
            "SELECT miner_uid, statement_position, outcome, event_id, reported_at FROM dkg_statement_results ORDER BY statement_position"
        )
        miner_uid = UID_OF[MINER.ss58_address]
        assert [tuple(row) for row in forecasts] == [
            (miner_uid, "question-1", json.dumps(OPENED), None, None),
            (miner_uid, "question-2", None, "the seal does not open", None),
        ]
        assert [tuple(row) for row in results] == [
            (miner_uid, 0, 1, "event-1", "2026-10-20T09:00:00Z"),
            (miner_uid, 1, 0, None, None),
        ]
        logger.info.assert_called_once_with(
            "DKG results pulled",
            extra={"since": HISTORY_START.isoformat(), "forecasts": 2},
        )

    @pytest.mark.parametrize(
        "tampered, failed_check",
        [
            ({"sealed": sealed_bytes("question-2").hex()}, HASH_DOES_NOT_MATCH),
            (
                {"signature": OTHER_MINER.sign(b"question-1" + sealed_bytes("question-1")).hex()},
                SIGNATURE_DOES_NOT_VERIFY,
            ),
        ],
    )
    async def test_a_forecast_that_fails_a_check_is_stored_with_the_reason(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        db_client: DatabaseClient,
        tampered: dict,
        failed_check: str,
    ):
        released = make_released(MINER, "question-1", OPENED, []).model_copy(update=tampered)
        dkg_client.get_results.return_value = ResultsPage(forecasts=[released], next_offset=None)

        await task.run()

        row = await db_client.one("SELECT failed_check FROM dkg_forecasts")
        assert row[0] == failed_check

    async def test_a_forecast_without_its_hash_holds_the_cursor(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
        logger: MagicMock,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(MINER, "question-1", OPENED, []),
                make_released(OTHER_MINER, "question-1", OPENED, []),
            ],
            next_offset=None,
        )

        await task.run()

        stored = await db_client.many("SELECT miner_hotkey FROM dkg_forecasts")
        assert [row[0] for row in stored] == [MINER.ss58_address]
        assert await dkg_operations.get_last_results_pull() is None
        logger.warning.assert_called_once_with(
            "DKG results waiting for their round, pull not recorded",
            extra={"since": HISTORY_START.isoformat(), "waiting_for_round": 1},
        )
        logger.info.assert_not_called()

    async def test_the_same_hotkey_under_another_uid_waits_for_its_own_hash(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(MINER, "question-1", OPENED, []).model_copy(
                    update={"miner_uid": REGISTERED_AGAIN}
                )
            ],
            next_offset=None,
        )

        await task.run()

        assert (await db_client.one("SELECT COUNT(*) FROM dkg_forecasts"))[0] == 0
        assert await dkg_operations.get_last_results_pull() is None

    async def test_the_same_hotkey_under_another_uid_is_stored_as_another_forecast(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
    ):
        await dkg_operations.save_forecast_hashes(
            [
                DkgForecastHashModel(
                    miner_uid=REGISTERED_AGAIN,
                    miner_hotkey=MINER.ss58_address,
                    question_id="question-1",
                    sealed_sha256=hashlib.sha256(sealed_bytes("question-1")).hexdigest(),
                    received_at=datetime(2026, 10, 12, 19, 0, tzinfo=timezone.utc),
                )
            ]
        )
        first = make_released(MINER, "question-1", OPENED, [make_result(0, True)])
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                first,
                first.model_copy(
                    update={"miner_uid": REGISTERED_AGAIN, "results": [make_result(0, False)]}
                ),
            ],
            next_offset=None,
        )

        await task.run()

        results = await db_client.many(
            "SELECT miner_uid, outcome FROM dkg_statement_results ORDER BY miner_uid"
        )
        assert [tuple(row) for row in results] == [
            (UID_OF[MINER.ss58_address], 1),
            (REGISTERED_AGAIN, 0),
        ]
        assert await dkg_operations.get_last_results_pull() is not None

    async def test_walks_every_page(
        self, task: PullDkgResults, dkg_client: AsyncMock, db_client: DatabaseClient
    ):
        dkg_client.get_results.side_effect = [
            ResultsPage(forecasts=[make_released(MINER, "question-1", OPENED, [])], next_offset=50),
            ResultsPage(forecasts=[make_released(MINER, "question-2", None, [])], next_offset=None),
        ]

        await task.run()

        assert dkg_client.get_results.call_args_list == [
            call(HISTORY_START, 0),
            call(HISTORY_START, 50),
        ]
        assert (await db_client.one("SELECT COUNT(*) FROM dkg_forecasts"))[0] == 2

    async def test_a_forecast_released_again_overwrites(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        db_client: DatabaseClient,
        frozen_now: FrozenDateTimeFactory,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[make_released(MINER, "question-1", OPENED, [make_result(0, False)])],
            next_offset=None,
        )
        await task.run()
        frozen_now.tick(timedelta(minutes=15))

        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(
                    MINER, "question-1", OPENED, [make_result(0, False), make_result(1, True)]
                )
            ],
            next_offset=None,
        )
        await task.run()

        assert (await db_client.one("SELECT COUNT(*) FROM dkg_forecasts"))[0] == 1
        results = await db_client.many(
            "SELECT statement_position, outcome FROM dkg_statement_results ORDER BY statement_position"
        )
        assert [tuple(row) for row in results] == [(0, 0), (1, 1)]

    async def test_after_a_long_downtime_it_asks_from_the_history_start(
        self, task: PullDkgResults, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.record_results_pull(NOW - timedelta(days=120))

        await task.run()

        dkg_client.get_results.assert_called_once_with(HISTORY_START, 0)

    async def test_a_forecast_of_a_round_past_the_history_is_skipped_not_waited_for(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
        db_client: DatabaseClient,
        logger: MagicMock,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(MINER, "question-1", OPENED, []),
                make_released(MINER, "question-of-an-expired-round", OPENED, []).model_copy(
                    update={"deadline_at": HISTORY_START}
                ),
            ],
            next_offset=None,
        )

        await task.run()

        stored = await db_client.many("SELECT question_id FROM dkg_forecasts")
        assert [row[0] for row in stored] == ["question-1"]
        assert await dkg_operations.get_last_results_pull() == NOW
        logger.warning.assert_not_called()
        logger.info.assert_called_once_with(
            "DKG results pulled", extra={"since": HISTORY_START.isoformat(), "forecasts": 1}
        )

    async def test_a_forecast_just_inside_the_history_still_waits_for_its_round(
        self,
        task: PullDkgResults,
        dkg_client: AsyncMock,
        dkg_operations: DkgDatabaseOperations,
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(MINER, "question-of-a-round-not-pulled-yet", OPENED, []).model_copy(
                    update={"deadline_at": HISTORY_START + timedelta(seconds=1)}
                )
            ],
            next_offset=None,
        )

        await task.run()

        assert await dkg_operations.get_last_results_pull() is None

    async def test_a_forecast_past_the_history_is_skipped_even_with_its_hash_stored(
        self, task: PullDkgResults, dkg_client: AsyncMock, db_client: DatabaseClient
    ):
        dkg_client.get_results.return_value = ResultsPage(
            forecasts=[
                make_released(MINER, "question-1", OPENED, []).model_copy(
                    update={"deadline_at": HISTORY_START - timedelta(days=1)}
                )
            ],
            next_offset=None,
        )

        await task.run()

        assert (await db_client.one("SELECT COUNT(*) FROM dkg_forecasts"))[0] == 0
