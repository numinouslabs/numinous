import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

import neurons.validator.tasks.score_dkg_miners as score_dkg_miners
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgForecastHashModel,
    DkgForecastModel,
    DkgForecastRightsModel,
    DkgMinerModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgStatementResultModel,
)
from neurons.validator.tasks.score_dkg_miners import (
    SCORE_WINDOW,
    ScoreDkgMiners,
    _credit,
    _last_scoring_moment,
)
from neurons.validator.utils.logger.logger import NuminousLogger

MINER = DkgMinerModel(miner_uid=1, miner_hotkey="5MinerHotkey")
OTHER_MINER = DkgMinerModel(miner_uid=2, miner_hotkey="5OtherMinerHotkey")
IDLE_MINER = DkgMinerModel(miner_uid=3, miner_hotkey="5IdleMinerHotkey")
NOW = datetime(2026, 10, 20, 15, 10, tzinfo=timezone.utc)
MOMENT = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)


def rights_of(miner: DkgMinerModel, rights: int, statements: int) -> DkgForecastRightsModel:
    return DkgForecastRightsModel(
        miner_uid=miner.miner_uid,
        miner_hotkey=miner.miner_hotkey,
        rights=rights,
        statements=statements,
    )


class TestCredit:
    @pytest.mark.parametrize(
        "rights, statements, credit",
        [(1, 1, 1.0), (1, 2, 0.7071), (2, 2, 1.4142), (1, 5, 0.4472), (5, 5, 2.2361)],
    )
    def test_share_right_times_the_square_root_bonus(
        self, rights: int, statements: int, credit: float
    ):
        assert _credit(rights_of(MINER, rights, statements)) == pytest.approx(credit, abs=1e-4)

    @pytest.mark.parametrize(
        "exponent, credit",
        [(0, 0.2), (0.5, 0.4472), (1, 1.0)],
    )
    def test_the_bonus_exponent_sets_how_much_more_statements_pay(
        self, monkeypatch: pytest.MonkeyPatch, exponent: float, credit: float
    ):
        monkeypatch.setattr(score_dkg_miners, "STATEMENT_BONUS_EXPONENT", exponent)

        assert _credit(rights_of(MINER, 1, 5)) == pytest.approx(credit, abs=1e-4)


class TestLastScoringMoment:
    @pytest.mark.parametrize(
        "now, moment",
        [
            (datetime(2026, 10, 20, 15, 0), datetime(2026, 10, 20, 15, 0)),
            (datetime(2026, 10, 20, 15, 0, 1), datetime(2026, 10, 20, 15, 0)),
            (datetime(2026, 10, 20, 23, 59, 59), datetime(2026, 10, 20, 15, 0)),
            (datetime(2026, 10, 20, 14, 59, 59), datetime(2026, 10, 19, 15, 0)),
            (datetime(2026, 10, 20, 0, 30), datetime(2026, 10, 19, 15, 0)),
            (datetime(2026, 11, 1, 10, 0), datetime(2026, 10, 31, 15, 0)),
            (datetime(2027, 1, 1, 9, 0), datetime(2026, 12, 31, 15, 0)),
        ],
    )
    def test_the_last_daily_scoring_time_that_has_passed(self, now: datetime, moment: datetime):
        utc_now = now.replace(tzinfo=timezone.utc)

        assert _last_scoring_moment(utc_now) == moment.replace(tzinfo=timezone.utc)


class TestScoreDkgMiners:
    @pytest.fixture
    def dkg_operations(self) -> AsyncMock:
        dkg_operations = AsyncMock(spec=DkgDatabaseOperations)
        dkg_operations.get_last_scored_at.return_value = MOMENT - timedelta(days=1)
        dkg_operations.get_last_results_pull.return_value = MOMENT + timedelta(minutes=7)
        dkg_operations.get_current_miners.return_value = [MINER, OTHER_MINER, IDLE_MINER]
        dkg_operations.get_forecast_rights.return_value = []
        return dkg_operations

    @pytest.fixture
    def logger(self) -> MagicMock:
        return MagicMock(spec=NuminousLogger)

    @pytest.fixture
    def task(self, dkg_operations: AsyncMock, logger: MagicMock) -> ScoreDkgMiners:
        return ScoreDkgMiners(interval_seconds=300.0, dkg_operations=dkg_operations, logger=logger)

    def saved_scores(self, dkg_operations: AsyncMock) -> dict[tuple[int, str], float]:
        (scores,) = dkg_operations.save_scores.call_args.args
        return {(score.miner_uid, score.miner_hotkey): score.score for score in scores}

    @pytest.mark.parametrize(
        "changed, error",
        [
            ({"interval_seconds": 300}, ValueError),
            ({"interval_seconds": 0.0}, ValueError),
            ({"dkg_operations": "not operations"}, TypeError),
            ({"logger": "not a logger"}, TypeError),
        ],
    )
    def test_init_rejects_bad_arguments(
        self, dkg_operations: AsyncMock, logger: MagicMock, changed: dict, error: type[Exception]
    ):
        arguments = {
            "interval_seconds": 300.0,
            "dkg_operations": dkg_operations,
            "logger": logger,
        }

        with pytest.raises(error):
            ScoreDkgMiners(**(arguments | changed))

    def test_name_and_interval(self, task: ScoreDkgMiners):
        assert task.name == "score-dkg-miners"
        assert task.interval_seconds == 300.0

    @freeze_time(NOW)
    async def test_scores_every_current_miner_as_of_the_moment(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, logger: MagicMock
    ):
        dkg_operations.get_forecast_rights.return_value = [
            rights_of(MINER, 1, 1),
            rights_of(MINER, 1, 5),
            rights_of(OTHER_MINER, 2, 2),
        ]

        await task.run()

        dkg_operations.get_forecast_rights.assert_awaited_once_with(
            judged_after=MOMENT - SCORE_WINDOW, judged_before=MOMENT
        )
        (scores,) = dkg_operations.save_scores.call_args.args
        assert {score.scored_at for score in scores} == {MOMENT}
        saved = self.saved_scores(dkg_operations)
        assert saved.keys() == {
            (MINER.miner_uid, MINER.miner_hotkey),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey),
            (IDLE_MINER.miner_uid, IDLE_MINER.miner_hotkey),
        }
        assert saved[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(1.4472, abs=1e-4)
        assert saved[(OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey)] == pytest.approx(
            1.4142, abs=1e-4
        )
        assert saved[(IDLE_MINER.miner_uid, IDLE_MINER.miner_hotkey)] == 0.0
        logger.info.assert_called_once_with(
            "DKG miners scored",
            extra={"moment": MOMENT.isoformat(), "miners": 3, "miners_with_a_score": 2},
        )

    @freeze_time(NOW)
    async def test_every_score_carries_the_version_that_scored_it(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(score_dkg_miners, "spec_version", 4021)
        dkg_operations.get_forecast_rights.return_value = [rights_of(MINER, 1, 1)]

        await task.run()

        (scores,) = dkg_operations.save_scores.call_args.args
        assert [score.spec_version for score in scores] == [4021, 4021, 4021]

    @freeze_time(NOW)
    async def test_a_moment_already_scored_is_not_scored_again(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        dkg_operations.get_last_scored_at.return_value = MOMENT

        await task.run()

        dkg_operations.get_last_results_pull.assert_not_awaited()
        dkg_operations.get_forecast_rights.assert_not_awaited()
        dkg_operations.save_scores.assert_not_awaited()

    @freeze_time(NOW)
    async def test_the_first_moment_is_scored_when_nothing_was_scored_yet(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        dkg_operations.get_last_scored_at.return_value = None

        await task.run()

        dkg_operations.save_scores.assert_awaited_once()

    @pytest.mark.parametrize(
        "last_results_pull",
        [None, MOMENT - timedelta(minutes=8), MOMENT],
    )
    @freeze_time(NOW)
    async def test_waits_for_a_results_pull_that_started_after_the_moment(
        self,
        task: ScoreDkgMiners,
        dkg_operations: AsyncMock,
        logger: MagicMock,
        last_results_pull: datetime | None,
    ):
        dkg_operations.get_last_results_pull.return_value = last_results_pull

        await task.run()

        dkg_operations.get_forecast_rights.assert_not_awaited()
        dkg_operations.save_scores.assert_not_awaited()
        logger.debug.assert_called_once_with(
            "DKG scoring waits for a results pull after its moment",
            extra={"moment": MOMENT.isoformat()},
        )

    @freeze_time(datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc))
    async def test_before_the_daily_time_the_moment_is_the_day_before(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        dkg_operations.get_last_scored_at.return_value = MOMENT - timedelta(days=1)
        dkg_operations.get_last_results_pull.return_value = datetime(
            2026, 10, 21, 8, 52, tzinfo=timezone.utc
        )

        await task.run()

        (scores,) = dkg_operations.save_scores.call_args.args
        assert {score.scored_at for score in scores} == {MOMENT}

    @freeze_time(NOW)
    async def test_rights_of_a_miner_no_longer_registered_are_ignored(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        deregistered = DkgMinerModel(miner_uid=MINER.miner_uid, miner_hotkey="5Deregistered")
        dkg_operations.get_forecast_rights.return_value = [rights_of(deregistered, 1, 1)]

        await task.run()

        scores = self.saved_scores(dkg_operations)
        assert (deregistered.miner_uid, deregistered.miner_hotkey) not in scores
        assert set(scores.values()) == {0.0}

    @freeze_time(NOW)
    async def test_no_current_miners_saves_an_empty_run(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, logger: MagicMock
    ):
        dkg_operations.get_current_miners.return_value = []
        dkg_operations.get_forecast_rights.return_value = [rights_of(MINER, 1, 1)]

        await task.run()

        dkg_operations.save_scores.assert_awaited_once_with([])
        logger.info.assert_called_once_with(
            "DKG miners scored",
            extra={"moment": MOMENT.isoformat(), "miners": 0, "miners_with_a_score": 0},
        )


class TestScoreDkgMinersEndToEnd:
    async def add_forecast(
        self,
        dkg_operations: DkgDatabaseOperations,
        miner_uid: int,
        miner_hotkey: str,
        question_id: str,
        statements: int,
        rights: dict[int, tuple[str, datetime]],
    ) -> None:
        await dkg_operations.save_forecast_hashes(
            [
                DkgForecastHashModel(
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    question_id=question_id,
                    sealed_sha256="sha256",
                    received_at=datetime(2026, 10, 15, 18, tzinfo=timezone.utc),
                )
            ]
        )
        await dkg_operations.save_forecasts(
            [
                DkgForecastModel(
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    question_id=question_id,
                    opened=json.dumps({"version": 2, "statements": [{}] * statements}),
                    invalid_reason=None,
                    failed_check=None,
                )
            ],
            [
                DkgStatementResultModel(
                    miner_uid=miner_uid,
                    miner_hotkey=miner_hotkey,
                    question_id=question_id,
                    statement_position=position,
                    outcome=position in rights,
                    reason="judged",
                    event_id=rights[position][0] if position in rights else None,
                    reported_at=None,
                    judged_at=rights[position][1] if position in rights else MOMENT,
                )
                for position in range(statements)
            ],
        )

    @freeze_time(NOW)
    async def test_round_b_of_staging(self, db_client: DatabaseClient):
        dkg_operations = DkgDatabaseOperations(db_client=db_client)
        await db_client.insert_many(
            "INSERT INTO miners (miner_uid, miner_hotkey, blocktime) VALUES (?, ?, ?)",
            [("207", "5StgMiner3", 500), ("196", "5StgMiner", 500), ("197", "5StgMiner2", 500)],
        )
        await dkg_operations.save_round(
            DkgRoundModel(
                round_id="round-b",
                deadline_at=datetime(2026, 10, 15, 19, tzinfo=timezone.utc),
                horizon_end_at=datetime(2026, 10, 16, 19, tzinfo=timezone.utc),
            ),
            [
                DkgQuestionModel(
                    question_id=question_id,
                    round_id="round-b",
                    entity_id=f"entity-{question_id}",
                    entity=question_id,
                    counterparty_id=None,
                    counterparty=None,
                    relationship=None,
                )
                for question_id in ["netflix", "visa", "ibm"]
            ],
        )
        judged = MOMENT - timedelta(hours=8)
        await self.add_forecast(
            dkg_operations,
            207,
            "5StgMiner3",
            "netflix",
            2,
            {0: ("event-netflix", judged), 1: ("event-netflix", judged)},
        )
        await self.add_forecast(
            dkg_operations, 207, "5StgMiner3", "visa", 5, {0: ("event-visa", judged)}
        )
        await self.add_forecast(
            dkg_operations, 196, "5StgMiner", "visa", 2, {0: ("event-visa", judged)}
        )
        await self.add_forecast(
            dkg_operations,
            196,
            "5StgMiner",
            "ibm",
            1,
            {0: ("event-ibm", MOMENT + timedelta(minutes=5))},
        )
        await dkg_operations.record_results_pull(MOMENT + timedelta(minutes=7))
        task = ScoreDkgMiners(
            interval_seconds=300.0,
            dkg_operations=dkg_operations,
            logger=MagicMock(spec=NuminousLogger),
        )

        await task.run()
        await task.run()

        rows = await db_client.many(
            "SELECT miner_uid, scored_at, score FROM dkg_scores ORDER BY miner_uid"
        )
        assert [(row[0], row[1], round(row[2], 4)) for row in rows] == [
            (196, "2026-10-20T15:00:00Z", 0.7071),
            (197, "2026-10-20T15:00:00Z", 0.0),
            (207, "2026-10-20T15:00:00Z", 1.4472),
        ]
