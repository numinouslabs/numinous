import json
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgDistributionStatementModel,
    DkgForecastHashModel,
    DkgForecastModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgStatementResultModel,
)

MINER = (1, "5MinerHotkey")
OTHER_MINER = (2, "5OtherMinerHotkey")
DEADLINE = datetime(2026, 11, 19, tzinfo=timezone.utc)
MOMENT = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
WINDOW_START = MOMENT - timedelta(days=60)
RECEIVED = DEADLINE - timedelta(hours=2)
JUDGED = DEADLINE + timedelta(days=2, hours=7)
APPEARED = DEADLINE + timedelta(days=1, hours=5)
QUESTIONS = ["question-1", "question-2"]

EARLY = [0.2, 0.5, 0.1] + [0.0] * 27 + [0.2]
LATE = [0.0] * 29 + [0.6, 0.4]
UNLIKELY = [0.1] + [0.0] * 29 + [0.9]
HOUSE = [0.05, 0.05, 0.05] + [0.0] * 27 + [0.85]
ITS_HOUSE_FOR_EVERY_STATEMENT = "its house for every statement"


class Result(NamedTuple):
    position: int
    outcome: bool
    reason: str
    judged_at: datetime
    event_id: str | None
    reported_at: datetime | None


def right(position: int, event_id: str, judged_at: datetime = JUDGED) -> Result:
    return Result(position, True, "MATCHED", judged_at, event_id, APPEARED)


def wrong(position: int, reason: str = "NOT_REPORTED", judged_at: datetime = JUDGED) -> Result:
    return Result(position, False, reason, judged_at, None, None)


async def add_forecast(
    dkg_operations: DkgDatabaseOperations,
    miner: tuple[int, str],
    question_id: str,
    statements: list[dict[str, list[float] | float]],
    results: tuple[Result, ...] = (),
    version: int = 3,
    received_at: datetime = RECEIVED,
    failed_check: str | None = None,
    baselines: list[list[float] | None] | None | str = ITS_HOUSE_FOR_EVERY_STATEMENT,
    sent_whole: bool = False,
) -> None:
    miner_uid, miner_hotkey = miner
    with_distributions = version == 3
    if baselines == ITS_HOUSE_FOR_EVERY_STATEMENT:
        baselines = [HOUSE] * len(statements)
    await dkg_operations.save_forecast_hashes(
        [
            DkgForecastHashModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                sealed_sha256=f"sha256-of-{miner_uid}-{question_id}",
                received_at=received_at,
            )
        ]
    )
    await dkg_operations.save_forecasts(
        [
            DkgForecastModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                opened=(
                    json.dumps({"version": version, "statements": statements})
                    if sent_whole or not with_distributions
                    else None
                ),
                distributions=(
                    json.dumps([statement["distribution"] for statement in statements])
                    if with_distributions
                    else None
                ),
                baselines=json.dumps(baselines) if with_distributions and baselines else None,
                invalid_reason=None,
                failed_check=failed_check,
            )
        ],
        [
            DkgStatementResultModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                statement_position=result.position,
                outcome=result.outcome,
                reason=result.reason,
                event_id=result.event_id,
                reported_at=result.reported_at,
                judged_at=result.judged_at,
                baseline=None,
            )
            for result in results
        ],
    )


def with_distributions(*distributions: list[float]) -> list[dict[str, list[float] | float]]:
    return [{"distribution": distribution} for distribution in distributions]


async def counted_statements(
    dkg_operations: DkgDatabaseOperations, moment: datetime = MOMENT
) -> list[tuple[int, str, list[float], datetime | None]]:
    statements = await dkg_operations.get_distribution_statements(
        deadline_after=WINDOW_START, moment=moment
    )
    return sorted(
        (
            statement.miner_uid,
            statement.question_id,
            statement.distribution,
            statement.appeared_at,
        )
        for statement in statements
    )


class TestDistributionStatements:
    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        dkg_operations = DkgDatabaseOperations(db_client=db_client)
        await self.add_round(dkg_operations, "round-1", DEADLINE, QUESTIONS)
        return dkg_operations

    async def add_round(
        self,
        dkg_operations: DkgDatabaseOperations,
        round_id: str,
        deadline_at: datetime,
        question_ids: list[str],
    ) -> None:
        await dkg_operations.save_round(
            DkgRoundModel(
                round_id=round_id,
                deadline_at=deadline_at,
                horizon_end_at=deadline_at + timedelta(days=30),
            ),
            [
                DkgQuestionModel(
                    question_id=question_id,
                    round_id=round_id,
                    entity_id="entity-israel",
                    entity="Israel",
                    counterparty_id=None,
                    counterparty=None,
                    relationship=None,
                )
                for question_id in question_ids
            ],
        )

    async def test_a_statement_counts_before_it_has_any_result(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", with_distributions(EARLY, LATE))

        assert await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        ) == [
            DkgDistributionStatementModel(
                miner_uid=1,
                miner_hotkey="5MinerHotkey",
                question_id="question-1",
                deadline_at=DEADLINE,
                distribution=json.dumps(distribution),
                baseline=json.dumps(HOUSE),
                appeared_at=None,
                event_id=None,
            )
            for distribution in (EARLY, LATE)
        ]

    async def test_each_statement_carries_its_own_baseline(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE, UNLIKELY),
            baselines=[HOUSE, None, UNLIKELY],
        )

        statements = await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        )

        assert {tuple(statement.distribution): statement.baseline for statement in statements} == {
            tuple(EARLY): HOUSE,
            tuple(LATE): None,
            tuple(UNLIKELY): UNLIKELY,
        }

    async def test_a_forecast_without_baselines_still_counts_its_statements(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", with_distributions(EARLY, LATE), baselines=None
        )

        statements = await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        )

        assert [statement.baseline for statement in statements] == [None, None]

    async def test_a_forecast_counts_the_same_before_and_after_it_is_sent_whole(
        self, dkg_operations: DkgDatabaseOperations
    ):
        results = (right(0, "event-1"), wrong(1))
        await add_forecast(
            dkg_operations, MINER, "question-1", with_distributions(EARLY, LATE), results
        )
        not_whole_yet = await counted_statements(dkg_operations)

        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE),
            results,
            sent_whole=True,
        )

        assert await counted_statements(dkg_operations) == not_whole_yet
        assert not_whole_yet == [
            (1, "question-1", LATE, None),
            (1, "question-1", EARLY, APPEARED),
        ]

    async def test_a_forecast_sent_whole_that_fails_its_check_drops_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", with_distributions(EARLY))
        assert await counted_statements(dkg_operations) == [(1, "question-1", EARLY, None)]

        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            sent_whole=True,
            failed_check="the sealed forecast does not match its hash",
        )

        assert await counted_statements(dkg_operations) == []

    async def test_only_a_statement_that_appeared_carries_its_event(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE, UNLIKELY),
            (right(0, "event-1"), wrong(1)),
        )

        statements = await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        )

        assert {tuple(statement.distribution): statement.event_id for statement in statements} == {
            tuple(EARLY): "event-1",
            tuple(LATE): None,
            tuple(UNLIKELY): None,
        }

    async def test_a_right_judged_after_the_moment_carries_no_event_yet(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            (right(0, "event-1", MOMENT + timedelta(hours=3)),),
        )

        (statement,) = await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        )

        assert (statement.appeared_at, statement.event_id) == (None, None)

    async def test_miners_right_through_the_same_event_carry_the_same_event(
        self, dkg_operations: DkgDatabaseOperations
    ):
        for miner in (MINER, OTHER_MINER):
            await add_forecast(
                dkg_operations, miner, "question-1", with_distributions(EARLY), (right(0, "e"),)
            )

        statements = await dkg_operations.get_distribution_statements(
            deadline_after=WINDOW_START, moment=MOMENT
        )

        assert sorted((statement.miner_uid, statement.event_id) for statement in statements) == [
            (1, "e"),
            (2, "e"),
        ]

    async def test_a_right_appeared_when_its_event_was_reported(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE),
            (right(0, "event-1"),),
        )

        assert await counted_statements(dkg_operations) == [
            (1, "question-1", LATE, None),
            (1, "question-1", EARLY, APPEARED),
        ]

    @pytest.mark.parametrize("reason", ["NOT_REPORTED", "ALREADY_HELD", "NOT_NAMED"])
    async def test_a_wrong_counts_as_not_appeared(
        self, dkg_operations: DkgDatabaseOperations, reason: str
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", with_distributions(EARLY), (wrong(0, reason),)
        )

        assert await counted_statements(dkg_operations) == [(1, "question-1", EARLY, None)]

    async def test_a_void_is_left_out(self, dkg_operations: DkgDatabaseOperations):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE),
            (wrong(0, "VOID"),),
        )

        assert await counted_statements(dkg_operations) == [(1, "question-1", LATE, None)]

    async def test_the_same_event_twice_counts_once(self, dkg_operations: DkgDatabaseOperations):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE, UNLIKELY),
            (right(0, "event-1"), right(2, "event-1")),
        )

        assert await counted_statements(dkg_operations) == [
            (1, "question-1", LATE, None),
            (1, "question-1", EARLY, APPEARED),
        ]

    async def test_the_same_event_in_another_question_counts_in_the_forecast_received_first(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            (right(0, "event-1"),),
            received_at=RECEIVED + timedelta(minutes=5),
        )
        await add_forecast(
            dkg_operations, MINER, "question-2", with_distributions(LATE), (right(0, "event-1"),)
        )

        assert await counted_statements(dkg_operations) == [(1, "question-2", LATE, APPEARED)]

    async def test_each_miner_keeps_its_own_right_through_the_same_event(
        self, dkg_operations: DkgDatabaseOperations
    ):
        for miner in (MINER, OTHER_MINER):
            await add_forecast(
                dkg_operations, miner, "question-1", with_distributions(EARLY), (right(0, "e"),)
            )

        assert await counted_statements(dkg_operations) == [
            (1, "question-1", EARLY, APPEARED),
            (2, "question-1", EARLY, APPEARED),
        ]

    @pytest.mark.parametrize("judged_at", [MOMENT, MOMENT + timedelta(hours=3)])
    async def test_a_result_judged_at_or_after_the_moment_is_not_seen_yet(
        self, dkg_operations: DkgDatabaseOperations, judged_at: datetime
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY, LATE, UNLIKELY),
            (right(0, "event-1", judged_at), wrong(1, "VOID", judged_at), right(2, "event-1")),
        )

        assert await counted_statements(dkg_operations) == [
            (1, "question-1", LATE, None),
            (1, "question-1", UNLIKELY, APPEARED),
            (1, "question-1", EARLY, None),
        ]

    async def test_a_forecast_with_one_probability_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", [{"probability": 0.3}], (wrong(0),), version=2
        )

        assert await counted_statements(dkg_operations) == []

    async def test_a_forecast_that_failed_a_check_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            failed_check="the miner signature does not verify",
        )

        assert await counted_statements(dkg_operations) == []

    async def test_a_forecast_that_failed_a_check_does_not_take_the_event_from_a_later_one(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            (right(0, "event-1"),),
            failed_check="the miner signature does not verify",
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            with_distributions(LATE),
            (right(0, "event-1"),),
            received_at=RECEIVED + timedelta(minutes=5),
        )

        assert await counted_statements(dkg_operations) == [(1, "question-2", LATE, APPEARED)]

    async def test_an_invalid_forecast_is_left_out(self, dkg_operations: DkgDatabaseOperations):
        await dkg_operations.save_forecast_hashes(
            [
                DkgForecastHashModel(
                    miner_uid=1,
                    miner_hotkey="5MinerHotkey",
                    question_id="question-1",
                    sealed_sha256="sha256-of-the-invalid-one",
                    received_at=RECEIVED,
                )
            ]
        )
        await dkg_operations.save_forecasts(
            [
                DkgForecastModel(
                    miner_uid=1,
                    miner_hotkey="5MinerHotkey",
                    question_id="question-1",
                    opened=None,
                    distributions=None,
                    baselines=None,
                    invalid_reason="the seal does not open",
                    failed_check=None,
                )
            ],
            [],
        )

        assert await counted_statements(dkg_operations) == []

    @pytest.mark.parametrize(
        "deadline_at, counts",
        [
            (WINDOW_START - timedelta(seconds=1), False),
            (WINDOW_START, False),
            (WINDOW_START + timedelta(seconds=1), True),
            (MOMENT - timedelta(seconds=1), True),
            (MOMENT, False),
            (MOMENT + timedelta(hours=9), False),
        ],
    )
    async def test_a_forecast_counts_from_its_deadline_until_the_window_ends(
        self, dkg_operations: DkgDatabaseOperations, deadline_at: datetime, counts: bool
    ):
        await self.add_round(dkg_operations, "another-round", deadline_at, ["another-question"])
        await add_forecast(dkg_operations, MINER, "another-question", with_distributions(EARLY))

        counted = await counted_statements(dkg_operations)

        assert counted == ([(1, "another-question", EARLY, None)] if counts else [])


class TestOneProbabilityAndDistributionForecastsTogether:
    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        dkg_operations = DkgDatabaseOperations(db_client=db_client)
        await TestDistributionStatements().add_round(dkg_operations, "round-1", DEADLINE, QUESTIONS)
        return dkg_operations

    async def forecast_points(
        self, dkg_operations: DkgDatabaseOperations
    ) -> list[tuple[int, float, int]]:
        return [
            (forecast.miner_uid, forecast.points, forecast.statements)
            for forecast in await dkg_operations.get_forecast_points(
                judged_after=MOMENT - timedelta(days=30), judged_before=MOMENT
            )
        ]

    @pytest.mark.parametrize("sent_whole", [False, True])
    async def test_a_distribution_forecast_is_left_out_of_the_forecast_points(
        self, dkg_operations: DkgDatabaseOperations, sent_whole: bool
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            with_distributions(EARLY),
            (right(0, "event-1"),),
            sent_whole=sent_whole,
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            [{"probability": 0.3}],
            (right(0, "event-2"),),
            version=2,
        )

        assert await self.forecast_points(dkg_operations) == [(1, 1.0, 1)]

    async def test_an_event_counted_in_a_distribution_forecast_is_not_paid_again(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", with_distributions(EARLY), (right(0, "event-1"),)
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            [{"probability": 0.3}, {"probability": 0.3}],
            (right(0, "event-1"), right(1, "event-2")),
            version=2,
            received_at=RECEIVED + timedelta(minutes=5),
        )

        assert await self.forecast_points(dkg_operations) == [(1, 1.0, 1)]
        assert await counted_statements(dkg_operations) == [(1, "question-1", EARLY, APPEARED)]

    async def test_an_event_paid_in_a_one_probability_forecast_leaves_out_the_distribution_statement(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            [{"probability": 0.3}],
            (right(0, "event-1"),),
            version=2,
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            with_distributions(EARLY, LATE),
            (right(0, "event-1"),),
            received_at=RECEIVED + timedelta(minutes=5),
        )

        assert await self.forecast_points(dkg_operations) == [(1, 1.0, 1)]
        assert await counted_statements(dkg_operations) == [(1, "question-2", LATE, None)]
