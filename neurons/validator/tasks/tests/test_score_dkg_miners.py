import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

import neurons.validator.tasks.score_dkg_miners as score_dkg_miners
from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgDistributionStatementModel,
    DkgForecastHashModel,
    DkgForecastModel,
    DkgForecastPointsModel,
    DkgMinerModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgStatementResultModel,
)
from neurons.validator.tasks.score_dkg_miners import (
    DISTRIBUTION_SCORE_WINDOW,
    SCORE_WINDOW,
    ScoreDkgMiners,
    _add_credits,
    _credit,
    _distribution_credits,
    _last_scoring_moment,
)
from neurons.validator.utils.common.distribution_scoring import statement_points
from neurons.validator.utils.logger.logger import NuminousLogger

MINER = DkgMinerModel(miner_uid=1, miner_hotkey="5MinerHotkey")
OTHER_MINER = DkgMinerModel(miner_uid=2, miner_hotkey="5OtherMinerHotkey")
IDLE_MINER = DkgMinerModel(miner_uid=3, miner_hotkey="5IdleMinerHotkey")
NOW = datetime(2026, 10, 20, 15, 10, tzinfo=timezone.utc)
MOMENT = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)


def points_of(miner: DkgMinerModel, points: float, statements: int) -> DkgForecastPointsModel:
    return DkgForecastPointsModel(
        miner_uid=miner.miner_uid,
        miner_hotkey=miner.miner_hotkey,
        points=points,
        statements=statements,
    )


DEADLINE = MOMENT - timedelta(days=5, hours=15)
APPEARED_ON_DAY_TWO = DEADLINE + timedelta(days=1, hours=5)
EARLY = [0.2, 0.5, 0.1] + [0.0] * 27 + [0.2]
UNLIKELY = [0.1, 0.1, 0.1] + [0.0] * 27 + [0.7]
HOUSE = [0.05, 0.05, 0.05] + [0.0] * 27 + [0.85]
EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO = 0.04 + 0.09 + 28 * 0.04
UNLIKELY_ERRORS_AFTER_FIVE_DAYS = 0.01 + 0.04 + 0.09 + 0.09 + 0.09
HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO = 0.0025 + 0.81 + 28 * 0.7225
HOUSE_ERRORS_WHEN_NOTHING_APPEARS = 0.0025 + 0.01 + 0.0225 + 0.0225 + 0.0225
UNLIKELY_POINTS = (HOUSE_ERRORS_WHEN_NOTHING_APPEARS - UNLIKELY_ERRORS_AFTER_FIVE_DAYS) / 30


def statement_of(
    miner: DkgMinerModel,
    question_id: str,
    distribution: list[float],
    appeared_at: datetime | None = None,
    baseline: list[float] | None = HOUSE,
    deadline_at: datetime = DEADLINE,
    event_id: str | None = None,
) -> DkgDistributionStatementModel:
    its_own_event = uuid.uuid4().hex if appeared_at else None
    return DkgDistributionStatementModel(
        miner_uid=miner.miner_uid,
        miner_hotkey=miner.miner_hotkey,
        question_id=question_id,
        deadline_at=deadline_at,
        distribution=json.dumps(distribution),
        baseline=json.dumps(baseline) if baseline else None,
        appeared_at=appeared_at,
        event_id=event_id or its_own_event,
    )


def credits_per_miner(
    statements: list[DkgDistributionStatementModel], moment: datetime
) -> dict[tuple[int, str], float]:
    every_miner = {(statement.miner_uid, statement.miner_hotkey) for statement in statements}
    credits: dict[tuple[int, str], float] = {}
    for miner, credit in _distribution_credits(statements, moment, every_miner):
        credits[miner] = credits.get(miner, 0.0) + credit
    return credits


class TestDistributionCredits:
    def test_one_credit_per_forecast(self):
        statements = [
            statement_of(MINER, "question-1", UNLIKELY),
            statement_of(MINER, "question-1", UNLIKELY),
            statement_of(MINER, "question-2", UNLIKELY),
            statement_of(OTHER_MINER, "question-1", UNLIKELY),
        ]

        registered = {
            (MINER.miner_uid, MINER.miner_hotkey),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey),
        }

        credits = _distribution_credits(statements, MOMENT, registered)

        assert sorted(credits) == [
            ((MINER.miner_uid, MINER.miner_hotkey), pytest.approx(2 * UNLIKELY_POINTS / 2**0.5)),
            ((MINER.miner_uid, MINER.miner_hotkey), pytest.approx(UNLIKELY_POINTS)),
            ((OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey), pytest.approx(UNLIKELY_POINTS)),
        ]

    def test_no_statements_no_credits(self):
        assert credits_per_miner([], MOMENT) == {}

    def test_a_statement_that_appeared_earns_what_it_beats_the_house_by_over_the_horizon(self):
        statements = [statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO)]

        credits = credits_per_miner(statements, MOMENT)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                (HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO)
                / 30
            )
        }

    def test_a_statement_not_appeared_is_scored_as_not_yet_against_the_house(self):
        credits = credits_per_miner([statement_of(MINER, "question-1", UNLIKELY)], MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(UNLIKELY_POINTS)
        assert UNLIKELY_POINTS < 0

    @pytest.mark.parametrize("appeared_at", [None, APPEARED_ON_DAY_TWO])
    def test_the_same_numbers_as_the_house_earn_exactly_zero(self, appeared_at: datetime | None):
        credits = credits_per_miner([statement_of(MINER, "question-1", HOUSE, appeared_at)], MOMENT)

        assert credits == {(MINER.miner_uid, MINER.miner_hotkey): 0.0}

    def test_a_statement_without_a_baseline_earns_nothing_and_still_counts(self):
        statements = [
            statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO),
            statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO, baseline=None),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(
            (HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO)
            / 30
            / 2**0.5
        )

    def test_a_forecast_without_any_baseline_earns_nothing(self):
        statements = [
            statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO, baseline=None),
            statement_of(MINER, "question-1", UNLIKELY, baseline=None),
        ]

        assert credits_per_miner(statements, MOMENT) == {(MINER.miner_uid, MINER.miner_hotkey): 0.0}

    def test_the_statements_of_a_forecast_share_the_square_root_of_their_number(self):
        statements = [
            statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO),
            statement_of(MINER, "question-1", UNLIKELY),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(
            (
                (HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO)
                / 30
                + UNLIKELY_POINTS
            )
            / 2**0.5
        )

    def test_the_forecasts_of_a_miner_add_up_and_each_miner_keeps_its_own(self):
        statements = [
            statement_of(MINER, "question-1", UNLIKELY),
            statement_of(MINER, "question-2", UNLIKELY),
            statement_of(OTHER_MINER, "question-1", UNLIKELY),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(2 * UNLIKELY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(UNLIKELY_POINTS),
        }

    def test_each_forecast_counts_the_days_of_its_own_round(self):
        statements = [
            statement_of(MINER, "question-1", UNLIKELY, deadline_at=MOMENT - timedelta(hours=15)),
            statement_of(MINER, "question-2", UNLIKELY, deadline_at=MOMENT - timedelta(days=45)),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(
            ((0.0025 + 0.01 + 28 * 0.0225) - (0.01 + 0.04 + 28 * 0.09)) / 30
        )


EARLY_POINTS = (
    HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO
) / 30
NEVER = [0.0] * 30 + [1.0]
ALL_ON_DAY_FOUR = [0.0, 0.0, 0.0, 1.0] + [0.0] * 27
APPEARED_ON_DAY_FOUR = DEADLINE + timedelta(days=3, hours=5)


class TestAStatementThatAppeared:
    @pytest.mark.parametrize(
        "day_it_appeared, days_since_the_deadline",
        [(1, 1), (1, 5), (3, 5), (5, 5), (5, 12), (12, 29), (30, 30), (30, 45)],
    )
    def test_is_scored_on_the_whole_horizon_at_once(
        self, day_it_appeared: int, days_since_the_deadline: int
    ):
        moment = DEADLINE + timedelta(days=days_since_the_deadline, hours=15)
        appeared_at = DEADLINE + timedelta(days=day_it_appeared - 1, hours=5)

        credits = credits_per_miner([statement_of(MINER, "question-1", EARLY, appeared_at)], moment)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): statement_points(EARLY, HOUSE, day_it_appeared)
        }

    def test_its_points_no_longer_move_as_the_days_pass(self):
        statement = statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO)

        credits = [
            credits_per_miner([statement], DEADLINE + timedelta(days=days, hours=15))
            for days in (2, 5, 12, 30, 59)
        ]

        assert all(credit == credits[0] for credit in credits)

    def test_all_on_never_takes_its_whole_loss_the_day_it_appears(self):
        appeared_on_day_five = DEADLINE + timedelta(days=4, hours=5)
        house_errors = 0.0025 + 0.01 + 0.0225 + 0.0225 + 26 * 0.7225
        errors_of_never = 26 * 1.0

        credits = credits_per_miner(
            [statement_of(MINER, "question-1", NEVER, appeared_on_day_five)], MOMENT
        )

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                (house_errors - errors_of_never) / 30
            )
        }
        assert (house_errors - errors_of_never) / 30 < -0.2

    def test_the_day_before_it_appears_it_is_still_scored_on_the_days_passed(self):
        four_days_in = DEADLINE + timedelta(days=4, hours=15)

        credits = credits_per_miner([statement_of(MINER, "question-1", UNLIKELY)], four_days_in)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                ((0.0025 + 0.01 + 0.0225 + 0.0225) - (0.01 + 0.04 + 0.09 + 0.09)) / 30
            )
        }


class TestAStatementThatHasNotAppeared:
    def test_all_on_never_earns_nothing_while_it_does_not_appear(self):
        assert HOUSE_ERRORS_WHEN_NOTHING_APPEARS > 0

        credits = credits_per_miner([statement_of(MINER, "question-1", NEVER)], MOMENT)

        assert credits == {(MINER.miner_uid, MINER.miner_hotkey): 0.0}

    def test_worse_than_the_house_keeps_its_points_below_zero(self):
        credits = credits_per_miner([statement_of(MINER, "question-1", UNLIKELY)], MOMENT)

        assert credits == {(MINER.miner_uid, MINER.miner_hotkey): pytest.approx(UNLIKELY_POINTS)}

    def test_once_it_appears_the_days_before_it_appeared_pay_too(self):
        statements = [statement_of(MINER, "question-1", ALL_ON_DAY_FOUR, APPEARED_ON_DAY_FOUR)]
        house_errors_before_day_four = 0.0025 + 0.01 + 0.0225
        house_errors_from_day_four = 27 * 0.7225

        credits = credits_per_miner(statements, MOMENT)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                (house_errors_before_day_four + house_errors_from_day_four) / 30
            )
        }

    def test_points_above_zero_so_far_wait_for_the_day_it_appears(self):
        three_days_in = DEADLINE + timedelta(days=3, hours=15)
        four_days_in = DEADLINE + timedelta(days=4, hours=15)
        waiting = statement_of(MINER, "question-1", ALL_ON_DAY_FOUR)
        appeared = statement_of(MINER, "question-1", ALL_ON_DAY_FOUR, APPEARED_ON_DAY_FOUR)

        assert credits_per_miner([waiting], three_days_in) == {
            (MINER.miner_uid, MINER.miner_hotkey): 0.0
        }
        assert credits_per_miner([appeared], four_days_in) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                (0.0025 + 0.01 + 0.0225 + 27 * 0.7225) / 30
            )
        }

    @pytest.mark.parametrize("days_after_the_deadline", [30, 31, 45])
    def test_reported_after_the_horizon_all_on_never_earns_nothing(
        self, days_after_the_deadline: int
    ):
        long_ago = MOMENT - timedelta(days=50)
        reported_late = long_ago + timedelta(days=days_after_the_deadline, hours=1)
        statements = [statement_of(MINER, "question-1", NEVER, reported_late, deadline_at=long_ago)]

        assert credits_per_miner(statements, MOMENT) == {(MINER.miner_uid, MINER.miner_hotkey): 0.0}

    def test_reported_on_the_last_day_of_the_horizon_still_counts_as_appeared(self):
        long_ago = MOMENT - timedelta(days=50)
        on_day_thirty = long_ago + timedelta(days=29, hours=5)
        all_on_day_thirty = [0.0] * 29 + [1.0, 0.0]
        statements = [
            statement_of(
                MINER, "question-1", all_on_day_thirty, on_day_thirty, deadline_at=long_ago
            )
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(
            (0.0025 + 0.01 + 27 * 0.0225 + 0.7225) / 30
        )

    def test_a_statement_earning_nothing_still_counts_in_its_forecast(self):
        statements = [
            statement_of(MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO),
            statement_of(MINER, "question-1", NEVER),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2**0.5)
        }

    def test_what_a_statement_not_appeared_loses_is_not_offset_by_another_one(self):
        statements = [
            statement_of(MINER, "question-1", UNLIKELY),
            statement_of(MINER, "question-1", NEVER),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(UNLIKELY_POINTS / 2**0.5)
        }

    def test_the_numbers_of_the_house_stay_at_zero(self):
        credits = credits_per_miner([statement_of(MINER, "question-1", HOUSE)], MOMENT)

        assert credits == {(MINER.miner_uid, MINER.miner_hotkey): 0.0}


LATE = [0.0] * 29 + [0.5, 0.5]
LATE_POINTS = (HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - (28 * 1.0 + 0.25)) / 30
THIRD_MINER = DkgMinerModel(miner_uid=4, miner_hotkey="5ThirdMinerHotkey")


def right_through(
    miner: DkgMinerModel,
    event_id: str,
    distribution: list[float] = EARLY,
    question_id: str = "question-1",
    baseline: list[float] | None = HOUSE,
    deadline_at: datetime = DEADLINE,
) -> DkgDistributionStatementModel:
    appeared_at = deadline_at + timedelta(days=1, hours=5)
    return statement_of(
        miner, question_id, distribution, appeared_at, baseline, deadline_at, event_id
    )


class TestAnEventSharedAmongMiners:
    def test_two_miners_paid_for_the_same_event_get_half_each(self):
        statements = [right_through(MINER, "event-1"), right_through(OTHER_MINER, "event-1")]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2),
        }

    def test_three_miners_get_a_third_each_of_their_own_points(self):
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1"),
            right_through(THIRD_MINER, "event-1", [0.0, 1.0] + [0.0] * 29),
        ]
        best_possible_points = HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO / 30

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(EARLY_POINTS / 3)
        assert credits[(THIRD_MINER.miner_uid, THIRD_MINER.miner_hotkey)] == pytest.approx(
            best_possible_points / 3
        )

    def test_a_miner_alone_on_its_event_keeps_its_points(self):
        statements = [right_through(MINER, "event-1"), right_through(OTHER_MINER, "event-2")]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(EARLY_POINTS),
        }

    def test_points_below_zero_are_not_divided_and_do_not_count_as_a_miner(self):
        assert LATE_POINTS < 0
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1", LATE),
            right_through(THIRD_MINER, "event-1", LATE),
        ]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(LATE_POINTS),
            (THIRD_MINER.miner_uid, THIRD_MINER.miner_hotkey): pytest.approx(LATE_POINTS),
        }

    def test_a_miner_with_the_numbers_of_the_house_does_not_take_a_share(self):
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1", HOUSE),
        ]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): 0.0,
        }

    def test_a_statement_without_a_baseline_does_not_take_a_share(self):
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1", baseline=None),
        ]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): 0.0,
        }

    def test_a_statement_that_did_not_appear_is_left_as_it_is(self):
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1"),
            statement_of(MINER, "question-2", UNLIKELY),
        ]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(
                EARLY_POINTS / 2 + UNLIKELY_POINTS
            ),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2),
        }

    def test_the_share_is_taken_before_the_square_root(self):
        statements = [
            right_through(MINER, "event-1"),
            right_through(MINER, "event-2"),
            right_through(OTHER_MINER, "event-1", question_id="question-2"),
        ]

        credits = credits_per_miner(statements, MOMENT)

        assert credits[(MINER.miner_uid, MINER.miner_hotkey)] == pytest.approx(
            (EARLY_POINTS / 2 + EARLY_POINTS) / 2**0.5
        )

    def test_an_event_is_shared_across_rounds(self):
        earlier_round = DEADLINE - timedelta(days=3)
        statements = [
            right_through(MINER, "event-1", deadline_at=earlier_round),
            right_through(OTHER_MINER, "event-1"),
        ]

        assert credits_per_miner(statements, MOMENT) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(EARLY_POINTS / 2),
        }

    def test_a_miner_no_longer_registered_takes_no_share_and_earns_nothing(self):
        gone = DkgMinerModel(miner_uid=9, miner_hotkey="5Gone")
        statements = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1"),
            right_through(gone, "event-1"),
        ]
        registered = {
            (MINER.miner_uid, MINER.miner_hotkey),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey),
        }

        credits = _distribution_credits(statements, MOMENT, registered)

        assert sorted(credits) == [
            ((MINER.miner_uid, MINER.miner_hotkey), pytest.approx(EARLY_POINTS / 2)),
            ((OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey), pytest.approx(EARLY_POINTS / 2)),
        ]

    def test_a_hotkey_registered_again_under_another_uid_does_not_share_with_its_old_uid(self):
        registered_again = DkgMinerModel(miner_uid=99, miner_hotkey=MINER.miner_hotkey)
        statements = [right_through(MINER, "event-1"), right_through(registered_again, "event-1")]
        registered = {(registered_again.miner_uid, registered_again.miner_hotkey)}

        credits = _distribution_credits(statements, MOMENT, registered)

        assert credits == [
            (
                (registered_again.miner_uid, registered_again.miner_hotkey),
                pytest.approx(EARLY_POINTS),
            )
        ]

    async def test_the_scoring_run_shares_an_event_among_the_registered_miners_only(self):
        dkg_operations = AsyncMock(spec=DkgDatabaseOperations)
        dkg_operations.get_current_miners.return_value = [MINER, OTHER_MINER, IDLE_MINER]
        dkg_operations.get_forecast_points.return_value = []
        dkg_operations.get_distribution_statements.return_value = [
            right_through(MINER, "event-1"),
            right_through(OTHER_MINER, "event-1"),
            right_through(DkgMinerModel(miner_uid=9, miner_hotkey="5Gone"), "event-1"),
            right_through(MINER, "event-2", question_id="question-2"),
            right_through(DkgMinerModel(miner_uid=9, miner_hotkey="5Gone"), "event-2"),
        ]
        logger = MagicMock(spec=NuminousLogger)
        task = ScoreDkgMiners(interval_seconds=300.0, dkg_operations=dkg_operations, logger=logger)

        await task.score(MOMENT)

        (scores,) = dkg_operations.save_scores.call_args.args
        assert {score.miner_uid: score.score for score in scores} == {
            MINER.miner_uid: pytest.approx(EARLY_POINTS / 2 + EARLY_POINTS),
            OTHER_MINER.miner_uid: pytest.approx(EARLY_POINTS / 2),
            IDLE_MINER.miner_uid: 0.0,
        }
        extra = logger.info.call_args.kwargs["extra"]
        assert extra["distribution_forecasts"] == 3
        assert extra["distribution_credits"] == pytest.approx(2 * EARLY_POINTS)


class TestAddCredits:
    def test_adds_each_credit_to_its_miner_and_counts_them(self):
        miner, other_miner = (1, "5MinerHotkey"), (2, "5OtherMinerHotkey")
        scores = {miner: 0.5, other_miner: 0.0}

        counted = _add_credits(scores, [(miner, 0.25), (miner, -0.125), (other_miner, 1.0)])

        assert scores == {miner: 0.625, other_miner: 1.0}
        assert counted == (3, 1.125)

    def test_a_miner_no_longer_registered_is_not_added_or_counted(self):
        scores = {(1, "5MinerHotkey"): 0.0}

        counted = _add_credits(scores, [((9, "5Gone"), 1.0)])

        assert scores == {(1, "5MinerHotkey"): 0.0}
        assert counted == (0, 0.0)


class TestCredit:
    @pytest.mark.parametrize(
        "points, statements, credit",
        [
            (1, 1, 1.0),
            (1, 2, 0.7071),
            (2, 2, 1.4142),
            (1, 5, 0.4472),
            (5, 5, 2.2361),
            (0.88, 1, 0.88),
            (1.25, 4, 0.625),
            (-0.3, 1, -0.3),
            (-0.3, 2, -0.2121),
            (0, 3, 0.0),
        ],
    )
    def test_points_over_the_square_root_of_the_statements(
        self, points: float, statements: int, credit: float
    ):
        assert _credit(points_of(MINER, points, statements)) == pytest.approx(credit, abs=1e-4)

    @pytest.mark.parametrize(
        "exponent, credit",
        [(0, 0.2), (0.5, 0.4472), (1, 1.0)],
    )
    def test_the_bonus_exponent_sets_how_much_more_statements_pay(
        self, monkeypatch: pytest.MonkeyPatch, exponent: float, credit: float
    ):
        monkeypatch.setattr(score_dkg_miners, "STATEMENT_BONUS_EXPONENT", exponent)

        assert _credit(points_of(MINER, 1, 5)) == pytest.approx(credit, abs=1e-4)


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
        dkg_operations.get_forecast_points.return_value = []
        dkg_operations.get_distribution_statements.return_value = []
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
        dkg_operations.get_forecast_points.return_value = [
            points_of(MINER, 1, 1),
            points_of(MINER, 1, 5),
            points_of(OTHER_MINER, 2, 2),
        ]

        await task.run()

        dkg_operations.get_forecast_points.assert_awaited_once_with(
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
            extra={
                "moment": MOMENT.isoformat(),
                "miners": 3,
                "miners_with_a_score": 2,
                "probability_forecasts": 3,
                "probability_credits": pytest.approx(2.8614, abs=1e-4),
                "distribution_forecasts": 0,
                "distribution_credits": 0.0,
            },
        )

    @freeze_time(NOW)
    async def test_distribution_forecasts_add_to_the_same_score(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, logger: MagicMock
    ):
        dkg_operations.get_forecast_points.return_value = [
            points_of(MINER, 1, 1),
            points_of(OTHER_MINER, 0.88, 1),
        ]
        dkg_operations.get_distribution_statements.return_value = [
            statement_of(MINER, "question-1", UNLIKELY),
            statement_of(IDLE_MINER, "question-1", UNLIKELY),
            statement_of(OTHER_MINER, "question-1", EARLY, APPEARED_ON_DAY_TWO),
            statement_of(DkgMinerModel(miner_uid=9, miner_hotkey="5Gone"), "question-1", UNLIKELY),
        ]

        await task.run()

        dkg_operations.get_distribution_statements.assert_awaited_once_with(
            deadline_after=MOMENT - DISTRIBUTION_SCORE_WINDOW, moment=MOMENT
        )
        assert DISTRIBUTION_SCORE_WINDOW == timedelta(days=60)
        assert self.saved_scores(dkg_operations) == {
            (MINER.miner_uid, MINER.miner_hotkey): pytest.approx(1 + UNLIKELY_POINTS),
            (OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey): pytest.approx(
                0.88
                + (
                    HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO
                    - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO
                )
                / 30
            ),
            (IDLE_MINER.miner_uid, IDLE_MINER.miner_hotkey): 0.0,
        }
        early_points = (
            HOUSE_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO - EARLY_ERRORS_WHEN_IT_APPEARS_ON_DAY_TWO
        ) / 30
        logger.info.assert_called_once_with(
            "DKG miners scored",
            extra={
                "moment": MOMENT.isoformat(),
                "miners": 3,
                "miners_with_a_score": 2,
                "probability_forecasts": 2,
                "probability_credits": pytest.approx(1.88),
                "distribution_forecasts": 3,
                "distribution_credits": pytest.approx(2 * UNLIKELY_POINTS + early_points),
            },
        )

    @freeze_time(NOW)
    async def test_every_score_carries_the_version_that_scored_it(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(score_dkg_miners, "spec_version", 4021)
        dkg_operations.get_forecast_points.return_value = [points_of(MINER, 1, 1)]

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
        dkg_operations.get_forecast_points.assert_not_awaited()
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

        dkg_operations.get_forecast_points.assert_not_awaited()
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
    async def test_points_of_a_miner_no_longer_registered_are_ignored(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        deregistered = DkgMinerModel(miner_uid=MINER.miner_uid, miner_hotkey="5Deregistered")
        dkg_operations.get_forecast_points.return_value = [points_of(deregistered, 1, 1)]

        await task.run()

        scores = self.saved_scores(dkg_operations)
        assert (deregistered.miner_uid, deregistered.miner_hotkey) not in scores
        assert set(scores.values()) == {0.0}

    async def test_a_miner_below_zero_is_scored_zero(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        dkg_operations.get_forecast_points.return_value = [
            points_of(MINER, -0.3, 1),
            points_of(OTHER_MINER, 0.88, 1),
        ]

        await task.score(MOMENT)

        saved = self.saved_scores(dkg_operations)
        assert saved[(MINER.miner_uid, MINER.miner_hotkey)] == 0.0
        assert saved[(OTHER_MINER.miner_uid, OTHER_MINER.miner_hotkey)] == pytest.approx(0.88)

    async def test_a_negative_forecast_lowers_the_others_before_the_floor(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock
    ):
        dkg_operations.get_forecast_points.return_value = [
            points_of(MINER, 0.88, 1),
            points_of(MINER, -0.6, 1),
        ]

        await task.score(MOMENT)

        assert self.saved_scores(dkg_operations)[
            (MINER.miner_uid, MINER.miner_hotkey)
        ] == pytest.approx(0.28)

    @freeze_time(NOW)
    async def test_no_current_miners_saves_an_empty_run(
        self, task: ScoreDkgMiners, dkg_operations: AsyncMock, logger: MagicMock
    ):
        dkg_operations.get_current_miners.return_value = []
        dkg_operations.get_forecast_points.return_value = [points_of(MINER, 1, 1)]

        await task.run()

        dkg_operations.save_scores.assert_awaited_once_with([])
        logger.info.assert_called_once_with(
            "DKG miners scored",
            extra={
                "moment": MOMENT.isoformat(),
                "miners": 0,
                "miners_with_a_score": 0,
                "probability_forecasts": 0,
                "probability_credits": 0.0,
                "distribution_forecasts": 0,
                "distribution_credits": 0.0,
            },
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
                    distributions=None,
                    baselines=None,
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
                    baseline=None,
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
