from datetime import datetime, timedelta, timezone
from math import sqrt

import pytest

from neurons.validator.utils.common.distribution_scoring import (
    HORIZON_DAYS,
    NOT_WITHIN_HORIZON,
    daily_squared_errors,
    days_passed,
    distribution_score,
    distribution_score_so_far,
    outcome_day,
    points_while_not_appeared,
    question_credit,
    share_of_an_event,
    statement_points,
    statement_points_so_far,
)

DEADLINE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def day_one_or_never(chance_of_day_one: float) -> list[float]:
    return [chance_of_day_one] + [0.0] * (HORIZON_DAYS - 1) + [1 - chance_of_day_one]


def all_on(day: int) -> list[float]:
    return [1.0 if slot == day else 0.0 for slot in range(1, NOT_WITHIN_HORIZON + 1)]


HOUSE = day_one_or_never(0.2)
SPREAD_OVER_FIVE_DAYS = [0.1] * 5 + [0.0] * 25 + [0.5]
UNIFORM = [1 / NOT_WITHIN_HORIZON] * NOT_WITHIN_HORIZON


class TestOutcomeDay:
    @pytest.mark.parametrize(
        "after_the_deadline, day",
        [
            (timedelta(seconds=1), 1),
            (timedelta(hours=23, minutes=59), 1),
            (timedelta(days=1), 1),
            (timedelta(days=1, seconds=1), 2),
            (timedelta(days=1, hours=5), 2),
            (timedelta(days=6, hours=22), 7),
            (timedelta(days=29, seconds=1), 30),
            (timedelta(days=30), 30),
        ],
    )
    def test_the_days_from_the_deadline_rounded_up(self, after_the_deadline: timedelta, day: int):
        assert outcome_day(DEADLINE, DEADLINE + after_the_deadline) == day

    @pytest.mark.parametrize("before_the_deadline", [timedelta(0), timedelta(hours=3)])
    def test_at_least_day_one(self, before_the_deadline: timedelta):
        assert outcome_day(DEADLINE, DEADLINE - before_the_deadline) == 1

    def test_not_appeared_is_not_within_the_horizon(self):
        assert outcome_day(DEADLINE, None) == NOT_WITHIN_HORIZON

    @pytest.mark.parametrize("after_the_deadline", [timedelta(days=30, seconds=1), timedelta(90)])
    def test_appearing_after_the_horizon_is_not_within_it(self, after_the_deadline: timedelta):
        assert outcome_day(DEADLINE, DEADLINE + after_the_deadline) == NOT_WITHIN_HORIZON


class TestDaysPassed:
    @pytest.mark.parametrize(
        "after_the_deadline, days",
        [
            (timedelta(hours=15), 0),
            (timedelta(hours=23, minutes=59, seconds=59), 0),
            (timedelta(days=1), 1),
            (timedelta(days=1, hours=15), 1),
            (timedelta(days=5, hours=15), 5),
            (timedelta(days=30), 30),
            (timedelta(days=30, hours=15), 30),
            (timedelta(days=59), 30),
        ],
    )
    def test_whole_days_since_the_deadline_up_to_the_horizon(
        self, after_the_deadline: timedelta, days: int
    ):
        assert days_passed(DEADLINE, DEADLINE + after_the_deadline) == days

    def test_none_before_the_deadline(self):
        assert days_passed(DEADLINE, DEADLINE - timedelta(days=2)) == 0


class TestDistributionScoreSoFar:
    def test_nothing_before_the_first_day_has_passed(self):
        assert distribution_score_so_far(SPREAD_OVER_FIVE_DAYS, NOT_WITHIN_HORIZON, 0) == 0.0

    @pytest.mark.parametrize(
        "outcome, days, squared_errors",
        [
            (NOT_WITHIN_HORIZON, 1, 0.01),
            (NOT_WITHIN_HORIZON, 3, 0.01 + 0.04 + 0.09),
            (2, 1, 0.01),
            (2, 3, 0.01 + 0.64 + 0.49),
            (2, 5, 0.01 + 0.64 + 0.49 + 0.36 + 0.25),
        ],
    )
    def test_only_the_days_passed_count_always_over_the_horizon(
        self, outcome: int, days: int, squared_errors: float
    ):
        score = distribution_score_so_far(SPREAD_OVER_FIVE_DAYS, outcome, days)

        assert score == pytest.approx(-squared_errors / HORIZON_DAYS)

    @pytest.mark.parametrize("outcome", [1, 2, 30, NOT_WITHIN_HORIZON])
    def test_every_day_passed_is_the_whole_score(self, outcome: int):
        assert distribution_score_so_far(UNIFORM, outcome, HORIZON_DAYS) == distribution_score(
            UNIFORM, outcome
        )

    def test_never_rises_as_the_days_pass(self):
        scores = [distribution_score_so_far(UNIFORM, 12, days) for days in range(HORIZON_DAYS + 1)]

        assert scores == sorted(scores, reverse=True)

    def test_a_statement_not_appeared_yet_scores_the_same_days_as_one_that_appears_later(self):
        assert distribution_score_so_far(
            UNIFORM, NOT_WITHIN_HORIZON, 4
        ) == distribution_score_so_far(UNIFORM, 9, 4)


class TestDailySquaredErrors:
    def test_one_piece_per_day_of_the_horizon(self):
        assert len(daily_squared_errors(UNIFORM, 7)) == HORIZON_DAYS

    def test_before_the_outcome_day_the_cumulative_chance_is_the_error(self):
        errors = daily_squared_errors(SPREAD_OVER_FIVE_DAYS, 4)

        assert errors[:3] == pytest.approx([0.1**2, 0.2**2, 0.3**2])

    def test_from_the_outcome_day_on_the_missing_chance_is_the_error(self):
        errors = daily_squared_errors(SPREAD_OVER_FIVE_DAYS, 4)

        assert errors[3:5] == pytest.approx([0.6**2, 0.5**2])
        assert errors[5:] == pytest.approx([0.5**2] * 25)

    def test_never_appearing_leaves_the_cumulative_chance_as_the_error_every_day(self):
        errors = daily_squared_errors(SPREAD_OVER_FIVE_DAYS, NOT_WITHIN_HORIZON)

        assert errors == pytest.approx([0.01, 0.04, 0.09, 0.16] + [0.25] * 26)

    def test_the_first_days_do_not_depend_on_what_happens_later(self):
        appears_on_day_ten = daily_squared_errors(SPREAD_OVER_FIVE_DAYS, 10)
        never_appears = daily_squared_errors(SPREAD_OVER_FIVE_DAYS, NOT_WITHIN_HORIZON)

        assert appears_on_day_ten[:9] == never_appears[:9]

    @pytest.mark.parametrize("length", [0, 1, 30, 32])
    def test_rejects_a_vector_that_is_not_31_numbers(self, length: int):
        with pytest.raises(ValueError, match="Expected 31 probabilities"):
            daily_squared_errors([1 / max(length, 1)] * length, 1)

    @pytest.mark.parametrize("outcome_day", [-1, 0, 32])
    def test_rejects_an_outcome_day_outside_1_to_31(self, outcome_day: int):
        with pytest.raises(ValueError, match="outcome_day must be between 1 and 31"):
            daily_squared_errors(UNIFORM, outcome_day)


class TestDistributionScore:
    @pytest.mark.parametrize("outcome_day", [1, 2, 15, 30, NOT_WITHIN_HORIZON])
    def test_a_perfect_forecast_scores_zero(self, outcome_day: int):
        assert distribution_score(all_on(outcome_day), outcome_day) == 0.0

    @pytest.mark.parametrize(
        "forecast_day, outcome_day",
        [(1, NOT_WITHIN_HORIZON), (NOT_WITHIN_HORIZON, 1)],
    )
    def test_the_worst_forecast_scores_minus_one(self, forecast_day: int, outcome_day: int):
        assert distribution_score(all_on(forecast_day), outcome_day) == -1.0

    @pytest.mark.parametrize(
        "probabilities, outcome_day, score",
        [
            (day_one_or_never(0.8), 1, -0.04),
            (day_one_or_never(0.2), 1, -0.64),
            (day_one_or_never(0.4), NOT_WITHIN_HORIZON, -0.16),
            (day_one_or_never(0.2), NOT_WITHIN_HORIZON, -0.04),
            (SPREAD_OVER_FIVE_DAYS, 1, -8.8 / 30),
            (SPREAD_OVER_FIVE_DAYS, 2, -8.0 / 30),
            (SPREAD_OVER_FIVE_DAYS, 30, -6.8 / 30),
            (SPREAD_OVER_FIVE_DAYS, NOT_WITHIN_HORIZON, -6.8 / 30),
            (all_on(30), 1, -29 / 30),
            (all_on(1), 30, -29 / 30),
            (all_on(30), NOT_WITHIN_HORIZON, -1 / 30),
        ],
    )
    def test_minus_the_mean_squared_error_of_the_cumulative_chance(
        self, probabilities: list[float], outcome_day: int, score: float
    ):
        assert distribution_score(probabilities, outcome_day) == pytest.approx(score)

    @pytest.mark.parametrize("outcome_day", range(1, NOT_WITHIN_HORIZON + 1))
    def test_stays_between_minus_one_and_zero(self, outcome_day: int):
        for probabilities in (UNIFORM, SPREAD_OVER_FIVE_DAYS, HOUSE, all_on(17)):
            assert -1.0 <= distribution_score(probabilities, outcome_day) <= 0.0

    def test_a_forecast_closer_to_the_outcome_day_scores_higher(self):
        scores = [distribution_score(all_on(day), 10) for day in (10, 11, 15, 30)]

        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_the_never_slot_only_counts_through_what_it_takes_from_the_days(self):
        without_never_slot = SPREAD_OVER_FIVE_DAYS[:HORIZON_DAYS]

        assert distribution_score(without_never_slot + [0.5], 3) == distribution_score(
            without_never_slot + [0.0], 3
        )


class TestStatementPoints:
    def test_worked_example_statement_that_appears_on_day_one(self):
        assert statement_points(day_one_or_never(0.8), HOUSE, 1) == pytest.approx(0.60)

    def test_worked_example_statement_that_never_appears(self):
        points = statement_points(day_one_or_never(0.4), HOUSE, NOT_WITHIN_HORIZON)

        assert points == pytest.approx(-0.12)

    @pytest.mark.parametrize("outcome_day", [1, 2, 30, NOT_WITHIN_HORIZON])
    def test_a_miner_identical_to_the_house_earns_exactly_zero(self, outcome_day: int):
        for probabilities in (HOUSE, UNIFORM, SPREAD_OVER_FIVE_DAYS):
            assert statement_points(probabilities, list(probabilities), outcome_day) == 0.0

    def test_beating_the_house_pays_and_losing_to_it_costs_the_same(self):
        better, worse = day_one_or_never(0.9), day_one_or_never(0.1)

        assert statement_points(better, worse, 1) > 0
        assert statement_points(worse, better, 1) == -statement_points(better, worse, 1)

    def test_a_perfect_miner_earns_what_the_house_lost(self):
        assert statement_points(all_on(30), HOUSE, 30) == -distribution_score(HOUSE, 30)

    def test_points_stay_between_minus_one_and_one(self):
        assert statement_points(all_on(1), all_on(NOT_WITHIN_HORIZON), 1) == 1.0
        assert statement_points(all_on(NOT_WITHIN_HORIZON), all_on(1), 1) == -1.0


class TestStatementPointsSoFar:
    def test_nothing_before_the_first_day_has_passed(self):
        assert statement_points_so_far(SPREAD_OVER_FIVE_DAYS, HOUSE, 2, 0) == 0.0

    @pytest.mark.parametrize("outcome", [1, 2, 30, NOT_WITHIN_HORIZON])
    @pytest.mark.parametrize("days", [1, 5, 12, HORIZON_DAYS])
    def test_the_miner_and_the_house_are_scored_on_the_same_days(self, outcome: int, days: int):
        points = statement_points_so_far(SPREAD_OVER_FIVE_DAYS, HOUSE, outcome, days)

        assert points == distribution_score_so_far(
            SPREAD_OVER_FIVE_DAYS, outcome, days
        ) - distribution_score_so_far(HOUSE, outcome, days)

    @pytest.mark.parametrize("outcome", [1, 2, 30, NOT_WITHIN_HORIZON])
    @pytest.mark.parametrize("days", [0, 1, 5, HORIZON_DAYS])
    def test_a_miner_identical_to_the_house_earns_exactly_zero_on_any_day(
        self, outcome: int, days: int
    ):
        for probabilities in (HOUSE, UNIFORM, SPREAD_OVER_FIVE_DAYS):
            assert statement_points_so_far(probabilities, list(probabilities), outcome, days) == 0.0

    def test_while_nothing_appears_saying_less_than_the_house_pays(self):
        cautious, bold = day_one_or_never(0.05), day_one_or_never(0.6)

        assert statement_points_so_far(cautious, HOUSE, NOT_WITHIN_HORIZON, 3) == pytest.approx(
            3 * (0.2**2 - 0.05**2) / HORIZON_DAYS
        )
        assert statement_points_so_far(bold, HOUSE, NOT_WITHIN_HORIZON, 3) == pytest.approx(
            3 * (0.2**2 - 0.6**2) / HORIZON_DAYS
        )

    def test_every_day_passed_is_the_whole_statement_points(self):
        assert statement_points_so_far(
            day_one_or_never(0.8), HOUSE, 1, HORIZON_DAYS
        ) == statement_points(day_one_or_never(0.8), HOUSE, 1)


class TestPointsWhileNotAppeared:
    @pytest.mark.parametrize("points", [-1.0, -0.12, -1e-9])
    def test_points_below_zero_are_kept(self, points: float):
        assert points_while_not_appeared(points) == points

    @pytest.mark.parametrize("points", [0.0, 1e-9, 0.0027, 0.8, 1.0])
    def test_points_above_zero_are_not_earned(self, points: float):
        assert points_while_not_appeared(points) == 0.0

    def test_saying_never_against_the_house_earns_nothing_while_nothing_appears(self):
        never = [0.0] * HORIZON_DAYS + [1.0]
        so_far = statement_points_so_far(never, HOUSE, NOT_WITHIN_HORIZON, HORIZON_DAYS)

        assert so_far == pytest.approx(0.2**2)
        assert points_while_not_appeared(so_far) == 0.0


class TestShareOfAnEvent:
    @pytest.mark.parametrize(
        "miners, share", [(1, 1.0), (2, 0.5), (4, 0.25), (10, 0.1), (37, 1 / 37)]
    )
    def test_an_event_is_split_equally_among_the_miners_paid_for_it(
        self, miners: int, share: float
    ):
        assert share_of_an_event(miners) == share

    @pytest.mark.parametrize("miners", [1, 2, 3, 7, 50])
    def test_the_shares_of_an_event_add_up_to_one(self, miners: int):
        assert miners * share_of_an_event(miners) == pytest.approx(1.0)


class TestQuestionCredit:
    def test_an_empty_bundle_is_zero(self):
        assert question_credit([]) == 0.0

    def test_one_statement_keeps_its_points(self):
        assert question_credit([0.6]) == 0.6

    def test_worked_example_house_adjusted_credit(self):
        points = [
            statement_points(day_one_or_never(0.8), HOUSE, 1),
            statement_points(day_one_or_never(0.4), HOUSE, NOT_WITHIN_HORIZON),
        ]

        assert question_credit(points) == pytest.approx(0.339, abs=5e-4)
        assert question_credit(points) == pytest.approx(0.48 / sqrt(2))

    def test_worked_example_raw_credit(self):
        raw_scores = [
            distribution_score(day_one_or_never(0.8), 1),
            distribution_score(day_one_or_never(0.4), NOT_WITHIN_HORIZON),
        ]

        assert question_credit(raw_scores) == pytest.approx(-0.141, abs=5e-4)

    @pytest.mark.parametrize(
        "points, credit",
        [
            ([1.0, 1.0], 1.4142),
            ([0.5, 0.5, 0.5, 0.5], 1.0),
            ([0.6, -0.12, 0.0], 0.2771),
            ([-0.3, -0.3], -0.4243),
            ([0.0, 0.0, 0.0], 0.0),
        ],
    )
    def test_the_sum_of_the_points_over_the_square_root_of_the_statements(
        self, points: list[float], credit: float
    ):
        assert question_credit(points) == pytest.approx(credit, abs=1e-4)

    def test_a_statement_with_zero_points_still_counts_as_a_statement(self):
        assert question_credit([0.6, 0.0]) == pytest.approx(0.6 / sqrt(2))
