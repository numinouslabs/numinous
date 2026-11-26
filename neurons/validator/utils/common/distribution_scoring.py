from datetime import datetime, timedelta
from itertools import accumulate
from math import ceil, sqrt
from typing import Sequence

HORIZON_DAYS = 30
NOT_WITHIN_HORIZON = HORIZON_DAYS + 1
ONE_DAY = timedelta(days=1)


def outcome_day(deadline_at: datetime, appeared_at: datetime | None) -> int:
    if appeared_at is None:
        return NOT_WITHIN_HORIZON

    return min(max(ceil((appeared_at - deadline_at) / ONE_DAY), 1), NOT_WITHIN_HORIZON)


def days_passed(deadline_at: datetime, moment: datetime) -> int:
    return min(max((moment - deadline_at) // ONE_DAY, 0), HORIZON_DAYS)


def daily_squared_errors(probabilities: Sequence[float], outcome_day: int) -> list[float]:
    if len(probabilities) != NOT_WITHIN_HORIZON:
        raise ValueError(f"Expected {NOT_WITHIN_HORIZON} probabilities, got {len(probabilities)}.")

    if not 1 <= outcome_day <= NOT_WITHIN_HORIZON:
        raise ValueError(f"outcome_day must be between 1 and {NOT_WITHIN_HORIZON}.")

    chance_appeared_by_day = accumulate(probabilities[:HORIZON_DAYS])

    return [
        (chance - (1.0 if outcome_day <= day else 0.0)) ** 2
        for day, chance in enumerate(chance_appeared_by_day, start=1)
    ]


def distribution_score_so_far(
    probabilities: Sequence[float], outcome_day: int, days_passed: int
) -> float:
    passed_days_errors = daily_squared_errors(probabilities, outcome_day)[:days_passed]

    return 0.0 - sum(passed_days_errors) / HORIZON_DAYS


def distribution_score(probabilities: Sequence[float], outcome_day: int) -> float:
    return distribution_score_so_far(probabilities, outcome_day, HORIZON_DAYS)


def statement_points_so_far(
    miner_probabilities: Sequence[float],
    house_probabilities: Sequence[float],
    outcome_day: int,
    days_passed: int,
) -> float:
    return distribution_score_so_far(
        miner_probabilities, outcome_day, days_passed
    ) - distribution_score_so_far(house_probabilities, outcome_day, days_passed)


def statement_points(
    miner_probabilities: Sequence[float], house_probabilities: Sequence[float], outcome_day: int
) -> float:
    return statement_points_so_far(
        miner_probabilities, house_probabilities, outcome_day, HORIZON_DAYS
    )


def points_while_not_appeared(points_so_far: float) -> float:
    return min(points_so_far, 0.0)


def share_of_an_event(miners_paid_for_the_event: int) -> float:
    return 1 / miners_paid_for_the_event


def question_credit(points_per_statement: Sequence[float]) -> float:
    if not points_per_statement:
        return 0.0

    return sum(points_per_statement) / sqrt(len(points_per_statement))
