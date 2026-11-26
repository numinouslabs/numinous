from collections import defaultdict
from collections.abc import Set
from datetime import datetime, time, timedelta, timezone
from typing import NamedTuple

from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgDistributionStatementModel,
    DkgForecastPointsModel,
    DkgScoreModel,
)
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.distribution_scoring import (
    HORIZON_DAYS,
    NOT_WITHIN_HORIZON,
    days_passed,
    outcome_day,
    points_while_not_appeared,
    question_credit,
    share_of_an_event,
    statement_points,
    statement_points_so_far,
)
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version

DAILY_SCORING_TIME = time(15, 0, tzinfo=timezone.utc)
SCORE_WINDOW = timedelta(days=30)
DISTRIBUTION_SCORE_WINDOW = timedelta(days=HORIZON_DAYS) + SCORE_WINDOW
STATEMENT_BONUS_EXPONENT = 0.5

Miner = tuple[int, str]


class _CountedCredits(NamedTuple):
    forecasts: int
    credits: float


class _ScoredStatement(NamedTuple):
    statement: DkgDistributionStatementModel
    points: float


def _credit(forecast: DkgForecastPointsModel) -> float:
    share = forecast.points / forecast.statements
    bonus = forecast.statements**STATEMENT_BONUS_EXPONENT
    return share * bonus


def _distribution_points(statement: DkgDistributionStatementModel, moment: datetime) -> float:
    if statement.baseline is None:
        return 0.0

    day_it_appeared = outcome_day(statement.deadline_at, statement.appeared_at)

    if day_it_appeared == NOT_WITHIN_HORIZON:
        return points_while_not_appeared(
            statement_points_so_far(
                statement.distribution,
                statement.baseline,
                day_it_appeared,
                days_passed(statement.deadline_at, moment),
            )
        )

    return statement_points(statement.distribution, statement.baseline, day_it_appeared)


def _is_paid_for_an_event(scored: _ScoredStatement) -> bool:
    return scored.statement.event_id is not None and scored.points > 0


def _miners_paid_per_event(scored_statements: list[_ScoredStatement]) -> dict[str, set[Miner]]:
    miners_per_event: dict[str, set[Miner]] = defaultdict(set)
    for scored in scored_statements:
        if _is_paid_for_an_event(scored):
            statement = scored.statement
            miners_per_event[statement.event_id].add((statement.miner_uid, statement.miner_hotkey))

    return miners_per_event


def _shared_points(scored: _ScoredStatement, miners_per_event: dict[str, set[Miner]]) -> float:
    if not _is_paid_for_an_event(scored):
        return scored.points

    return scored.points * share_of_an_event(len(miners_per_event[scored.statement.event_id]))


def _distribution_credits(
    statements: list[DkgDistributionStatementModel], moment: datetime, registered_miners: Set[Miner]
) -> list[tuple[Miner, float]]:
    scored_statements = [
        _ScoredStatement(statement, _distribution_points(statement, moment))
        for statement in statements
        if (statement.miner_uid, statement.miner_hotkey) in registered_miners
    ]
    miners_per_event = _miners_paid_per_event(scored_statements)

    points_per_forecast: dict[tuple[Miner, str], list[float]] = defaultdict(list)
    for scored in scored_statements:
        statement = scored.statement
        miner = (statement.miner_uid, statement.miner_hotkey)
        points_per_forecast[(miner, statement.question_id)].append(
            _shared_points(scored, miners_per_event)
        )

    return [(miner, question_credit(points)) for (miner, _), points in points_per_forecast.items()]


def _add_credits(
    scores: dict[Miner, float], forecast_credits: list[tuple[Miner, float]]
) -> _CountedCredits:
    counted = [(miner, credit) for miner, credit in forecast_credits if miner in scores]
    for miner, credit in counted:
        scores[miner] += credit

    return _CountedCredits(
        forecasts=len(counted), credits=sum((credit for _, credit in counted), 0.0)
    )


def _last_scoring_moment(now: datetime) -> datetime:
    today = datetime.combine(now.date(), DAILY_SCORING_TIME)
    return today if now >= today else today - timedelta(days=1)


class ScoreDkgMiners(AbstractTask):
    interval: float
    dkg_operations: DkgDatabaseOperations
    logger: NuminousLogger

    def __init__(
        self, interval_seconds: float, dkg_operations: DkgDatabaseOperations, logger: NuminousLogger
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(dkg_operations, DkgDatabaseOperations):
            raise TypeError("dkg_operations must be an instance of DkgDatabaseOperations.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.dkg_operations = dkg_operations
        self.logger = logger

    @property
    def name(self):
        return "score-dkg-miners"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        moment = _last_scoring_moment(datetime.now(timezone.utc))

        if await self.dkg_operations.get_last_scored_at() == moment:
            return

        last_results_pull = await self.dkg_operations.get_last_results_pull()
        if last_results_pull is None or last_results_pull <= moment:
            self.logger.debug(
                "DKG scoring waits for a results pull after its moment",
                extra={"moment": moment.isoformat()},
            )
            return

        await self.score(moment)

    async def score(self, moment: datetime):
        scores = {
            (miner.miner_uid, miner.miner_hotkey): 0.0
            for miner in await self.dkg_operations.get_current_miners()
        }

        forecast_points = await self.dkg_operations.get_forecast_points(
            judged_after=moment - SCORE_WINDOW, judged_before=moment
        )
        probability = _add_credits(
            scores,
            [
                ((forecast.miner_uid, forecast.miner_hotkey), _credit(forecast))
                for forecast in forecast_points
            ],
        )

        distribution_statements = await self.dkg_operations.get_distribution_statements(
            deadline_after=moment - DISTRIBUTION_SCORE_WINDOW, moment=moment
        )
        distribution = _add_credits(
            scores, _distribution_credits(distribution_statements, moment, scores.keys())
        )

        await self.dkg_operations.save_scores(
            [
                DkgScoreModel(
                    miner_uid=uid,
                    miner_hotkey=hotkey,
                    scored_at=moment,
                    score=max(score, 0.0),
                    spec_version=spec_version,
                )
                for (uid, hotkey), score in scores.items()
            ]
        )

        self.logger.info(
            "DKG miners scored",
            extra={
                "moment": moment.isoformat(),
                "miners": len(scores),
                "miners_with_a_score": sum(1 for score in scores.values() if score > 0),
                "probability_forecasts": probability.forecasts,
                "probability_credits": probability.credits,
                "distribution_forecasts": distribution.forecasts,
                "distribution_credits": distribution.credits,
            },
        )
