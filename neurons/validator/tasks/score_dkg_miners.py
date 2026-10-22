from datetime import datetime, time, timedelta, timezone

from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import DkgForecastRightsModel, DkgScoreModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version

DAILY_SCORING_TIME = time(15, 0, tzinfo=timezone.utc)
SCORE_WINDOW = timedelta(days=30)
STATEMENT_BONUS_EXPONENT = 0.5


def _credit(forecast: DkgForecastRightsModel) -> float:
    share_right = forecast.rights / forecast.statements
    bonus = forecast.statements**STATEMENT_BONUS_EXPONENT
    return share_right * bonus


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

        for forecast in await self.dkg_operations.get_forecast_rights(
            judged_after=moment - SCORE_WINDOW, judged_before=moment
        ):
            miner = (forecast.miner_uid, forecast.miner_hotkey)
            if miner in scores:
                scores[miner] += _credit(forecast)

        await self.dkg_operations.save_scores(
            [
                DkgScoreModel(
                    miner_uid=uid,
                    miner_hotkey=hotkey,
                    scored_at=moment,
                    score=score,
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
            },
        )
