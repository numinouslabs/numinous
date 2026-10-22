from datetime import datetime
from itertools import groupby

from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import DkgScoreModel
from neurons.validator.models.dkg_client import MinerScore, ScoringRun
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


def _scoring_run(scored_at: datetime, run_scores: list[DkgScoreModel]) -> ScoringRun:
    return ScoringRun(
        scored_at=scored_at,
        spec_version=run_scores[0].spec_version,
        scores=[
            MinerScore(
                miner_uid=score.miner_uid, miner_hotkey=score.miner_hotkey, score=score.score
            )
            for score in run_scores
        ],
    )


def _scoring_runs(scores: list[DkgScoreModel]) -> list[ScoringRun]:
    return [
        _scoring_run(scored_at, list(run_scores))
        for scored_at, run_scores in groupby(scores, key=lambda score: score.scored_at)
    ]


class PostDkgScores(AbstractTask):
    interval: float
    dkg_operations: DkgDatabaseOperations
    dkg_client: DkgClient
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        dkg_operations: DkgDatabaseOperations,
        dkg_client: DkgClient,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(dkg_operations, DkgDatabaseOperations):
            raise TypeError("dkg_operations must be an instance of DkgDatabaseOperations.")

        if not isinstance(dkg_client, DkgClient):
            raise TypeError("dkg_client must be an instance of DkgClient.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.dkg_operations = dkg_operations
        self.dkg_client = dkg_client
        self.logger = logger

    @property
    def name(self):
        return "post-dkg-scores"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        for scoring_run in _scoring_runs(await self.dkg_operations.get_unposted_scores()):
            stored = await self.dkg_client.post_scores(scoring_run)
            await self.dkg_operations.mark_scores_posted(scoring_run.scored_at)

            self.logger.info(
                "DKG scores posted",
                extra={
                    "scored_at": scoring_run.scored_at.isoformat(),
                    "spec_version": scoring_run.spec_version,
                    "miners": len(scoring_run.scores),
                    "stored": stored.stored,
                },
            )
