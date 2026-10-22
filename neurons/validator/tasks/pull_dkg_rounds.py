import json
from datetime import datetime, timezone

from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.models.dkg import DkgForecastHashModel, DkgQuestionModel, DkgRoundModel
from neurons.validator.models.dkg_client import DkgRoundResponse
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


def _round_model(closed_round: DkgRoundResponse) -> DkgRoundModel:
    return DkgRoundModel(
        round_id=closed_round.id,
        deadline_at=closed_round.deadline_at,
        horizon_end_at=closed_round.horizon_end_at,
    )


def _question_models(closed_round: DkgRoundResponse) -> list[DkgQuestionModel]:
    return [
        DkgQuestionModel(
            question_id=question.id,
            round_id=closed_round.id,
            entity_id=question.entity_id,
            entity=question.entity,
            counterparty_id=question.counterparty_id,
            counterparty=question.counterparty,
            relationship=(
                json.dumps(question.relationship.model_dump()) if question.relationship else None
            ),
        )
        for question in closed_round.questions
    ]


class PullDkgRounds(AbstractTask):
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
        return "pull-dkg-rounds"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        history_start = datetime.now(timezone.utc) - DKG_HISTORY
        last_deadline = await self.dkg_operations.get_last_complete_round_deadline()
        closed_since = max(last_deadline, history_start) if last_deadline else history_start
        offset = 0

        while True:
            page = await self.dkg_client.get_closed_rounds(closed_since, offset)

            for closed_round in page.rounds:
                await self.pull_round(closed_round)

            if page.next_offset is None:
                break

            offset = page.next_offset

    async def pull_round(self, closed_round: DkgRoundResponse):
        await self.dkg_operations.save_round(
            _round_model(closed_round), _question_models(closed_round)
        )
        hashes = 0
        offset = 0

        while True:
            page = await self.dkg_client.get_round_hashes(closed_round.id, offset)
            await self.dkg_operations.save_forecast_hashes(
                [DkgForecastHashModel(**forecast.model_dump()) for forecast in page.forecasts]
            )
            hashes += len(page.forecasts)

            if page.next_offset is None:
                break

            offset = page.next_offset

        await self.dkg_operations.mark_round_hashes_pulled(closed_round.id)

        self.logger.info(
            "DKG round pulled",
            extra={
                "round_id": closed_round.id,
                "deadline_at": closed_round.deadline_at.isoformat(),
                "questions": len(closed_round.questions),
                "hashes": hashes,
            },
        )
