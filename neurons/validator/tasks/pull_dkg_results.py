import hashlib
import json
from datetime import datetime, timedelta, timezone

from bittensor_wallet import Keypair

from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.models.dkg import DkgForecastModel, DkgStatementResultModel
from neurons.validator.models.dkg_client import ReleasedForecastResponse
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger

RESULTS_SINCE_MARGIN = timedelta(hours=1)
HASH_DOES_NOT_MATCH = "the sealed forecast does not match its hash"
SIGNATURE_DOES_NOT_VERIFY = "the miner signature does not verify"


def _failed_check(released: ReleasedForecastResponse, sealed_sha256: str) -> str | None:
    sealed = bytes.fromhex(released.sealed)
    if hashlib.sha256(sealed).hexdigest() != sealed_sha256:
        return HASH_DOES_NOT_MATCH

    signed = released.question_id.encode() + sealed
    if not Keypair(ss58_address=released.miner_hotkey).verify(
        signed, bytes.fromhex(released.signature)
    ):
        return SIGNATURE_DOES_NOT_VERIFY

    return None


def _forecast_key(released: ReleasedForecastResponse) -> tuple[int, str, str]:
    return released.miner_uid, released.miner_hotkey, released.question_id


def _forecast_model(released: ReleasedForecastResponse, sealed_sha256: str) -> DkgForecastModel:
    return DkgForecastModel(
        miner_uid=released.miner_uid,
        miner_hotkey=released.miner_hotkey,
        question_id=released.question_id,
        opened=json.dumps(released.opened) if released.opened else None,
        invalid_reason=released.invalid_reason,
        failed_check=_failed_check(released, sealed_sha256),
    )


def _result_models(released: ReleasedForecastResponse) -> list[DkgStatementResultModel]:
    return [
        DkgStatementResultModel(
            miner_uid=released.miner_uid,
            miner_hotkey=released.miner_hotkey,
            question_id=released.question_id,
            **result.model_dump(),
        )
        for result in released.results
    ]


class PullDkgResults(AbstractTask):
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
        return "pull-dkg-results"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        started_at = datetime.now(timezone.utc)
        history_start = started_at - DKG_HISTORY
        last_pull = await self.dkg_operations.get_last_results_pull()
        since = max(last_pull - RESULTS_SINCE_MARGIN, history_start) if last_pull else history_start
        stored, waiting_for_round = 0, 0
        offset = 0

        while True:
            page = await self.dkg_client.get_results(since, offset)
            in_history = [
                released for released in page.forecasts if released.deadline_at > history_start
            ]
            saved = await self.save_page(in_history)
            stored += saved
            waiting_for_round += len(in_history) - saved

            if page.next_offset is None:
                break

            offset = page.next_offset

        if waiting_for_round:
            self.logger.warning(
                "DKG results waiting for their round, pull not recorded",
                extra={"since": since.isoformat(), "waiting_for_round": waiting_for_round},
            )
            return

        await self.dkg_operations.record_results_pull(started_at)

        self.logger.info(
            "DKG results pulled", extra={"since": since.isoformat(), "forecasts": stored}
        )

    async def save_page(self, released_forecasts: list[ReleasedForecastResponse]) -> int:
        stored_hashes = {
            (stored.miner_uid, stored.miner_hotkey, stored.question_id): stored.sealed_sha256
            for stored in await self.dkg_operations.get_forecast_hashes(
                list({released.question_id for released in released_forecasts})
            )
        }
        with_hash = [
            released for released in released_forecasts if _forecast_key(released) in stored_hashes
        ]

        await self.dkg_operations.save_forecasts(
            [
                _forecast_model(released, stored_hashes[_forecast_key(released)])
                for released in with_hash
            ],
            [result for released in with_hash for result in _result_models(released)],
        )

        return len(with_hash)
