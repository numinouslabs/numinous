from datetime import datetime, timezone

from bittensor import AsyncSubtensor

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class SyncMinersMetadata(AbstractTask):
    """Sync miners' metadata from metagraph to database."""

    interval: float
    db_operations: DatabaseOperations
    subtensor: AsyncSubtensor
    netuid: int
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        netuid: int,
        subtensor: AsyncSubtensor,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive float")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(netuid, int) or netuid < 0:
            raise ValueError("netuid must be a non-negative integer.")

        if not isinstance(subtensor, AsyncSubtensor):
            raise TypeError("subtensor must be an instance of AsyncSubtensor.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.netuid = netuid
        self.subtensor = subtensor
        self.logger = logger

    @property
    def name(self) -> str:
        return "sync-miners-metadata"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        async with self.subtensor as subtensor:
            metagraph = await subtensor.metagraph(netuid=self.netuid, lite=True)

        block = metagraph.block.item()
        miners_count = await self.db_operations.get_miners_count()

        registered_date = (
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            if miners_count > 0
            else datetime(year=2024, month=1, day=1).isoformat()
        )

        miners = []
        for uid in metagraph.uids:
            int_uid = int(uid)
            axon = metagraph.axons[int_uid]

            if axon is None:
                continue

            trust_value = metagraph.validator_trust[int_uid]
            is_validating = bool(float(trust_value) > 0.0)
            validator_permit = bool(int(metagraph.validator_permit[int_uid]) > 0)

            miners.append(
                (
                    int_uid,
                    axon.hotkey,
                    axon.ip,
                    registered_date,
                    block,
                    is_validating,
                    validator_permit,
                    axon.ip,
                    block,
                )
            )

        if miners:
            await self.db_operations.upsert_miners(miners=miners)

            self.logger.debug(
                "Miners metadata synced",
                extra={"miners_count": len(miners), "block": block},
            )
