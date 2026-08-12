from datetime import datetime, timezone

from bittensor import Metagraph, Subtensor

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger

UNSERVED_AXON_IP = "0.0.0.0"


def axon_ip(served_axon: str | None) -> str:
    if not served_axon:
        return UNSERVED_AXON_IP

    return served_axon.rsplit(":", 1)[0].strip("[]")


class SyncMinersMetadata(AbstractTask):
    """Sync miners' metadata from metagraph to database."""

    interval: float
    db_operations: DatabaseOperations
    network: str
    netuid: int
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        netuid: int,
        network: str,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive float")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(netuid, int) or netuid < 0:
            raise ValueError("netuid must be a non-negative integer.")

        if not isinstance(network, str) or not network:
            raise ValueError("network must be a non-empty string.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.netuid = netuid
        self.network = network
        self.logger = logger

    @property
    def name(self) -> str:
        return "sync-miners-metadata"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        async with Subtensor(self.network) as chain_client:
            metagraph: Metagraph = await chain_client.subnets.metagraph(self.netuid)

        block = metagraph.block
        miners_count = await self.db_operations.get_miners_count()

        registered_date = (
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            if miners_count > 0
            else datetime(year=2024, month=1, day=1).isoformat()
        )

        miners = []
        for neuron in metagraph.neurons:
            node_ip = axon_ip(neuron.axon)

            miners.append(
                (
                    neuron.uid,
                    neuron.hotkey,
                    node_ip,
                    registered_date,
                    block,
                    node_ip,
                    block,
                )
            )

        if miners:
            await self.db_operations.upsert_miners(miners=miners)

            self.logger.debug(
                "Miners metadata synced",
                extra={"miners_count": len(miners), "block": block},
            )
