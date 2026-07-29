from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.numinous_client import MemoryCommitRequestBody, MemoryEntry
from neurons.validator.models.reforecast_memory import ReforecastMemoryForExport
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.interval import get_interval_iso_datetime
from neurons.validator.utils.logger.logger import NuminousLogger


class ExportMemory(AbstractTask):
    interval: float
    batch_size: int
    db_operations: DatabaseOperations
    api_client: NuminousClient
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        batch_size: int,
        db_operations: DatabaseOperations,
        api_client: NuminousClient,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.batch_size = batch_size
        self.db_operations = db_operations
        self.api_client = api_client

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self) -> str:
        return "export-memory"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    def prepare_payload(self, memories: list[ReforecastMemoryForExport]) -> MemoryCommitRequestBody:
        entries = [
            MemoryEntry(
                miner_uid=memory.miner_uid,
                miner_hotkey=memory.miner_hotkey,
                event_id=memory.event_id,
                interval_datetime=get_interval_iso_datetime(memory.interval_start_minutes),
                memory=memory.memory,
            )
            for memory in memories
        ]

        return MemoryCommitRequestBody(entries=entries)

    async def run(self) -> None:
        unexported = await self.db_operations.get_reforecast_memories_for_export(
            limit=self.batch_size
        )

        if not unexported:
            self.logger.debug("No unexported memories to export")
        else:
            self.logger.debug(
                "Found unexported memories to export",
                extra={"n_memories": len(unexported)},
            )

            payload = self.prepare_payload(memories=unexported)

            try:
                response = await self.api_client.commit_memory(body=payload)
            except Exception:
                self.errors_count += 1
                self.logger.exception("Failed to export memories to backend")
                return

            run_ids = [memory.run_id for memory in unexported]
            await self.db_operations.mark_reforecast_memories_as_exported(run_ids=run_ids)

            # inserted < len when another validator won the first-writer race for an
            # interval - expected (first-writer-wins), not an error. We still mark
            # them exported so we don't keep re-sending.
            self.logger.debug(
                "Exported memories",
                extra={"n_memories": len(run_ids), "inserted": response.inserted},
            )

        self.logger.debug(
            "Export memory task completed",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
