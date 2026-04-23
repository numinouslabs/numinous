from datetime import datetime, timezone
from uuid import UUID

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.numinous_client import MinerSourceSubmission, PostSourcesRequestBody
from neurons.validator.models.sources import SourcesForExport
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class ExportSources(AbstractTask):
    interval: float
    batch_size: int
    db_operations: DatabaseOperations
    api_client: NuminousClient
    logger: NuminousLogger
    validator_uid: int
    validator_hotkey: str

    def __init__(
        self,
        interval_seconds: float,
        batch_size: int,
        db_operations: DatabaseOperations,
        api_client: NuminousClient,
        logger: NuminousLogger,
        validator_uid: int,
        validator_hotkey: str,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.batch_size = batch_size
        self.db_operations = db_operations
        self.api_client = api_client
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self) -> str:
        return "export-sources"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    def prepare_payload(self, sources_entries: list[SourcesForExport]) -> PostSourcesRequestBody:
        submissions = []

        for entry in sources_entries:
            submitted_at = entry.created_at or datetime.now(timezone.utc)

            submission = MinerSourceSubmission(
                event_id=entry.event_id,
                miner_uid=entry.miner_uid,
                miner_hotkey=entry.miner_hotkey,
                track=entry.track,
                validator_uid=self.validator_uid,
                validator_hotkey=self.validator_hotkey,
                run_id=UUID(entry.run_id),
                submitted_at=submitted_at,
                sources=entry.sources,
            )
            submissions.append(submission)

        return PostSourcesRequestBody(submissions=submissions)

    async def run(self) -> None:
        unexported = await self.db_operations.get_sources_for_export(limit=self.batch_size)

        if not unexported:
            self.logger.debug("No unexported sources to export")
        else:
            self.logger.debug(
                "Found unexported sources to export",
                extra={"n_sources": len(unexported)},
            )

            payload = self.prepare_payload(sources_entries=unexported)

            try:
                await self.api_client.post_sources(body=payload)
            except Exception:
                self.errors_count += 1
                self.logger.exception("Failed to export sources to backend")
                return

            run_ids = [entry.run_id for entry in unexported]
            await self.db_operations.mark_sources_as_exported(run_ids=run_ids)

            self.logger.debug(
                "Exported sources",
                extra={"n_sources": len(run_ids)},
            )

        self.logger.debug(
            "Export sources task completed",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
