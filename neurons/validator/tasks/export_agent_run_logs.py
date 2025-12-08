from uuid import UUID

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_run_logs import AgentRunLogsModel
from neurons.validator.models.numinous_client import PostAgentLogsRequestBody
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class ExportAgentRunLogs(AbstractTask):
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
        return "export-agent-run-logs"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def export_log_to_backend(self, log: AgentRunLogsModel) -> None:
        payload = PostAgentLogsRequestBody(
            run_id=UUID(log.run_id),
            log_content=log.log_content,
        )

        await self.api_client.post_agent_logs(body=payload)

    async def run(self) -> None:
        unexported_logs = await self.db_operations.get_unexported_agent_run_logs(
            limit=self.batch_size
        )

        if not unexported_logs:
            self.logger.debug("No unexported logs to export")
        else:
            self.logger.debug(
                "Found unexported logs to export",
                extra={"n_logs": len(unexported_logs)},
            )

            successfully_exported_run_ids = []

            for log in unexported_logs:
                try:
                    await self.export_log_to_backend(log)
                    successfully_exported_run_ids.append(log.run_id)
                except Exception:
                    self.errors_count += 1
                    self.logger.warning(
                        "Failed to export log to backend",
                        extra={"run_id": log.run_id},
                        exc_info=True,
                    )

            if successfully_exported_run_ids:
                await self.db_operations.mark_agent_run_logs_as_exported(
                    run_ids=successfully_exported_run_ids
                )

                self.logger.debug(
                    "Marked logs as exported",
                    extra={"n_logs": len(successfully_exported_run_ids)},
                )

        self.logger.debug(
            "Export logs task completed",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
