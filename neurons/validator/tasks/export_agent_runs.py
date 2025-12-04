from uuid import UUID

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel
from neurons.validator.models.numinous_client import AgentRunSubmission, PostAgentRunsRequestBody
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class ExportAgentRuns(AbstractTask):
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
        return "export-agent-runs"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    def prepare_runs_payload(self, db_runs: list[AgentRunsModel]) -> PostAgentRunsRequestBody:
        runs = []

        for db_run in db_runs:
            run = AgentRunSubmission(
                run_id=UUID(db_run.run_id),
                miner_uid=db_run.miner_uid,
                miner_hotkey=db_run.miner_hotkey,
                vali_uid=self.validator_uid,
                vali_hotkey=self.validator_hotkey,
                status=db_run.status.value,
                event_id=db_run.unique_event_id,
                version_id=UUID(db_run.agent_version_id),
                is_final=db_run.is_final,
            )
            runs.append(run)

        return PostAgentRunsRequestBody(runs=runs)

    async def export_runs_to_backend(self, payload: PostAgentRunsRequestBody) -> None:
        await self.api_client.post_agent_runs(body=payload)

        self.logger.debug(
            "Exported runs to backend",
            extra={"n_runs": len(payload.runs)},
        )

    async def run(self) -> None:
        unexported_runs = await self.db_operations.get_unexported_agent_runs(limit=self.batch_size)

        if not unexported_runs:
            self.logger.debug("No unexported runs to export")
        else:
            self.logger.debug(
                "Found unexported runs to export",
                extra={"n_runs": len(unexported_runs)},
            )

            payload = self.prepare_runs_payload(db_runs=unexported_runs)

            try:
                await self.export_runs_to_backend(payload)
            except Exception:
                self.errors_count += 1
                self.logger.exception("Failed to export runs to backend")
                return

            run_ids = [run.run_id for run in unexported_runs]
            await self.db_operations.mark_agent_runs_as_exported(run_ids=run_ids)

        self.logger.debug(
            "Export runs task completed",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
