import asyncio

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class DbCleaner(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    batch_size: int
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        batch_size: int,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        # Validate batch_size
        max_batch_size = 4000

        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > max_batch_size:
            raise ValueError(
                f"batch_size must be a positive integer equal or less than {max_batch_size}."
            )

        # Validate logger
        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.batch_size = batch_size
        self.logger = logger

    @property
    def name(self):
        return "db-cleaner"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        # Delete predictions
        deleted_predictions = await self.db_operations.delete_predictions(self.batch_size)

        if len(deleted_predictions) > 0:
            self.logger.debug(
                "Predictions deleted", extra={"deleted_count": len(deleted_predictions)}
            )

        await asyncio.sleep(1)

        # Delete scores
        deleted_scores = await self.db_operations.delete_scores(self.batch_size)

        if len(deleted_scores) > 0:
            self.logger.debug("Scores deleted", extra={"deleted_count": len(deleted_scores)})

        await asyncio.sleep(1)

        # Delete reasonings
        deleted_reasonings = await self.db_operations.delete_reasonings(self.batch_size)

        if len(deleted_reasonings) > 0:
            self.logger.debug(
                "Reasonings deleted", extra={"deleted_count": len(deleted_reasonings)}
            )

        await asyncio.sleep(1)

        # Delete agent run logs
        deleted_agent_run_logs = await self.db_operations.delete_agent_run_logs(self.batch_size)

        if len(deleted_agent_run_logs) > 0:
            self.logger.debug(
                "Agent run logs deleted", extra={"deleted_count": len(deleted_agent_run_logs)}
            )

        await asyncio.sleep(1)

        # Delete agent runs
        deleted_agent_runs = await self.db_operations.delete_agent_runs(self.batch_size)

        if len(deleted_agent_runs) > 0:
            self.logger.debug(
                "Agent runs deleted", extra={"deleted_count": len(deleted_agent_runs)}
            )

        await asyncio.sleep(1)

        if 2 > 1:
            return  # Temporarily disable event hard deletion

        # Delete events
        deleted_events = await self.db_operations.delete_events_hard_delete(self.batch_size)

        if len(deleted_events) > 0:
            self.logger.debug("Events hard deleted", extra={"deleted_count": len(deleted_events)})
