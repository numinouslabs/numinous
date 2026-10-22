from datetime import datetime, timezone

from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class DbCleaner(AbstractTask):
    interval: float
    dkg_operations: DkgDatabaseOperations
    batch_size: int
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        dkg_operations: DkgDatabaseOperations,
        batch_size: int,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(dkg_operations, DkgDatabaseOperations):
            raise TypeError("dkg_operations must be an instance of DkgDatabaseOperations.")

        max_batch_size = 4000

        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > max_batch_size:
            raise ValueError(
                f"batch_size must be a positive integer equal or less than {max_batch_size}."
            )

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.dkg_operations = dkg_operations
        self.batch_size = batch_size
        self.logger = logger

    @property
    def name(self):
        return "db-cleaner"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        deleted_dkg_rows = await self.dkg_operations.delete_expired(
            expired_until=datetime.now(timezone.utc) - DKG_HISTORY, batch_size=self.batch_size
        )

        if deleted_dkg_rows > 0:
            self.logger.debug("DKG history deleted", extra={"deleted_count": deleted_dkg_rows})
