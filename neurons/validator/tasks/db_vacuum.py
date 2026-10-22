from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger


class DbVacuum(AbstractTask):
    interval: float
    dkg_operations: DkgDatabaseOperations
    logger: NuminousLogger
    pages: int
    _first_run: bool

    def __init__(
        self,
        interval_seconds: float,
        dkg_operations: DkgDatabaseOperations,
        logger: NuminousLogger,
        pages: int,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(dkg_operations, DkgDatabaseOperations):
            raise TypeError("dkg_operations must be an instance of DkgDatabaseOperations.")

        # Validate logger
        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        # Validate pages
        if not isinstance(pages, int) or pages <= 0:
            raise ValueError("pages must be a positive integer.")

        self.interval = interval_seconds
        self.dkg_operations = dkg_operations
        self.logger = logger
        self.pages = pages
        self._first_run = True

    @property
    def name(self):
        return "vacuum-task"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        if self._first_run:
            self._first_run = False

            return

        await self.dkg_operations.vacuum_database(self.pages)
