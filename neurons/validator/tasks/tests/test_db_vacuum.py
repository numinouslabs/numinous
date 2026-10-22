from unittest.mock import AsyncMock, MagicMock

import pytest

from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.utils.logger.logger import NuminousLogger


class TestDbVacuumTask:
    @pytest.fixture
    def dkg_operations_mock(self):
        dkg_operations = AsyncMock(spec=DkgDatabaseOperations)

        return dkg_operations

    def test_vacuum_task_properties(self, dkg_operations_mock: AsyncMock):
        logger_mock = MagicMock(spec=NuminousLogger)
        interval = 300.0
        pages = 500

        vacuum_task = DbVacuum(
            interval_seconds=interval,
            dkg_operations=dkg_operations_mock,
            logger=logger_mock,
            pages=pages,
        )

        assert vacuum_task.name == "vacuum-task"
        assert vacuum_task.interval_seconds == interval
        assert vacuum_task.pages == pages
        assert vacuum_task._first_run is True

    async def test_db_vacuum_run(self, dkg_operations_mock: AsyncMock):
        dkg_operations_mock.vacuum_database = AsyncMock(return_value=None)
        logger_mock = MagicMock(spec=NuminousLogger)
        pages = 500

        vacuum_task = DbVacuum(
            interval_seconds=300.0,
            dkg_operations=dkg_operations_mock,
            logger=logger_mock,
            pages=pages,
        )

        await vacuum_task.run()

        # Assert - first run should be skipped
        dkg_operations_mock.vacuum_database.assert_not_awaited()

        await vacuum_task.run()

        # Assert - second run should execute vacuum
        dkg_operations_mock.vacuum_database.assert_awaited_once_with(pages)
