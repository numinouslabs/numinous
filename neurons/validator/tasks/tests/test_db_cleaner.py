from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from neurons.validator.db.dkg_operations import DKG_HISTORY, DkgDatabaseOperations
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.utils.logger.logger import NuminousLogger

NOW = datetime(2026, 10, 19, 16, 30, tzinfo=timezone.utc)


class TestDbCleanerTask:
    @pytest.fixture
    def dkg_operations(self) -> AsyncMock:
        return AsyncMock(spec=DkgDatabaseOperations)

    @pytest.fixture
    def logger(self) -> MagicMock:
        return MagicMock(spec=NuminousLogger)

    @pytest.mark.parametrize(
        "changed, error",
        [
            ({"interval_seconds": 3600}, ValueError),
            ({"interval_seconds": 0.0}, ValueError),
            ({"dkg_operations": "not operations"}, TypeError),
            ({"batch_size": 0}, ValueError),
            ({"batch_size": 4001}, ValueError),
            ({"logger": "not a logger"}, TypeError),
        ],
    )
    def test_init_rejects_bad_arguments(
        self, dkg_operations: AsyncMock, logger: MagicMock, changed: dict, error: type[Exception]
    ):
        arguments = {
            "interval_seconds": 3600.0,
            "dkg_operations": dkg_operations,
            "batch_size": 4000,
            "logger": logger,
        }

        with pytest.raises(error):
            DbCleaner(**(arguments | changed))

    def test_name_and_interval(self, dkg_operations: AsyncMock, logger: MagicMock):
        task = DbCleaner(
            interval_seconds=3600.0, dkg_operations=dkg_operations, batch_size=4000, logger=logger
        )

        assert task.name == "db-cleaner"
        assert task.interval_seconds == 3600.0

    @freeze_time(NOW)
    async def test_deletes_what_is_older_than_the_history_and_logs_it(
        self, dkg_operations: AsyncMock, logger: MagicMock
    ):
        dkg_operations.delete_expired.return_value = 33
        task = DbCleaner(
            interval_seconds=3600.0, dkg_operations=dkg_operations, batch_size=100, logger=logger
        )

        await task.run()

        dkg_operations.delete_expired.assert_awaited_once_with(
            expired_until=NOW - DKG_HISTORY, batch_size=100
        )
        logger.debug.assert_called_once_with("DKG history deleted", extra={"deleted_count": 33})

    async def test_nothing_expired_logs_nothing(self, dkg_operations: AsyncMock, logger: MagicMock):
        dkg_operations.delete_expired.return_value = 0
        task = DbCleaner(
            interval_seconds=3600.0, dkg_operations=dkg_operations, batch_size=100, logger=logger
        )

        await task.run()

        dkg_operations.delete_expired.assert_awaited_once()
        logger.debug.assert_not_called()
