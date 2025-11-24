from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.sync_miners_metadata import SyncMinersMetadata
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import NuminousLogger


@pytest.fixture
def mock_metagraph():
    metagraph = MagicMock(spec=IfMetagraph)
    metagraph.sync = AsyncMock()
    metagraph.block = np.int64(12345)
    metagraph.uids = np.array([], dtype=np.int64)
    metagraph.axons = {}
    metagraph.validator_trust = {}
    metagraph.validator_permit = np.array([], dtype=np.int64)
    return metagraph


class TestSyncMinersMetadataInit:
    def test_valid_initialization(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            metagraph=mock_metagraph,
            logger=logger,
        )

        assert task.interval == 300.0
        assert task.db_operations == db_operations
        assert task.metagraph == mock_metagraph
        assert task.logger == logger

    def test_invalid_interval_negative(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=-1.0,
                db_operations=db_operations,
                metagraph=mock_metagraph,
                logger=logger,
            )

    def test_invalid_interval_zero(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=0.0,
                db_operations=db_operations,
                metagraph=mock_metagraph,
                logger=logger,
            )

    def test_invalid_interval_not_float(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=300,
                db_operations=db_operations,
                metagraph=mock_metagraph,
                logger=logger,
            )

    def test_invalid_db_operations_type(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)

        with pytest.raises(TypeError, match="db_operations must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations="not_db_ops",
                metagraph=mock_metagraph,
                logger=logger,
            )

    def test_invalid_metagraph_type(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(TypeError, match="metagraph must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                metagraph="not_metagraph",
                logger=logger,
            )

    def test_invalid_logger_type(self, mock_metagraph):
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(TypeError, match="logger must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                metagraph=mock_metagraph,
                logger="not_logger",
            )


class TestSyncMinersMetadataProperties:
    def test_name_property(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            metagraph=mock_metagraph,
            logger=logger,
        )

        assert task.name == "sync-miners-metadata"

    def test_interval_seconds_property(self, mock_metagraph):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            metagraph=mock_metagraph,
            logger=logger,
        )

        assert task.interval_seconds == 300.0


class TestSyncMinersMetadataRun:
    async def test_run_first_sync(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        metagraph = MagicMock(spec=IfMetagraph)

        metagraph.sync = AsyncMock()
        metagraph.block = np.int64(12345)
        metagraph.uids = np.array([0, 1, 2], dtype=np.int64)

        axon0 = MagicMock()
        axon0.hotkey = "hotkey0"
        axon0.ip = "192.168.1.1"

        axon1 = MagicMock()
        axon1.hotkey = "hotkey1"
        axon1.ip = "192.168.1.2"

        axon2 = MagicMock()
        axon2.hotkey = "hotkey2"
        axon2.ip = "192.168.1.3"

        metagraph.axons = {0: axon0, 1: axon1, 2: axon2}
        metagraph.validator_trust = {
            0: np.float64(0.0),
            1: np.float64(0.5),
            2: np.float64(0.0),
        }
        metagraph.validator_permit = np.array([0, 1, 0], dtype=np.int64)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            metagraph=metagraph,
            logger=logger,
        )

        await task.run()

        metagraph.sync.assert_awaited_once()

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 3

    async def test_run_skips_none_axons(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        metagraph = MagicMock(spec=IfMetagraph)

        metagraph.sync = AsyncMock()
        metagraph.block = np.int64(12345)
        metagraph.uids = np.array([0, 1, 2], dtype=np.int64)

        axon0 = MagicMock()
        axon0.hotkey = "hotkey0"
        axon0.ip = "192.168.1.1"

        metagraph.axons = {0: axon0, 1: None, 2: None}
        metagraph.validator_trust = {
            0: np.float64(0.0),
            1: np.float64(0.0),
            2: np.float64(0.0),
        }
        metagraph.validator_permit = np.array([0, 0, 0], dtype=np.int64)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            metagraph=metagraph,
            logger=logger,
        )

        await task.run()

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 1
