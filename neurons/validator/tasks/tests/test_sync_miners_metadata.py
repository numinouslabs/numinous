from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from bittensor import AsyncSubtensor

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.sync_miners_metadata import SyncMinersMetadata
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import NuminousLogger


@pytest.fixture
def mock_subtensor():
    subtensor = MagicMock(spec=AsyncSubtensor)

    return subtensor


class TestSyncMinersMetadataInit:
    def test_valid_initialization(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            subtensor=mock_subtensor,
            logger=logger,
        )

        assert task.interval == 300.0
        assert task.db_operations == db_operations
        assert task.netuid == 99
        assert task.subtensor == mock_subtensor
        assert task.logger == logger

    def test_invalid_interval_negative(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=-1.0,
                db_operations=db_operations,
                netuid=99,
                subtensor=mock_subtensor,
                logger=logger,
            )

    def test_invalid_interval_zero(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=0.0,
                db_operations=db_operations,
                netuid=99,
                subtensor=mock_subtensor,
                logger=logger,
            )

    def test_invalid_interval_not_float(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=300,
                db_operations=db_operations,
                netuid=99,
                subtensor=mock_subtensor,
                logger=logger,
            )

    def test_invalid_db_operations_type(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)

        with pytest.raises(TypeError, match="db_operations must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations="not_db_ops",
                netuid=99,
                subtensor=mock_subtensor,
                logger=logger,
            )

    def test_invalid_netuid(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="netuid must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=-1,
                subtensor=mock_subtensor,
                logger=logger,
            )

    def test_invalid_subtensor_type(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(TypeError, match="subtensor must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=99,
                subtensor="not_subtensor",
                logger=logger,
            )

    def test_invalid_logger_type(self, mock_subtensor):
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(TypeError, match="logger must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=99,
                subtensor=mock_subtensor,
                logger="not_logger",
            )


class TestSyncMinersMetadataProperties:
    def test_name_property(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            subtensor=mock_subtensor,
            logger=logger,
        )

        assert task.name == "sync-miners-metadata"

    def test_interval_seconds_property(self, mock_subtensor):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            subtensor=mock_subtensor,
            logger=logger,
        )

        assert task.interval_seconds == 300.0


class TestSyncMinersMetadataRun:
    async def test_run_first_sync(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        metagraph = MagicMock(spec=IfMetagraph)

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

        subtensor = AsyncMock()
        subtensor.metagraph = AsyncMock(return_value=metagraph)

        subtensor_cm = AsyncMock(spec=AsyncSubtensor)
        subtensor_cm.__aenter__ = AsyncMock(return_value=subtensor)
        subtensor_cm.__aexit__ = AsyncMock(return_value=False)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            subtensor=subtensor_cm,
            logger=logger,
        )

        await task.run()

        subtensor_cm.__aenter__.assert_awaited_once()
        subtensor_cm.__aexit__.assert_awaited_once()
        subtensor.metagraph.assert_called_once_with(netuid=99, lite=True)

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 3

    async def test_run_skips_none_axons(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        metagraph = MagicMock(spec=IfMetagraph)

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

        subtensor = AsyncMock()
        subtensor.metagraph = AsyncMock(return_value=metagraph)

        subtensor_cm = AsyncMock(spec=AsyncSubtensor)
        subtensor_cm.__aenter__ = AsyncMock(return_value=subtensor)
        subtensor_cm.__aexit__ = AsyncMock(return_value=False)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            subtensor=subtensor_cm,
            logger=logger,
        )

        await task.run()

        subtensor_cm.__aenter__.assert_awaited_once()
        subtensor_cm.__aexit__.assert_awaited_once()
        subtensor.metagraph.assert_called_once_with(netuid=99, lite=True)

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 1
