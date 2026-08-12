from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.tasks.sync_miners_metadata import SyncMinersMetadata
from neurons.validator.utils.logger.logger import NuminousLogger


class TestSyncMinersMetadataInit:
    def test_valid_initialization(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            network="test",
            logger=logger,
        )

        assert task.interval == 300.0
        assert task.db_operations == db_operations
        assert task.netuid == 99
        assert task.network == "test"
        assert task.logger == logger

    def test_invalid_interval_negative(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=-1.0,
                db_operations=db_operations,
                netuid=99,
                network="test",
                logger=logger,
            )

    def test_invalid_interval_zero(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=0.0,
                db_operations=db_operations,
                netuid=99,
                network="test",
                logger=logger,
            )

    def test_invalid_interval_not_float(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            SyncMinersMetadata(
                interval_seconds=300,
                db_operations=db_operations,
                netuid=99,
                network="test",
                logger=logger,
            )

    def test_invalid_db_operations_type(self):
        logger = MagicMock(spec=NuminousLogger)

        with pytest.raises(TypeError, match="db_operations must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations="not_db_ops",
                netuid=99,
                network="test",
                logger=logger,
            )

    def test_invalid_netuid(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="netuid must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=-1,
                network="test",
                logger=logger,
            )

    def test_invalid_network(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(ValueError, match="network must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=99,
                network="",
                logger=logger,
            )

    def test_invalid_logger_type(self):
        db_operations = MagicMock(spec=DatabaseOperations)

        with pytest.raises(TypeError, match="logger must be"):
            SyncMinersMetadata(
                interval_seconds=300.0,
                db_operations=db_operations,
                netuid=99,
                network="test",
                logger="not_logger",
            )


class TestSyncMinersMetadataProperties:
    def test_name_property(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            network="test",
            logger=logger,
        )

        assert task.name == "sync-miners-metadata"

    def test_interval_seconds_property(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            network="test",
            logger=logger,
        )

        assert task.interval_seconds == 300.0


class TestSyncMinersMetadataRun:
    def _neuron(self, uid: int, hotkey: str, axon: str | None):
        neuron = MagicMock()
        neuron.uid = uid
        neuron.hotkey = hotkey
        neuron.axon = axon
        return neuron

    def _patched_subtensor(self, metagraph: MagicMock) -> tuple[MagicMock, AsyncMock]:
        chain_client = AsyncMock()
        chain_client.subnets.metagraph = AsyncMock(return_value=metagraph)

        subtensor = MagicMock()
        subtensor.return_value.__aenter__ = AsyncMock(return_value=chain_client)
        subtensor.return_value.__aexit__ = AsyncMock(return_value=False)

        return subtensor, chain_client

    async def test_run_first_sync(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)

        metagraph = MagicMock()
        metagraph.block = 12345
        metagraph.neurons = [
            self._neuron(0, "hotkey0", "192.168.1.1:8091"),
            self._neuron(1, "hotkey1", "192.168.1.2:8091"),
            self._neuron(2, "hotkey2", "192.168.1.3:8091"),
        ]

        subtensor, chain_client = self._patched_subtensor(metagraph)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            network="test",
            logger=logger,
        )

        with patch("neurons.validator.tasks.sync_miners_metadata.Subtensor", subtensor):
            await task.run()

        subtensor.assert_called_once_with("test")
        chain_client.subnets.metagraph.assert_awaited_once_with(99)

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 3

        stored_ips = dict(
            await db_client.many("SELECT miner_uid, node_ip FROM miners ORDER BY miner_uid")
        )
        assert stored_ips == {"0": "192.168.1.1", "1": "192.168.1.2", "2": "192.168.1.3"}

    async def test_run_stores_unserved_axons_with_zero_ip(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)

        metagraph = MagicMock()
        metagraph.block = 12345
        metagraph.neurons = [
            self._neuron(0, "hotkey0", "192.168.1.1:8091"),
            self._neuron(1, "hotkey1", None),
            self._neuron(2, "hotkey2", None),
        ]

        subtensor, chain_client = self._patched_subtensor(metagraph)

        task = SyncMinersMetadata(
            interval_seconds=300.0,
            db_operations=db_operations,
            netuid=99,
            network="test",
            logger=logger,
        )

        with patch("neurons.validator.tasks.sync_miners_metadata.Subtensor", subtensor):
            await task.run()

        miners_count = await db_operations.get_miners_count()
        assert miners_count == 3

        stored_ips = dict(
            await db_client.many("SELECT miner_uid, node_ip FROM miners ORDER BY miner_uid")
        )
        assert stored_ips == {"0": "192.168.1.1", "1": "0.0.0.0", "2": "0.0.0.0"}
