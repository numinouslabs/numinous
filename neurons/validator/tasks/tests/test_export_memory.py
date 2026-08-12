from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import (
    MemoryCommitRequestBody,
    MemoryCommitResponse,
    MemoryEntry,
)
from neurons.validator.models.reforecast_memory import ReforecastMemoryForExport
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_memory import ExportMemory
from neurons.validator.utils.common.interval import get_interval_iso_datetime
from neurons.validator.utils.logger.logger import NuminousLogger


class TestExportMemory:
    async def _create_event(self, db_operations: DatabaseOperations, unique_event_id: str) -> None:
        event = EventsModel(
            unique_event_id=unique_event_id,
            event_id=f"event_{unique_event_id}",
            market_type="test_market",
            event_type="test_type",
            description="Test event",
            outcome=None,
            status=EventStatus.PENDING,
            metadata="{}",
            created_at="2024-01-01T00:00:00+00:00",
            cutoff="2024-12-31T23:59:59+00:00",
        )
        await db_operations.upsert_events([event])

    async def _create_miner_agent(
        self, db_operations: DatabaseOperations, version_id: str, miner_uid: int, miner_hotkey: str
    ) -> None:
        agent = MinerAgentsModel(
            version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            agent_name="TestAgent",
            version_number=1,
            file_path=f"/data/agents/{miner_uid}/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_miner_agents([agent])

    async def _create_agent_run(
        self,
        db_operations: DatabaseOperations,
        run_id: str,
        unique_event_id: str,
        agent_version_id: str,
        miner_uid: int,
        miner_hotkey: str,
    ) -> None:
        run = AgentRunsModel(
            interval_start_minutes=1000,
            run_id=run_id,
            unique_event_id=unique_event_id,
            agent_version_id=agent_version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            status=AgentRunStatus.SUCCESS,
            exported=False,
            is_final=True,
        )
        await db_operations.upsert_agent_runs([run])

    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "validator_hotkey_test"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)
        bt_wallet.hotkey.ss58_address = "validator_hotkey_test"

        return bt_wallet

    @pytest.fixture
    def export_memory_task(self, db_operations: DatabaseOperations, bt_wallet: Wallet):
        api_client = NuminousClient(
            env="test", logger=MagicMock(spec=NuminousLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=NuminousLogger)

        return ExportMemory(
            interval_seconds=300.0,
            batch_size=500,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger,
        )

    def test_init(self, export_memory_task):
        unit = export_memory_task

        assert isinstance(unit, ExportMemory)
        assert unit.interval == 300.0
        assert unit.interval_seconds == 300.0
        assert unit.batch_size == 500
        assert unit.errors_count == 0
        assert unit.name == "export-memory"

    def test_prepare_payload(self, export_memory_task: ExportMemory):
        memories = [
            ReforecastMemoryForExport(
                run_id="123e4567-e89b-12d3-a456-426614174000",
                memory="belief blob 1",
                interval_start_minutes=100,
                created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
                event_id="ifgames-event-1",
                miner_uid=10,
                miner_hotkey="miner_hotkey_1",
            ),
            ReforecastMemoryForExport(
                run_id="223e4567-e89b-12d3-a456-426614174001",
                memory="belief blob 2",
                interval_start_minutes=200,
                created_at=datetime(2024, 6, 15, 12, 5, 0, tzinfo=timezone.utc),
                event_id="ifgames-event-2",
                miner_uid=20,
                miner_hotkey="miner_hotkey_2",
            ),
        ]

        payload = export_memory_task.prepare_payload(memories)

        assert isinstance(payload, MemoryCommitRequestBody)
        assert len(payload.entries) == 2

        first = payload.entries[0]
        assert isinstance(first, MemoryEntry)
        assert first.miner_uid == 10
        assert first.miner_hotkey == "miner_hotkey_1"
        assert first.event_id == "ifgames-event-1"
        assert first.memory == "belief blob 1"
        assert first.interval_datetime == datetime.fromisoformat(get_interval_iso_datetime(100))

        second = payload.entries[1]
        assert second.event_id == "ifgames-event-2"
        assert second.interval_datetime == datetime.fromisoformat(get_interval_iso_datetime(200))

    async def test_run_no_unexported_memories(self, export_memory_task: ExportMemory):
        export_memory_task.api_client = AsyncMock(spec=NuminousClient)

        await export_memory_task.run()

        export_memory_task.logger.debug.assert_any_call("No unexported memories to export")
        export_memory_task.api_client.commit_memory.assert_not_called()

    async def test_run_with_unexported_memories(
        self,
        export_memory_task: ExportMemory,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_memory_task
        unit.api_client.commit_memory = AsyncMock(return_value=MemoryCommitResponse(inserted=2))

        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_miner_agent(
            db_operations, "c23e4567-e89b-12d3-a456-42661417400b", 10, "miner_hotkey_1"
        )
        await self._create_miner_agent(
            db_operations, "e23e4567-e89b-12d3-a456-42661417400d", 20, "miner_hotkey_2"
        )

        await self._create_agent_run(
            db_operations,
            run_id="b23e4567-e89b-12d3-a456-42661417400a",
            unique_event_id="event_1",
            agent_version_id="c23e4567-e89b-12d3-a456-42661417400b",
            miner_uid=10,
            miner_hotkey="miner_hotkey_1",
        )
        await self._create_agent_run(
            db_operations,
            run_id="d23e4567-e89b-12d3-a456-42661417400c",
            unique_event_id="event_2",
            agent_version_id="e23e4567-e89b-12d3-a456-42661417400d",
            miner_uid=20,
            miner_hotkey="miner_hotkey_2",
        )

        await db_operations.insert_reforecast_memory(
            "b23e4567-e89b-12d3-a456-42661417400a", "memory for event 1", 100
        )
        await db_operations.insert_reforecast_memory(
            "d23e4567-e89b-12d3-a456-42661417400c", "memory for event 2", 200
        )

        await unit.run()

        unit.api_client.commit_memory.assert_called_once()
        payload = unit.api_client.commit_memory.call_args.kwargs["body"]

        assert len(payload.entries) == 2
        assert payload.entries[0].event_id == "event_1"
        assert payload.entries[0].miner_uid == 10
        assert payload.entries[0].miner_hotkey == "miner_hotkey_1"
        assert payload.entries[0].memory == "memory for event 1"

        result = await db_client.many("SELECT exported FROM reforecast_memory ORDER BY run_id")
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 1

    async def test_run_marks_exported_even_when_inserted_zero(
        self,
        export_memory_task: ExportMemory,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        # First-writer-wins: another validator won the interval, backend returns 0.
        # We must still mark the row exported so it's not re-sent forever.
        unit = export_memory_task
        unit.api_client.commit_memory = AsyncMock(return_value=MemoryCommitResponse(inserted=0))

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(
            db_operations, "c23e4567-e89b-12d3-a456-42661417400b", 10, "miner_hotkey_1"
        )
        await self._create_agent_run(
            db_operations,
            run_id="b23e4567-e89b-12d3-a456-42661417400a",
            unique_event_id="event_1",
            agent_version_id="c23e4567-e89b-12d3-a456-42661417400b",
            miner_uid=10,
            miner_hotkey="miner_hotkey_1",
        )
        await db_operations.insert_reforecast_memory(
            "b23e4567-e89b-12d3-a456-42661417400a", "memory", 100
        )

        await unit.run()

        result = await db_client.many("SELECT exported FROM reforecast_memory")
        assert result[0][0] == 1

    async def test_run_export_exception(
        self,
        export_memory_task: ExportMemory,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_memory_task
        unit.api_client.commit_memory = AsyncMock(side_effect=Exception("Simulated failure"))

        await self._create_event(db_operations, "event_error")
        await self._create_miner_agent(
            db_operations, "023e4567-e89b-12d3-a456-42661417400f", 30, "miner_hotkey_3"
        )
        await self._create_agent_run(
            db_operations,
            run_id="f23e4567-e89b-12d3-a456-42661417400e",
            unique_event_id="event_error",
            agent_version_id="023e4567-e89b-12d3-a456-42661417400f",
            miner_uid=30,
            miner_hotkey="miner_hotkey_3",
        )
        await db_operations.insert_reforecast_memory(
            "f23e4567-e89b-12d3-a456-42661417400e", "some memory", 100
        )

        await unit.run()

        unit.logger.exception.assert_called_with("Failed to export memories to backend")

        result = await db_client.many("SELECT exported FROM reforecast_memory")
        assert len(result) == 1
        assert result[0][0] == 0
