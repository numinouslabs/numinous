from datetime import datetime, timezone

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.reforecast_memory import MAX_MEMORY_CHARS, ReforecastMemoryForExport


class TestDbOperationsPart5(TestDbOperationsBase):
    async def _setup_agent_run(
        self,
        db_operations: DatabaseOperations,
        run_id: str,
        unique_event_id: str,
        miner_uid: int,
        miner_hotkey: str,
    ) -> None:
        event = EventsModel(
            unique_event_id=unique_event_id,
            event_id=f"event_{unique_event_id}",
            market_type="market_type",
            event_type="type",
            description="Test event",
            outcome=None,
            status=EventStatus.PENDING,
            metadata="{}",
            created_at="2024-01-01T00:00:00+00:00",
            cutoff="2024-12-31T23:59:59+00:00",
        )
        await db_operations.upsert_events([event])

        version_id = f"v-{run_id}"
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

        run = AgentRunsModel(
            interval_start_minutes=1000,
            run_id=run_id,
            unique_event_id=unique_event_id,
            agent_version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            status=AgentRunStatus.SUCCESS,
            exported=False,
            is_final=True,
        )
        await db_operations.upsert_agent_runs([run])

    async def test_insert_reforecast_memory(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reforecast_memory("run_1", "belief blob", 100)

        rows = await db_client.many(
            "SELECT run_id, memory, interval_start_minutes, exported FROM reforecast_memory"
        )

        assert len(rows) == 1
        assert rows[0] == ("run_1", "belief blob", 100, 0)

    async def test_insert_reforecast_memory_upsert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reforecast_memory("run_1", "original", 100)
        await db_operations.insert_reforecast_memory("run_1", "updated", 200)

        rows = await db_client.many(
            "SELECT run_id, memory, interval_start_minutes FROM reforecast_memory"
        )

        assert len(rows) == 1
        assert rows[0] == ("run_1", "updated", 200)

    async def test_insert_reforecast_memory_caps_length(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reforecast_memory("run_1", "x" * (MAX_MEMORY_CHARS + 500), 100)

        rows = await db_client.many("SELECT memory FROM reforecast_memory")

        assert len(rows[0][0]) == MAX_MEMORY_CHARS

    async def test_get_reforecast_memories_for_export(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_reforecast_memory("run_1", "memory 1", 100)
        await db_operations.insert_reforecast_memory("run_2", "memory 2", 200)

        result = await db_operations.get_reforecast_memories_for_export(limit=100)

        assert len(result) == 2
        assert all(isinstance(r, ReforecastMemoryForExport) for r in result)

        assert result[0].run_id == "run_1"
        assert result[0].memory == "memory 1"
        assert result[0].interval_start_minutes == 100
        assert result[0].event_id == "event_1"
        assert result[0].miner_uid == 10
        assert result[0].miner_hotkey == "hotkey_1"

        assert result[1].run_id == "run_2"
        assert result[1].memory == "memory 2"
        assert result[1].interval_start_minutes == 200
        assert result[1].event_id == "event_2"

    async def test_get_reforecast_memories_for_export_empty(
        self, db_operations: DatabaseOperations
    ):
        result = await db_operations.get_reforecast_memories_for_export(limit=100)
        assert len(result) == 0

    async def test_get_reforecast_memories_for_export_skips_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_reforecast_memory("run_1", "memory 1", 100)
        await db_operations.insert_reforecast_memory("run_2", "memory 2", 200)

        await db_operations.mark_reforecast_memories_as_exported(run_ids=["run_1"])

        result = await db_operations.get_reforecast_memories_for_export(limit=100)

        assert len(result) == 1
        assert result[0].run_id == "run_2"

    async def test_mark_reforecast_memories_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reforecast_memory("run_1", "memory 1", 100)
        await db_operations.insert_reforecast_memory("run_2", "memory 2", 100)
        await db_operations.insert_reforecast_memory("run_3", "memory 3", 100)

        await db_operations.mark_reforecast_memories_as_exported(run_ids=["run_1", "run_3"])

        rows = await db_client.many(
            "SELECT run_id, exported FROM reforecast_memory ORDER BY run_id"
        )

        assert rows == [("run_1", 1), ("run_2", 0), ("run_3", 1)]

    async def test_mark_reforecast_memories_as_exported_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reforecast_memory("run_1", "memory 1", 100)

        await db_operations.mark_reforecast_memories_as_exported(run_ids=[])

        rows = await db_client.many("SELECT run_id, exported FROM reforecast_memory")
        assert rows == [("run_1", 0)]

    async def test_delete_reforecast_memories_only_old_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        # Recent exported, old unexported, and old exported.
        await db_operations.insert_reforecast_memory("run_recent", "m", 100)
        await db_operations.insert_reforecast_memory("run_old_unexported", "m", 100)
        await db_operations.insert_reforecast_memory("run_old_exported", "m", 100)

        await db_operations.mark_reforecast_memories_as_exported(
            run_ids=["run_recent", "run_old_exported"]
        )
        await db_client.update(
            """
                UPDATE reforecast_memory
                SET created_at = datetime(CURRENT_TIMESTAMP, '-8 day')
                WHERE run_id IN ('run_old_unexported', 'run_old_exported')
            """,
            [],
        )

        await db_operations.delete_reforecast_memories(batch_size=100)

        rows = await db_client.many("SELECT run_id FROM reforecast_memory ORDER BY run_id")

        # Only the old AND exported row is deleted.
        assert rows == [("run_old_unexported",), ("run_recent",)]

    async def _setup_run_in_interval(
        self,
        db_operations: DatabaseOperations,
        run_id: str,
        interval_start_minutes: int,
        status: AgentRunStatus = AgentRunStatus.SUCCESS,
        is_final: bool = True,
    ) -> None:
        # Runs sharing one event + agent version, differing only by interval.
        event = EventsModel(
            unique_event_id="event_shared",
            event_id="event_shared_external",
            market_type="market_type",
            event_type="type",
            description="Test event",
            outcome=None,
            status=EventStatus.PENDING,
            metadata="{}",
            created_at="2024-01-01T00:00:00+00:00",
            cutoff="2024-12-31T23:59:59+00:00",
        )
        await db_operations.upsert_events([event])

        agent = MinerAgentsModel(
            version_id="version_shared",
            miner_uid=7,
            miner_hotkey="hotkey_shared",
            track="MAIN",
            agent_name="TestAgent",
            version_number=1,
            file_path="/data/agents/7/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_miner_agents([agent])

        await db_operations.upsert_agent_runs(
            [
                AgentRunsModel(
                    run_id=run_id,
                    unique_event_id="event_shared",
                    agent_version_id="version_shared",
                    miner_uid=7,
                    miner_hotkey="hotkey_shared",
                    track="MAIN",
                    status=status,
                    interval_start_minutes=interval_start_minutes,
                    exported=False,
                    is_final=is_final,
                )
            ]
        )

    async def test_has_final_run_true_for_same_interval(self, db_operations: DatabaseOperations):
        await self._setup_run_in_interval(db_operations, "run_a", interval_start_minutes=1440)

        assert await db_operations.has_final_run(
            unique_event_id="event_shared",
            agent_version_id="version_shared",
            interval_start_minutes=1440,
        )

    async def test_has_final_run_false_for_other_interval(self, db_operations: DatabaseOperations):
        # The lock is per interval: yesterday's final run must not block today.
        await self._setup_run_in_interval(db_operations, "run_a", interval_start_minutes=1440)

        assert not await db_operations.has_final_run(
            unique_event_id="event_shared",
            agent_version_id="version_shared",
            interval_start_minutes=2880,
        )

    async def test_count_runs_is_scoped_to_interval(self, db_operations: DatabaseOperations):
        await self._setup_run_in_interval(
            db_operations,
            "run_a",
            interval_start_minutes=1440,
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )
        await self._setup_run_in_interval(
            db_operations,
            "run_b",
            interval_start_minutes=2880,
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )

        # Timeout retries are counted within the interval, so each one starts fresh.
        for interval in (1440, 2880):
            assert (
                await db_operations.count_runs_for_event_and_agent(
                    unique_event_id="event_shared",
                    agent_version_id="version_shared",
                    interval_start_minutes=interval,
                    status=AgentRunStatus.SANDBOX_TIMEOUT,
                    is_final=False,
                )
                == 1
            )
