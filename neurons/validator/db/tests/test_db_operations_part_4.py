import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.utils.logger.logger import NuminousLogger


class TestDbOperationsPart4(TestDbOperationsBase):
    @pytest.fixture
    async def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        return db_operations

    async def _create_event(self, db_operations: DatabaseOperations, unique_event_id: str) -> None:
        """Helper to create an event for FK constraint"""
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
        self, db_operations: DatabaseOperations, version_id: str, miner_uid: int = 42
    ) -> None:
        """Helper to create a miner agent for FK constraint"""
        agent = MinerAgentsModel(
            version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey="5GTest...",
            agent_name="TestAgent",
            version_number=1,
            file_path=f"/data/agents/{miner_uid}/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_miner_agents([agent])

    async def test_upsert_agent_runs_insert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test inserting a new agent run"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )

        await db_operations.upsert_agent_runs([run])

        # Verify insertion
        rows = await db_client.many(
            "SELECT run_id, status, is_final FROM agent_runs WHERE run_id = ?",
            [run.run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run.run_id
        assert rows[0][1] == AgentRunStatus.SUCCESS.value
        assert rows[0][2] == 1  # is_final defaults to True

    async def test_upsert_agent_runs_update(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test updating an existing agent run"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())

        # Insert initial run
        initial_run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )

        await db_operations.upsert_agent_runs([initial_run])

        # Update with new status
        updated_run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
            is_final=True,
        )

        await db_operations.upsert_agent_runs([updated_run])

        # Verify update
        rows = await db_client.many(
            "SELECT run_id, status, is_final FROM agent_runs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run_id
        assert rows[0][1] == AgentRunStatus.SUCCESS.value
        assert rows[0][2] == 1  # is_final updated to True

    async def test_upsert_agent_runs_empty(self, db_operations: DatabaseOperations):
        """Test inserting with empty list does nothing"""
        await db_operations.upsert_agent_runs([])

    async def test_get_unexported_agent_runs_empty(self, db_operations: DatabaseOperations):
        """Test getting unexported runs when none exist"""
        runs = await db_operations.get_unexported_agent_runs()
        assert runs == []

    async def test_get_unexported_agent_runs(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test getting unexported runs"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_event(db_operations, "event_3")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)
        await self._create_miner_agent(db_operations, "agent_v2", miner_uid=2)
        await self._create_miner_agent(db_operations, "agent_v3", miner_uid=3)

        # Insert exported run
        exported_run = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=True,
        )
        await db_operations.upsert_agent_runs([exported_run])

        # Insert unexported runs
        unexported_run_1 = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id="event_2",
            agent_version_id="agent_v2",
            miner_uid=2,
            miner_hotkey="hotkey_2",
            status=AgentRunStatus.SUCCESS,
            exported=False,
        )
        await db_operations.upsert_agent_runs([unexported_run_1])

        unexported_run_2 = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id="event_3",
            agent_version_id="agent_v3",
            miner_uid=3,
            miner_hotkey="hotkey_3",
            status=AgentRunStatus.INTERNAL_AGENT_ERROR,
            exported=False,
        )
        await db_operations.upsert_agent_runs([unexported_run_2])

        # Get unexported runs
        runs = await db_operations.get_unexported_agent_runs()

        assert len(runs) == 2
        run_ids = [run.run_id for run in runs]
        assert unexported_run_1.run_id in run_ids
        assert unexported_run_2.run_id in run_ids
        assert exported_run.run_id not in run_ids

    async def test_get_unexported_agent_runs_with_limit(self, db_operations: DatabaseOperations):
        """Test getting unexported runs with limit"""
        # Create FK dependencies
        for i in range(5):
            await self._create_event(db_operations, f"event_{i}")
            await self._create_miner_agent(db_operations, f"agent_v{i}", miner_uid=i)

        # Insert 5 unexported runs
        for i in range(5):
            run = AgentRunsModel(
                run_id=str(uuid4()),
                unique_event_id=f"event_{i}",
                agent_version_id=f"agent_v{i}",
                miner_uid=i,
                miner_hotkey=f"hotkey_{i}",
                status=AgentRunStatus.SUCCESS,
                exported=False,
            )
            await db_operations.upsert_agent_runs([run])

        # Get only 3
        runs = await db_operations.get_unexported_agent_runs(limit=3)

        assert len(runs) == 3

    async def test_mark_agent_runs_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking runs as exported"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)
        await self._create_miner_agent(db_operations, "agent_v2", miner_uid=2)

        # Insert unexported runs
        run_1_id = str(uuid4())
        run_2_id = str(uuid4())

        run_1 = AgentRunsModel(
            run_id=run_1_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=False,
        )
        await db_operations.upsert_agent_runs([run_1])

        run_2 = AgentRunsModel(
            run_id=run_2_id,
            unique_event_id="event_2",
            agent_version_id="agent_v2",
            miner_uid=2,
            miner_hotkey="hotkey_2",
            status=AgentRunStatus.SUCCESS,
            exported=False,
        )
        await db_operations.upsert_agent_runs([run_2])

        # Mark as exported
        await db_operations.mark_agent_runs_as_exported([run_1_id, run_2_id])

        # Verify they're marked
        rows = await db_client.many(
            "SELECT run_id, exported FROM agent_runs WHERE run_id IN (?, ?)",
            [run_1_id, run_2_id],
        )

        assert len(rows) == 2
        for row in rows:
            assert row[1] == 1  # exported = True

    async def test_mark_agent_runs_as_exported_empty_list(self, db_operations: DatabaseOperations):
        """Test marking empty list does nothing"""
        await db_operations.mark_agent_runs_as_exported([])

    async def test_retry_scenario(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test multiple runs for same event/miner with is_final flag"""
        # Create FK dependencies
        event_id = "event_retry_test"
        miner_uid = 42
        agent_version_id = "agent_v1"
        await self._create_event(db_operations, event_id)
        await self._create_miner_agent(db_operations, agent_version_id, miner_uid=miner_uid)

        # First attempt - timeout, not final
        run_1 = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id=event_id,
            agent_version_id=agent_version_id,
            miner_uid=miner_uid,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )
        await db_operations.upsert_agent_runs([run_1])

        # Second attempt - timeout, not final
        run_2 = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id=event_id,
            agent_version_id=agent_version_id,
            miner_uid=miner_uid,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )
        await db_operations.upsert_agent_runs([run_2])

        # Third attempt - success, final
        run_3 = AgentRunsModel(
            run_id=str(uuid4()),
            unique_event_id=event_id,
            agent_version_id=agent_version_id,
            miner_uid=miner_uid,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SUCCESS,
            is_final=True,
        )
        await db_operations.upsert_agent_runs([run_3])

        # Verify all 3 runs exist
        all_runs = await db_client.many(
            "SELECT run_id, status, is_final FROM agent_runs WHERE unique_event_id = ? ORDER BY created_at",
            [event_id],
        )

        assert len(all_runs) == 3
        assert all_runs[0][2] == 0  # is_final = False
        assert all_runs[1][2] == 0  # is_final = False
        assert all_runs[2][2] == 1  # is_final = True

        # Verify only final run when filtering
        final_runs = await db_client.many(
            "SELECT run_id FROM agent_runs WHERE unique_event_id = ? AND is_final = 1",
            [event_id],
        )

        assert len(final_runs) == 1
        assert final_runs[0][0] == run_3.run_id

    async def test_insert_agent_run_log(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test inserting a new agent run log"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Insert log
        log_content = "Agent execution completed successfully\nOutput: prediction=0.75"
        await db_operations.insert_agent_run_log(run_id, log_content)

        # Verify insertion
        rows = await db_client.many(
            "SELECT run_id, log_content, exported FROM agent_run_logs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run_id
        assert rows[0][1] == log_content
        assert rows[0][2] == 0  # exported defaults to NOT_EXPORTED

    async def test_insert_agent_run_log_upsert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test updating an existing log (ON CONFLICT behavior)"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Insert initial log
        initial_log = "First log entry"
        await db_operations.insert_agent_run_log(run_id, initial_log)

        # Update with new log
        updated_log = "Updated log entry with more details"
        await db_operations.insert_agent_run_log(run_id, updated_log)

        # Verify only one row exists with updated content
        rows = await db_client.many(
            "SELECT run_id, log_content FROM agent_run_logs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run_id
        assert rows[0][1] == updated_log

    async def test_insert_agent_run_log_truncation(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test log content truncation at 30,000 chars"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Create log larger than 30,000 chars
        large_log = "x" * 35000
        await db_operations.insert_agent_run_log(run_id, large_log)

        # Verify truncation
        rows = await db_client.many(
            "SELECT run_id, log_content FROM agent_run_logs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run_id
        assert len(rows[0][1]) == 30000  # Truncated
        assert rows[0][1] == "x" * 30000

    async def test_insert_agent_run_log_empty_content(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test inserting log with empty content"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Insert empty log
        await db_operations.insert_agent_run_log(run_id, "")

        # Verify insertion
        rows = await db_client.many(
            "SELECT run_id, log_content FROM agent_run_logs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == run_id
        assert rows[0][1] == ""

    async def test_insert_agent_run_log_special_characters(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test inserting log with special characters and unicode"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_123")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=42)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_123",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Log with special characters
        special_log = """Line 1: Normal text
Line 2: Special chars: !@#$%^&*()
Line 3: Unicode: 你好世界 🚀
Line 4: Newlines and tabs\t\n"""

        await db_operations.insert_agent_run_log(run_id, special_log)

        # Verify insertion
        rows = await db_client.many(
            "SELECT log_content FROM agent_run_logs WHERE run_id = ?",
            [run_id],
        )

        assert len(rows) == 1
        assert rows[0][0] == special_log

    async def test_insert_agent_run_log_fk_constraint(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test FK constraint - can't insert log without corresponding agent_run"""
        non_existent_run_id = str(uuid4())

        # Attempt to insert log for non-existent run_id
        with pytest.raises(Exception):  # Should raise FOREIGN KEY constraint error
            await db_operations.insert_agent_run_log(non_existent_run_id, "This should fail")

    async def test_get_unexported_agent_run_logs_empty(self, db_operations: DatabaseOperations):
        """Test getting unexported logs when none exist"""
        logs = await db_operations.get_unexported_agent_run_logs()
        assert logs == []

    async def test_get_unexported_agent_run_logs(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test getting unexported logs"""
        # Create FK dependencies and runs
        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_event(db_operations, "event_3")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)
        await self._create_miner_agent(db_operations, "agent_v2", miner_uid=2)
        await self._create_miner_agent(db_operations, "agent_v3", miner_uid=3)

        run_1_id = str(uuid4())
        run_2_id = str(uuid4())
        run_3_id = str(uuid4())

        # Create runs
        run_1 = AgentRunsModel(
            run_id=run_1_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run_1])

        run_2 = AgentRunsModel(
            run_id=run_2_id,
            unique_event_id="event_2",
            agent_version_id="agent_v2",
            miner_uid=2,
            miner_hotkey="hotkey_2",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run_2])

        run_3 = AgentRunsModel(
            run_id=run_3_id,
            unique_event_id="event_3",
            agent_version_id="agent_v3",
            miner_uid=3,
            miner_hotkey="hotkey_3",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run_3])

        # Insert exported log
        await db_operations.insert_agent_run_log(run_1_id, "Exported log content")
        await db_client.update(
            "UPDATE agent_run_logs SET exported = 1 WHERE run_id = ?", [run_1_id]
        )

        # Insert unexported logs
        await db_operations.insert_agent_run_log(run_2_id, "Unexported log 1")
        await db_operations.insert_agent_run_log(run_3_id, "Unexported log 2")

        # Get unexported logs
        logs = await db_operations.get_unexported_agent_run_logs()

        assert len(logs) == 2
        run_ids = [log.run_id for log in logs]
        assert run_2_id in run_ids
        assert run_3_id in run_ids
        assert run_1_id not in run_ids

    async def test_get_unexported_agent_run_logs_ordering(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test that logs are ordered by created_at ASC (oldest first)"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_event(db_operations, "event_3")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)
        await self._create_miner_agent(db_operations, "agent_v2", miner_uid=2)
        await self._create_miner_agent(db_operations, "agent_v3", miner_uid=3)

        run_ids = [str(uuid4()), str(uuid4()), str(uuid4())]

        # Create runs
        for i, run_id in enumerate(run_ids):
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id=f"event_{i+1}",
                agent_version_id=f"agent_v{i+1}",
                miner_uid=i + 1,
                miner_hotkey=f"hotkey_{i+1}",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log {i+1}")

        # Get logs
        logs = await db_operations.get_unexported_agent_run_logs()

        assert len(logs) == 3
        # Should be ordered by created_at (oldest first)
        assert logs[0].run_id == run_ids[0]
        assert logs[1].run_id == run_ids[1]
        assert logs[2].run_id == run_ids[2]

    async def test_get_unexported_agent_run_logs_with_limit(
        self, db_operations: DatabaseOperations
    ):
        """Test getting unexported logs with limit"""
        # Create FK dependencies
        for i in range(5):
            await self._create_event(db_operations, f"event_{i}")
            await self._create_miner_agent(db_operations, f"agent_v{i}", miner_uid=i)

        # Create 5 runs and logs
        for i in range(5):
            run_id = str(uuid4())
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id=f"event_{i}",
                agent_version_id=f"agent_v{i}",
                miner_uid=i,
                miner_hotkey=f"hotkey_{i}",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log {i}")

        # Get only 3
        logs = await db_operations.get_unexported_agent_run_logs(limit=3)

        assert len(logs) == 3

    async def test_get_unexported_agent_run_logs_mixed_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test filtering of mixed exported/unexported logs"""
        # Create FK dependencies
        for i in range(4):
            await self._create_event(db_operations, f"event_{i}")
            await self._create_miner_agent(db_operations, f"agent_v{i}", miner_uid=i)

        run_ids = []
        for i in range(4):
            run_id = str(uuid4())
            run_ids.append(run_id)
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id=f"event_{i}",
                agent_version_id=f"agent_v{i}",
                miner_uid=i,
                miner_hotkey=f"hotkey_{i}",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log {i}")

        # Mark some as exported (0 and 2)
        await db_client.update(
            "UPDATE agent_run_logs SET exported = 1 WHERE run_id IN (?, ?)",
            [run_ids[0], run_ids[2]],
        )

        # Get unexported
        logs = await db_operations.get_unexported_agent_run_logs()

        assert len(logs) == 2
        returned_run_ids = [log.run_id for log in logs]
        assert run_ids[1] in returned_run_ids
        assert run_ids[3] in returned_run_ids
        assert run_ids[0] not in returned_run_ids
        assert run_ids[2] not in returned_run_ids

    async def test_get_unexported_agent_run_logs_returns_model(
        self, db_operations: DatabaseOperations
    ):
        """Test that method returns proper AgentRunLogsModel objects"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        log_content = "Test log content"
        await db_operations.insert_agent_run_log(run_id, log_content)

        # Get logs
        logs = await db_operations.get_unexported_agent_run_logs()

        assert len(logs) == 1
        log = logs[0]

        # Verify it's a proper model with all fields
        assert log.run_id == run_id
        assert log.log_content == log_content
        assert log.exported is False
        assert log.created_at is not None
        assert log.updated_at is not None

    async def test_mark_agent_run_logs_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking logs as exported"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)
        await self._create_miner_agent(db_operations, "agent_v2", miner_uid=2)

        # Create agent runs
        run_1_id = str(uuid4())
        run_2_id = str(uuid4())

        run_1 = AgentRunsModel(
            run_id=run_1_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run_1])

        run_2 = AgentRunsModel(
            run_id=run_2_id,
            unique_event_id="event_2",
            agent_version_id="agent_v2",
            miner_uid=2,
            miner_hotkey="hotkey_2",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run_2])

        # Insert unexported logs
        await db_operations.insert_agent_run_log(run_1_id, "Log content 1")
        await db_operations.insert_agent_run_log(run_2_id, "Log content 2")

        # Mark as exported
        await db_operations.mark_agent_run_logs_as_exported([run_1_id, run_2_id])

        # Verify they're marked
        rows = await db_client.many(
            "SELECT run_id, exported FROM agent_run_logs WHERE run_id IN (?, ?)",
            [run_1_id, run_2_id],
        )

        assert len(rows) == 2
        for row in rows:
            assert row[1] == 1  # exported = True

    async def test_mark_agent_run_logs_as_exported_empty_list(
        self, db_operations: DatabaseOperations
    ):
        """Test marking empty list does nothing"""
        await db_operations.mark_agent_run_logs_as_exported([])

    async def test_mark_agent_run_logs_as_exported_single(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking single log as exported"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        # Create agent run
        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Insert log
        await db_operations.insert_agent_run_log(run_id, "Log content")

        # Mark as exported
        await db_operations.mark_agent_run_logs_as_exported([run_id])

        # Verify
        row = await db_client.one("SELECT exported FROM agent_run_logs WHERE run_id = ?", [run_id])
        assert row[0] == 1

    async def test_mark_agent_run_logs_as_exported_partial(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking only some logs as exported"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        # Create agent runs
        run_1_id = str(uuid4())
        run_2_id = str(uuid4())
        run_3_id = str(uuid4())

        for run_id in [run_1_id, run_2_id, run_3_id]:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log for {run_id}")

        # Mark only first two as exported
        await db_operations.mark_agent_run_logs_as_exported([run_1_id, run_2_id])

        # Verify only first two are exported
        rows = await db_client.many(
            "SELECT run_id, exported FROM agent_run_logs ORDER BY run_id",
            [],
        )

        assert len(rows) == 3
        exported_run_ids = {row[0] for row in rows if row[1] == 1}
        unexported_run_ids = {row[0] for row in rows if row[1] == 0}

        assert exported_run_ids == {run_1_id, run_2_id}
        assert unexported_run_ids == {run_3_id}

    async def test_mark_agent_run_logs_as_exported_updates_timestamp(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking as exported updates the updated_at timestamp"""
        # Create FK dependencies
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        # Create agent run
        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])

        # Insert log
        await db_operations.insert_agent_run_log(run_id, "Log content")

        # Get initial timestamp
        row_before = await db_client.one(
            "SELECT updated_at FROM agent_run_logs WHERE run_id = ?", [run_id]
        )
        updated_at_before = row_before[0]

        # Wait for timestamp to change (SQLite CURRENT_TIMESTAMP has second precision)
        await asyncio.sleep(1.1)

        # Mark as exported
        await db_operations.mark_agent_run_logs_as_exported([run_id])

        # Get new timestamp
        row_after = await db_client.one(
            "SELECT updated_at FROM agent_run_logs WHERE run_id = ?", [run_id]
        )
        updated_at_after = row_after[0]

        # Verify timestamp was updated
        assert updated_at_after > updated_at_before

    async def test_mark_agent_run_logs_as_exported_nonexistent(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test marking nonexistent logs does not error"""
        # Try to mark logs that don't exist
        await db_operations.mark_agent_run_logs_as_exported(["nonexistent_1", "nonexistent_2"])

        # Verify no logs exist
        count = await db_client.one("SELECT COUNT(*) FROM agent_run_logs", [])
        assert count[0] == 0

    async def test_delete_agent_run_logs_old_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test deletion of old exported logs"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id_old = str(uuid4())
        run_id_recent = str(uuid4())

        for run_id in [run_id_old, run_id_recent]:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])

        await db_operations.insert_agent_run_log(run_id_old, "Old log")
        await db_operations.insert_agent_run_log(run_id_recent, "Recent log")

        await db_operations.mark_agent_run_logs_as_exported([run_id_old, run_id_recent])

        await db_client.update(
            "UPDATE agent_run_logs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id_old],
        )

        deleted = await db_operations.delete_agent_run_logs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 1

        remaining = await db_client.many("SELECT run_id FROM agent_run_logs", [])
        assert len(remaining) == 1
        assert remaining[0][0] == run_id_recent

    async def test_delete_agent_run_logs_batch_size(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test batch_size limiting"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_ids = [str(uuid4()) for _ in range(5)]
        for run_id in run_ids:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, "Log content")

        await db_operations.mark_agent_run_logs_as_exported(run_ids)

        await db_client.update(
            "UPDATE agent_run_logs SET created_at = datetime('now', '-8 day')",
            [],
        )

        deleted = await db_operations.delete_agent_run_logs(batch_size=3)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 3

        remaining = await db_client.many("SELECT run_id FROM agent_run_logs", [])
        assert len(remaining) == 2

    async def test_delete_agent_run_logs_no_delete_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test unexported logs are not deleted"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])
        await db_operations.insert_agent_run_log(run_id, "Log content")

        await db_client.update(
            "UPDATE agent_run_logs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id],
        )

        deleted = await db_operations.delete_agent_run_logs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 0

        remaining = await db_client.many("SELECT run_id FROM agent_run_logs", [])
        assert len(remaining) == 1

    async def test_delete_agent_run_logs_no_delete_recent_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test recent exported logs are not deleted"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])
        await db_operations.insert_agent_run_log(run_id, "Log content")
        await db_operations.mark_agent_run_logs_as_exported([run_id])

        deleted = await db_operations.delete_agent_run_logs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 0

        remaining = await db_client.many("SELECT run_id FROM agent_run_logs", [])
        assert len(remaining) == 1

    async def test_delete_agent_run_logs_returns_rowids(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test that deleted ROWIDs are returned"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
        )
        await db_operations.upsert_agent_runs([run])
        await db_operations.insert_agent_run_log(run_id, "Log content")
        await db_operations.mark_agent_run_logs_as_exported([run_id])

        await db_client.update(
            "UPDATE agent_run_logs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id],
        )

        deleted = await db_operations.delete_agent_run_logs(batch_size=10)

        assert len(deleted) == 1
        assert isinstance(deleted[0], tuple)
        assert isinstance(deleted[0][0], int)

    async def test_delete_agent_runs_old_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test deletion of old exported runs"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id_old = str(uuid4())
        run_id_recent = str(uuid4())

        for run_id in [run_id_old, run_id_recent]:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=True,
            )
            await db_operations.upsert_agent_runs([run])

        await db_client.update(
            "UPDATE agent_runs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id_old],
        )

        deleted = await db_operations.delete_agent_runs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 1

        remaining = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining) == 1
        assert remaining[0][0] == run_id_recent

    async def test_delete_agent_runs_batch_size(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test batch_size limiting"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_ids = [str(uuid4()) for _ in range(5)]
        for run_id in run_ids:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=True,
            )
            await db_operations.upsert_agent_runs([run])

        await db_client.update(
            "UPDATE agent_runs SET created_at = datetime('now', '-8 day')",
            [],
        )

        deleted = await db_operations.delete_agent_runs(batch_size=3)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 3

        remaining = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining) == 2

    async def test_delete_agent_runs_no_delete_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test unexported runs are not deleted"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=False,
        )
        await db_operations.upsert_agent_runs([run])

        await db_client.update(
            "UPDATE agent_runs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id],
        )

        deleted = await db_operations.delete_agent_runs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 0

        remaining = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining) == 1

    async def test_delete_agent_runs_no_delete_recent_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test recent exported runs are not deleted"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=True,
        )
        await db_operations.upsert_agent_runs([run])

        deleted = await db_operations.delete_agent_runs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 0

        remaining = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining) == 1

    async def test_delete_agent_runs_no_delete_with_logs(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test runs with remaining logs are not deleted (FK constraint protection)"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id_with_log = str(uuid4())
        run_id_without_log = str(uuid4())

        for run_id in [run_id_with_log, run_id_without_log]:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=True,
            )
            await db_operations.upsert_agent_runs([run])

        await db_operations.insert_agent_run_log(run_id_with_log, "Log content")

        await db_client.update(
            "UPDATE agent_runs SET created_at = datetime('now', '-8 day')",
            [],
        )

        deleted = await db_operations.delete_agent_runs(batch_size=10)
        deleted_rowids = [row[0] for row in deleted]

        assert len(deleted_rowids) == 1

        remaining = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining) == 1
        assert remaining[0][0] == run_id_with_log

    async def test_delete_agent_runs_coordination_with_logs(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test deletion coordination: logs deleted first, then runs"""
        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", miner_uid=1)

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=True,
        )
        await db_operations.upsert_agent_runs([run])
        await db_operations.insert_agent_run_log(run_id, "Log content")
        await db_operations.mark_agent_run_logs_as_exported([run_id])

        await db_client.update(
            "UPDATE agent_runs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id],
        )
        await db_client.update(
            "UPDATE agent_run_logs SET created_at = datetime('now', '-8 day') WHERE run_id = ?",
            [run_id],
        )

        deleted_runs = await db_operations.delete_agent_runs(batch_size=10)
        assert len(deleted_runs) == 0

        deleted_logs = await db_operations.delete_agent_run_logs(batch_size=10)
        assert len(deleted_logs) == 1

        deleted_runs_after = await db_operations.delete_agent_runs(batch_size=10)
        assert len(deleted_runs_after) == 1

        remaining_runs = await db_client.many("SELECT run_id FROM agent_runs", [])
        assert len(remaining_runs) == 0
