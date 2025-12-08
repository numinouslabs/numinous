from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import PostAgentLogsRequestBody
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_agent_run_logs import ExportAgentRunLogs
from neurons.validator.utils.logger.logger import NuminousLogger


class TestExportAgentRunLogs:
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
            agent_name="TestAgent",
            version_number=1,
            file_path=f"/data/agents/{miner_uid}/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_miner_agents([agent])

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
    def export_agent_run_logs_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,
    ):
        api_client = NuminousClient(
            env="test", logger=MagicMock(spec=NuminousLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=NuminousLogger)

        return ExportAgentRunLogs(
            interval_seconds=180.0,
            batch_size=100,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger,
        )

    def test_init(self, export_agent_run_logs_task):
        unit = export_agent_run_logs_task

        assert isinstance(unit, ExportAgentRunLogs)
        assert unit.interval == 180.0
        assert unit.interval_seconds == 180.0
        assert unit.batch_size == 100
        assert unit.errors_count == 0

    async def test_export_log_to_backend(self, export_agent_run_logs_task: ExportAgentRunLogs):
        unit = export_agent_run_logs_task
        unit.api_client.post_agent_logs = AsyncMock(return_value=True)

        from neurons.validator.models.agent_run_logs import AgentRunLogsModel

        log = AgentRunLogsModel(
            run_id="123e4567-e89b-12d3-a456-426614174000",
            log_content="Test log content",
            exported=False,
        )

        await unit.export_log_to_backend(log)

        assert unit.api_client.post_agent_logs.call_count == 1
        call_args = unit.api_client.post_agent_logs.call_args.kwargs
        payload = call_args["body"]

        assert isinstance(payload, PostAgentLogsRequestBody)
        assert payload.run_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert payload.log_content == "Test log content"

    async def test_run_no_unexported_logs(self, export_agent_run_logs_task: ExportAgentRunLogs):
        export_agent_run_logs_task.api_client = AsyncMock(spec=NuminousClient)

        await export_agent_run_logs_task.run()

        export_agent_run_logs_task.logger.debug.assert_any_call("No unexported logs to export")
        export_agent_run_logs_task.api_client.post_agent_logs.assert_not_called()

    async def test_run_with_single_unexported_log(
        self,
        export_agent_run_logs_task: ExportAgentRunLogs,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_run_logs_task
        unit.api_client.post_agent_logs = AsyncMock(return_value=True)

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", 1, "miner_hotkey_1")

        run_id = str(uuid4())
        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="miner_hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=False,
            is_final=True,
        )
        await db_operations.upsert_agent_runs([run])
        await db_operations.insert_agent_run_log(run_id, "Log content for run")

        await unit.run()

        assert unit.api_client.post_agent_logs.call_count == 1

        result = await db_client.many(
            "SELECT exported FROM agent_run_logs WHERE run_id = ?", [run_id]
        )
        assert len(result) == 1
        assert result[0][0] == 1

    async def test_run_with_multiple_unexported_logs(
        self,
        export_agent_run_logs_task: ExportAgentRunLogs,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_run_logs_task
        unit.api_client.post_agent_logs = AsyncMock(return_value=True)

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", 1, "miner_hotkey_1")

        run_ids = [str(uuid4()) for _ in range(3)]
        for run_id in run_ids:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log for {run_id}")

        await unit.run()

        assert unit.api_client.post_agent_logs.call_count == 3

        result = await db_client.many("SELECT exported FROM agent_run_logs ORDER BY run_id")
        assert len(result) == 3
        assert all(row[0] == 1 for row in result)

    async def test_run_partial_export_failure(
        self,
        export_agent_run_logs_task: ExportAgentRunLogs,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_run_logs_task

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", 1, "miner_hotkey_1")

        run_id_success = str(uuid4())
        run_id_fail = str(uuid4())

        for run_id in [run_id_success, run_id_fail]:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log for {run_id}")

        def side_effect(body):
            if body.run_id == UUID(run_id_fail):
                raise Exception("Simulated failure")
            return True

        unit.api_client.post_agent_logs = AsyncMock(side_effect=side_effect)

        await unit.run()

        assert unit.api_client.post_agent_logs.call_count == 2
        unit.logger.warning.assert_called_once()

        result_success = await db_client.one(
            "SELECT exported FROM agent_run_logs WHERE run_id = ?", [run_id_success]
        )
        assert result_success[0] == 1

        result_fail = await db_client.one(
            "SELECT exported FROM agent_run_logs WHERE run_id = ?", [run_id_fail]
        )
        assert result_fail[0] == 0

    async def test_run_all_exports_fail(
        self,
        export_agent_run_logs_task: ExportAgentRunLogs,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_run_logs_task
        unit.api_client.post_agent_logs = AsyncMock(side_effect=Exception("All fail"))

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", 1, "miner_hotkey_1")

        run_ids = [str(uuid4()) for _ in range(2)]
        for run_id in run_ids:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log for {run_id}")

        await unit.run()

        assert unit.api_client.post_agent_logs.call_count == 2
        assert unit.logger.warning.call_count == 2

        result = await db_client.many("SELECT exported FROM agent_run_logs ORDER BY run_id")
        assert len(result) == 2
        assert all(row[0] == 0 for row in result)

    async def test_run_respects_batch_size(
        self,
        export_agent_run_logs_task: ExportAgentRunLogs,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_run_logs_task
        unit.batch_size = 2
        unit.api_client.post_agent_logs = AsyncMock(return_value=True)

        await self._create_event(db_operations, "event_1")
        await self._create_miner_agent(db_operations, "agent_v1", 1, "miner_hotkey_1")

        run_ids = [str(uuid4()) for _ in range(5)]
        for run_id in run_ids:
            run = AgentRunsModel(
                run_id=run_id,
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            )
            await db_operations.upsert_agent_runs([run])
            await db_operations.insert_agent_run_log(run_id, f"Log for {run_id}")

        await unit.run()

        assert unit.api_client.post_agent_logs.call_count == 2

        exported_count = await db_client.one(
            "SELECT COUNT(*) FROM agent_run_logs WHERE exported = 1", []
        )
        assert exported_count[0] == 2

        unexported_count = await db_client.one(
            "SELECT COUNT(*) FROM agent_run_logs WHERE exported = 0", []
        )
        assert unexported_count[0] == 3
