from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import (
    BatchUpdateAgentRunsRequest,
    UpdateAgentRunRequest,
)
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_agent_runs import ExportAgentRuns
from neurons.validator.utils.logger.logger import NuminousLogger


class TestExportAgentRuns:
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
        self, db_operations: DatabaseOperations, version_id: str, miner_uid: int, miner_hotkey: str
    ) -> None:
        """Helper to create a miner agent for FK constraint"""
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
    def export_agent_runs_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,
    ):
        api_client = NuminousClient(
            env="test", logger=MagicMock(spec=NuminousLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=NuminousLogger)

        return ExportAgentRuns(
            interval_seconds=180.0,
            batch_size=1000,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger,
            validator_uid=5,
            validator_hotkey=bt_wallet.hotkey.ss58_address,
        )

    def test_init(self, export_agent_runs_task):
        unit = export_agent_runs_task

        assert isinstance(unit, ExportAgentRuns)
        assert unit.interval == 180.0
        assert unit.interval_seconds == 180.0
        assert unit.batch_size == 1000
        assert unit.errors_count == 0
        assert unit.validator_uid == 5
        assert unit.validator_hotkey == "validator_hotkey_test"

    def test_prepare_runs_payload_single_run(self, export_agent_runs_task: ExportAgentRuns):
        db_runs = [
            AgentRunsModel(
                run_id="123e4567-e89b-12d3-a456-426614174000",
                unique_event_id="event_123",
                agent_version_id="223e4567-e89b-12d3-a456-426614174001",
                miner_uid=10,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            )
        ]

        payload = export_agent_runs_task.prepare_runs_payload(db_runs)

        assert isinstance(payload, BatchUpdateAgentRunsRequest)
        assert len(payload.runs) == 1

        run = payload.runs[0]
        assert isinstance(run, UpdateAgentRunRequest)
        assert run.run_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert run.status == "SUCCESS"
        assert run.is_final is True

    def test_prepare_runs_payload_multiple_runs(self, export_agent_runs_task: ExportAgentRuns):
        db_runs = [
            AgentRunsModel(
                run_id="323e4567-e89b-12d3-a456-426614174002",
                unique_event_id="event_1",
                agent_version_id="423e4567-e89b-12d3-a456-426614174003",
                miner_uid=1,
                miner_hotkey="miner_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            ),
            AgentRunsModel(
                run_id="523e4567-e89b-12d3-a456-426614174004",
                unique_event_id="event_2",
                agent_version_id="623e4567-e89b-12d3-a456-426614174005",
                miner_uid=2,
                miner_hotkey="miner_2",
                status=AgentRunStatus.SANDBOX_TIMEOUT,
                exported=False,
                is_final=False,
            ),
            AgentRunsModel(
                run_id="723e4567-e89b-12d3-a456-426614174006",
                unique_event_id="event_3",
                agent_version_id="823e4567-e89b-12d3-a456-426614174007",
                miner_uid=3,
                miner_hotkey="miner_3",
                status=AgentRunStatus.INTERNAL_AGENT_ERROR,
                exported=False,
                is_final=True,
            ),
        ]

        payload = export_agent_runs_task.prepare_runs_payload(db_runs)

        assert len(payload.runs) == 3
        assert payload.runs[0].run_id == UUID("323e4567-e89b-12d3-a456-426614174002")
        assert payload.runs[0].status == "SUCCESS"
        assert payload.runs[0].is_final is True
        assert payload.runs[1].run_id == UUID("523e4567-e89b-12d3-a456-426614174004")
        assert payload.runs[1].status == "SANDBOX_TIMEOUT"
        assert payload.runs[1].is_final is False
        assert payload.runs[2].run_id == UUID("723e4567-e89b-12d3-a456-426614174006")
        assert payload.runs[2].status == "INTERNAL_AGENT_ERROR"
        assert payload.runs[2].is_final is True

    async def test_export_runs_to_backend(self, export_agent_runs_task: ExportAgentRuns):
        unit = export_agent_runs_task
        unit.api_client.put_agent_runs = AsyncMock(return_value=None)

        dummy_payload = BatchUpdateAgentRunsRequest(
            runs=[
                UpdateAgentRunRequest(
                    run_id=UUID("923e4567-e89b-12d3-a456-426614174008"),
                    status="SUCCESS",
                    is_final=True,
                )
            ]
        )

        await export_agent_runs_task.export_runs_to_backend(dummy_payload)
        export_agent_runs_task.logger.debug.assert_called_with(
            "Exported runs to backend",
            extra={"n_runs": 1},
        )

        assert unit.api_client.put_agent_runs.call_count == 1
        assert unit.api_client.put_agent_runs.call_args.kwargs["body"] == dummy_payload

    async def test_run_no_unexported_runs(self, export_agent_runs_task: ExportAgentRuns):
        export_agent_runs_task.api_client = AsyncMock(spec=NuminousClient)

        await export_agent_runs_task.run()

        export_agent_runs_task.logger.debug.assert_any_call("No unexported runs to export")
        export_agent_runs_task.api_client.put_agent_runs.assert_not_called()

    async def test_run_with_unexported_runs(
        self,
        export_agent_runs_task: ExportAgentRuns,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_runs_task
        unit.api_client.put_agent_runs = AsyncMock(return_value=None)

        await self._create_event(db_operations, "event_1")
        await self._create_event(db_operations, "event_2")
        await self._create_miner_agent(
            db_operations, "c23e4567-e89b-12d3-a456-42661417400b", 10, "miner_hotkey_1"
        )
        await self._create_miner_agent(
            db_operations, "e23e4567-e89b-12d3-a456-42661417400d", 20, "miner_hotkey_2"
        )

        runs = [
            AgentRunsModel(
                run_id="b23e4567-e89b-12d3-a456-42661417400a",
                unique_event_id="event_1",
                agent_version_id="c23e4567-e89b-12d3-a456-42661417400b",
                miner_uid=10,
                miner_hotkey="miner_hotkey_1",
                status=AgentRunStatus.SUCCESS,
                exported=False,
                is_final=True,
            ),
            AgentRunsModel(
                run_id="d23e4567-e89b-12d3-a456-42661417400c",
                unique_event_id="event_2",
                agent_version_id="e23e4567-e89b-12d3-a456-42661417400d",
                miner_uid=20,
                miner_hotkey="miner_hotkey_2",
                status=AgentRunStatus.SANDBOX_TIMEOUT,
                exported=False,
                is_final=False,
            ),
        ]

        await db_operations.upsert_agent_runs(runs)

        await unit.run()

        unit.api_client.put_agent_runs.assert_called_once()
        call_args = unit.api_client.put_agent_runs.call_args.kwargs
        payload = call_args["body"]

        assert len(payload.runs) == 2
        assert payload.runs[0].run_id == UUID("b23e4567-e89b-12d3-a456-42661417400a")
        assert payload.runs[0].status == "SUCCESS"
        assert payload.runs[0].is_final is True
        assert payload.runs[1].run_id == UUID("d23e4567-e89b-12d3-a456-42661417400c")
        assert payload.runs[1].status == "SANDBOX_TIMEOUT"
        assert payload.runs[1].is_final is False

        result = await db_client.many("SELECT exported FROM agent_runs ORDER BY run_id")
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 1

    async def test_run_export_exception(
        self,
        export_agent_runs_task: ExportAgentRuns,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_agent_runs_task
        unit.api_client.put_agent_runs = AsyncMock(side_effect=Exception("Simulated failure"))

        await self._create_event(db_operations, "event_error")
        await self._create_miner_agent(
            db_operations, "023e4567-e89b-12d3-a456-42661417400f", 30, "miner_hotkey_3"
        )

        run = AgentRunsModel(
            run_id="f23e4567-e89b-12d3-a456-42661417400e",
            unique_event_id="event_error",
            agent_version_id="023e4567-e89b-12d3-a456-42661417400f",
            miner_uid=30,
            miner_hotkey="miner_hotkey_3",
            status=AgentRunStatus.SUCCESS,
            exported=False,
            is_final=True,
        )

        await db_operations.upsert_agent_runs([run])

        await unit.run()

        unit.logger.exception.assert_called_with("Failed to export runs to backend")

        result = await db_client.many("SELECT exported FROM agent_runs")
        assert len(result) == 1
        assert result[0][0] == 0
