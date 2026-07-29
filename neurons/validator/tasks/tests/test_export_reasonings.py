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
    MinerReasoningSubmission,
    PostReasoningRequestBody,
)
from neurons.validator.models.reasoning import ReasoningForExport
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_reasonings import ExportReasonings
from neurons.validator.utils.logger.logger import NuminousLogger


class TestExportReasonings:
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
        status: AgentRunStatus = AgentRunStatus.SUCCESS,
    ) -> None:
        run = AgentRunsModel(
            interval_start_minutes=1000,
            run_id=run_id,
            unique_event_id=unique_event_id,
            agent_version_id=agent_version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            status=status,
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
    def export_reasonings_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,
    ):
        api_client = NuminousClient(
            env="test", logger=MagicMock(spec=NuminousLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=NuminousLogger)

        return ExportReasonings(
            interval_seconds=300.0,
            batch_size=500,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger,
            validator_uid=5,
            validator_hotkey=bt_wallet.hotkey.ss58_address,
        )

    def test_init(self, export_reasonings_task):
        unit = export_reasonings_task

        assert isinstance(unit, ExportReasonings)
        assert unit.interval == 300.0
        assert unit.interval_seconds == 300.0
        assert unit.batch_size == 500
        assert unit.errors_count == 0
        assert unit.validator_uid == 5
        assert unit.validator_hotkey == "validator_hotkey_test"

    def test_prepare_payload(self, export_reasonings_task: ExportReasonings):
        reasonings = [
            ReasoningForExport(
                run_id="123e4567-e89b-12d3-a456-426614174000",
                reasoning="The market will go up because...",
                created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
                event_id="event_1",
                miner_uid=10,
                miner_hotkey="miner_hotkey_1",
                track="MAIN",
            ),
            ReasoningForExport(
                run_id="223e4567-e89b-12d3-a456-426614174001",
                reasoning="[NO_REASONING - SANDBOX_TIMEOUT]",
                created_at=datetime(2024, 6, 15, 12, 5, 0, tzinfo=timezone.utc),
                event_id="event_2",
                miner_uid=20,
                miner_hotkey="miner_hotkey_2",
                track="MAIN",
            ),
        ]

        payload = export_reasonings_task.prepare_payload(reasonings)

        assert isinstance(payload, PostReasoningRequestBody)
        assert len(payload.reasonings) == 2

        first = payload.reasonings[0]
        assert isinstance(first, MinerReasoningSubmission)
        assert first.event_id == "event_1"
        assert first.miner_uid == 10
        assert first.miner_hotkey == "miner_hotkey_1"
        assert first.track == "MAIN"
        assert first.validator_uid == 5
        assert first.validator_hotkey == "validator_hotkey_test"
        assert first.run_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert first.reasoning == "The market will go up because..."
        assert first.submitted_at == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        second = payload.reasonings[1]
        assert second.reasoning == "[NO_REASONING - SANDBOX_TIMEOUT]"

    async def test_run_no_unexported_reasonings(self, export_reasonings_task: ExportReasonings):
        export_reasonings_task.api_client = AsyncMock(spec=NuminousClient)

        await export_reasonings_task.run()

        export_reasonings_task.logger.debug.assert_any_call("No unexported reasonings to export")
        export_reasonings_task.api_client.post_reasonings.assert_not_called()

    async def test_run_with_unexported_reasonings(
        self,
        export_reasonings_task: ExportReasonings,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_reasonings_task
        unit.api_client.post_reasonings = AsyncMock(return_value=None)

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

        await db_operations.insert_reasoning(
            "b23e4567-e89b-12d3-a456-42661417400a", "Reasoning for event 1"
        )
        await db_operations.insert_reasoning(
            "d23e4567-e89b-12d3-a456-42661417400c", "[NO_REASONING - SUCCESS]"
        )

        await unit.run()

        unit.api_client.post_reasonings.assert_called_once()
        call_args = unit.api_client.post_reasonings.call_args.kwargs
        payload = call_args["body"]

        assert len(payload.reasonings) == 2
        assert payload.reasonings[0].run_id == UUID("b23e4567-e89b-12d3-a456-42661417400a")
        assert payload.reasonings[0].reasoning == "Reasoning for event 1"
        assert payload.reasonings[0].event_id == "event_1"
        assert payload.reasonings[0].miner_uid == 10
        assert payload.reasonings[0].miner_hotkey == "miner_hotkey_1"
        assert payload.reasonings[0].track == "MAIN"
        assert payload.reasonings[0].validator_uid == 5
        assert payload.reasonings[0].validator_hotkey == "validator_hotkey_test"

        assert payload.reasonings[1].run_id == UUID("d23e4567-e89b-12d3-a456-42661417400c")
        assert payload.reasonings[1].reasoning == "[NO_REASONING - SUCCESS]"

        result = await db_client.many("SELECT exported FROM reasoning ORDER BY run_id")
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 1

    async def test_run_export_exception(
        self,
        export_reasonings_task: ExportReasonings,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_reasonings_task
        unit.api_client.post_reasonings = AsyncMock(side_effect=Exception("Simulated failure"))

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

        await db_operations.insert_reasoning(
            "f23e4567-e89b-12d3-a456-42661417400e", "Some reasoning"
        )

        await unit.run()

        unit.logger.exception.assert_called_with("Failed to export reasonings to backend")

        result = await db_client.many("SELECT exported FROM reasoning")
        assert len(result) == 1
        assert result[0][0] == 0
