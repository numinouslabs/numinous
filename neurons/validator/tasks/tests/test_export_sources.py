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
from neurons.validator.models.numinous_client import MinerSourceSubmission, PostSourcesRequestBody
from neurons.validator.models.sources import (
    ImpactBucket,
    PersistenceBucket,
    SourceDirection,
    SourceItem,
    SourcesForExport,
)
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.export_sources import ExportSources
from neurons.validator.utils.logger.logger import NuminousLogger


def _build_source(url: str = "https://example.com") -> SourceItem:
    return SourceItem(
        url=url,
        source_type="news",
        direction=SourceDirection.UP,
        source_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        impact_bucket=ImpactBucket.HIGH,
        persistence_bucket=PersistenceBucket.MEDIUM,
        reasoning="supports outcome",
    )


class TestExportSources:
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
    def export_sources_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,
    ):
        api_client = NuminousClient(
            env="test", logger=MagicMock(spec=NuminousLogger), bt_wallet=bt_wallet
        )
        logger = MagicMock(spec=NuminousLogger)

        return ExportSources(
            interval_seconds=300.0,
            batch_size=500,
            db_operations=db_operations,
            api_client=api_client,
            logger=logger,
            validator_uid=5,
            validator_hotkey=bt_wallet.hotkey.ss58_address,
        )

    def test_init(self, export_sources_task: ExportSources):
        unit = export_sources_task

        assert isinstance(unit, ExportSources)
        assert unit.interval == 300.0
        assert unit.interval_seconds == 300.0
        assert unit.batch_size == 500
        assert unit.errors_count == 0
        assert unit.validator_uid == 5
        assert unit.validator_hotkey == "validator_hotkey_test"

    def test_prepare_payload(self, export_sources_task: ExportSources):
        entries = [
            SourcesForExport(
                run_id="123e4567-e89b-12d3-a456-426614174000",
                sources=[_build_source("https://a.com"), _build_source("https://b.com")],
                created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
                event_id="event_1",
                miner_uid=10,
                miner_hotkey="miner_hotkey_1",
                track="MAIN",
            ),
            SourcesForExport(
                run_id="223e4567-e89b-12d3-a456-426614174001",
                sources=[_build_source("https://c.com")],
                created_at=datetime(2024, 6, 15, 12, 5, 0, tzinfo=timezone.utc),
                event_id="event_2",
                miner_uid=20,
                miner_hotkey="miner_hotkey_2",
                track="MAIN",
            ),
        ]

        payload = export_sources_task.prepare_payload(entries)

        assert isinstance(payload, PostSourcesRequestBody)
        assert len(payload.submissions) == 2

        first = payload.submissions[0]
        assert isinstance(first, MinerSourceSubmission)
        assert first.event_id == "event_1"
        assert first.miner_uid == 10
        assert first.miner_hotkey == "miner_hotkey_1"
        assert first.track == "MAIN"
        assert first.validator_uid == 5
        assert first.validator_hotkey == "validator_hotkey_test"
        assert first.run_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert first.submitted_at == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert len(first.sources) == 2
        assert first.sources[0].url == "https://a.com"
        assert first.sources[1].url == "https://b.com"

        second = payload.submissions[1]
        assert second.run_id == UUID("223e4567-e89b-12d3-a456-426614174001")
        assert len(second.sources) == 1
        assert second.sources[0].url == "https://c.com"

    async def test_run_no_unexported_sources(self, export_sources_task: ExportSources):
        export_sources_task.api_client = AsyncMock(spec=NuminousClient)

        await export_sources_task.run()

        export_sources_task.logger.debug.assert_any_call("No unexported sources to export")
        export_sources_task.api_client.post_sources.assert_not_called()

    async def test_run_with_unexported_sources(
        self,
        export_sources_task: ExportSources,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_sources_task
        unit.api_client.post_sources = AsyncMock(return_value=None)

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

        await db_operations.insert_sources(
            "b23e4567-e89b-12d3-a456-42661417400a",
            [_build_source("https://a.com"), _build_source("https://b.com")],
        )
        await db_operations.insert_sources(
            "d23e4567-e89b-12d3-a456-42661417400c",
            [_build_source("https://c.com")],
        )

        await unit.run()

        unit.api_client.post_sources.assert_called_once()
        call_args = unit.api_client.post_sources.call_args.kwargs
        payload = call_args["body"]

        assert len(payload.submissions) == 2
        assert payload.submissions[0].run_id == UUID("b23e4567-e89b-12d3-a456-42661417400a")
        assert len(payload.submissions[0].sources) == 2
        assert payload.submissions[0].sources[0].url == "https://a.com"
        assert payload.submissions[0].event_id == "event_1"
        assert payload.submissions[0].miner_uid == 10

        assert payload.submissions[1].run_id == UUID("d23e4567-e89b-12d3-a456-42661417400c")
        assert len(payload.submissions[1].sources) == 1
        assert payload.submissions[1].sources[0].url == "https://c.com"

        result = await db_client.many("SELECT exported FROM sources ORDER BY run_id")
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 1

    async def test_run_export_exception(
        self,
        export_sources_task: ExportSources,
        db_operations: DatabaseOperations,
        db_client: DatabaseClient,
    ):
        unit = export_sources_task
        unit.api_client.post_sources = AsyncMock(side_effect=Exception("Simulated failure"))

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

        await db_operations.insert_sources(
            "f23e4567-e89b-12d3-a456-42661417400e", [_build_source()]
        )

        await unit.run()

        unit.logger.exception.assert_called_with("Failed to export sources to backend")

        result = await db_client.many("SELECT exported FROM sources")
        assert len(result) == 1
        assert result[0][0] == 0
