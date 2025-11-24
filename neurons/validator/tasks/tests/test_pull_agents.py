import base64
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.numinous_client import GetAgentsResponse, MinerAgentWithCode
from neurons.validator.tasks.pull_agents import PullAgents
from neurons.validator.utils.logger.logger import NuminousLogger


class TestPullAgentsInit:
    def test_valid_initialization(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        task = PullAgents(
            interval_seconds=300.0,
            api_client=api_client,
            db_operations=db_operations,
            agents_base_dir=base_dir,
            page_size=50,
            logger=logger,
        )

        assert task.interval == 300.0
        assert task.api_client == api_client
        assert task.db_operations == db_operations
        assert task.agents_base_dir == base_dir
        assert task.page_size == 50
        assert task.logger == logger

    def test_invalid_interval_negative(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            PullAgents(
                interval_seconds=-1.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

    def test_invalid_interval_zero(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            PullAgents(
                interval_seconds=0.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

    def test_invalid_interval_not_float(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            PullAgents(
                interval_seconds=300,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

    def test_invalid_db_operations_type(self):
        logger = MagicMock(spec=NuminousLogger)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(TypeError, match="db_operations must be"):
            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations="not_db_ops",
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

    def test_invalid_agents_base_dir_type(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()

        with pytest.raises(TypeError, match="agents_base_dir must be"):
            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir="/tmp/agents",
                page_size=50,
                logger=logger,
            )

    def test_invalid_page_size_negative(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(ValueError, match="page_size must be"):
            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=-1,
                logger=logger,
            )

    def test_invalid_page_size_too_large(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(ValueError, match="page_size must be"):
            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=101,
                logger=logger,
            )

    def test_invalid_logger_type(self):
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        with pytest.raises(TypeError, match="logger must be"):
            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger="not_logger",
            )

    def test_creates_base_directory(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "agents" / "nested"

            assert not base_dir.exists()

            PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            assert base_dir.exists()


class TestPullAgentsProperties:
    def test_name_property(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        task = PullAgents(
            interval_seconds=300.0,
            api_client=api_client,
            db_operations=db_operations,
            agents_base_dir=base_dir,
            page_size=50,
            logger=logger,
        )

        assert task.name == "pull-agents"

    def test_interval_seconds_property(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()
        base_dir = Path(tempfile.mkdtemp())

        task = PullAgents(
            interval_seconds=300.0,
            api_client=api_client,
            db_operations=db_operations,
            agents_base_dir=base_dir,
            page_size=50,
            logger=logger,
        )

        assert task.interval_seconds == 300.0


class TestPullAgentsRun:
    async def test_run_no_agents(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = AsyncMock()

        api_client.get_agents = AsyncMock(return_value=GetAgentsResponse(count=0, items=[]))

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            await task.run()

            api_client.get_agents.assert_called_once_with(offset=0, limit=50)

            agents = await db_operations.get_active_agents()
            assert len(agents) == 0

    async def test_run_single_agent(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = AsyncMock()

        code = "def agent_main(): return 0.5"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")

        version_id = uuid4()
        agent_data = MinerAgentWithCode(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            created_at=datetime.now(timezone.utc),
            code=code_b64,
        )

        api_client.get_agents = AsyncMock(
            return_value=GetAgentsResponse(count=1, items=[agent_data])
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            await task.run()

            agents = await db_operations.get_active_agents()
            assert len(agents) == 1

            agent = agents[0]
            assert agent.version_id == str(version_id)
            assert agent.miner_uid == 42
            assert agent.miner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            assert agent.agent_name == "TestAgent"
            assert agent.version_number == 1

            file_path = Path(agent.file_path)
            assert file_path.exists()
            assert file_path.read_text() == code

    async def test_run_pagination_multiple_pages(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = AsyncMock()

        code = "def agent_main(): return 0.5"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")

        page1_agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=i,
                miner_hotkey=f"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK{i:04d}",
                agent_name=f"Agent{i}",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code=code_b64,
            )
            for i in range(2)
        ]

        page2_agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=i,
                miner_hotkey=f"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK{i:04d}",
                agent_name=f"Agent{i}",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code=code_b64,
            )
            for i in range(2, 4)
        ]

        api_client.get_agents = AsyncMock(
            side_effect=[
                GetAgentsResponse(count=4, items=page1_agents),
                GetAgentsResponse(count=4, items=page2_agents),
                GetAgentsResponse(count=4, items=[]),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=2,
                logger=logger,
            )

            await task.run()

            assert api_client.get_agents.call_count == 3
            api_client.get_agents.assert_any_call(offset=0, limit=2)
            api_client.get_agents.assert_any_call(offset=2, limit=2)
            api_client.get_agents.assert_any_call(offset=4, limit=2)

            agents = await db_operations.get_active_agents()
            assert len(agents) == 4

    async def test_run_partial_last_page(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = AsyncMock()

        code = "def agent_main(): return 0.5"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")

        page1_agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=i,
                miner_hotkey=f"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK{i:04d}",
                agent_name=f"Agent{i}",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code=code_b64,
            )
            for i in range(2)
        ]

        page2_agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=2,
                miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK0002",
                agent_name="Agent2",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code=code_b64,
            )
        ]

        api_client.get_agents = AsyncMock(
            side_effect=[
                GetAgentsResponse(count=3, items=page1_agents),
                GetAgentsResponse(count=3, items=page2_agents),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=2,
                logger=logger,
            )

            await task.run()

            agents = await db_operations.get_active_agents()
            assert len(agents) == 3

    async def test_run_single_agent_fails_continues(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = DatabaseOperations(db_client=db_client, logger=logger)
        api_client = AsyncMock()

        code = "def agent_main(): return 0.5"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")

        agents_data = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=1,
                miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK0001",
                agent_name="Agent1",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code="INVALID_BASE64!!!",
            ),
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_uid=2,
                miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGK0002",
                agent_name="Agent2",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code=code_b64,
            ),
        ]

        api_client.get_agents = AsyncMock(
            return_value=GetAgentsResponse(count=2, items=agents_data)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            await task.run()

            agents = await db_operations.get_active_agents()
            assert len(agents) == 1
            assert agents[0].miner_uid == 2

            assert logger.error.called


class TestPullAgentsProcessAgent:
    async def test_process_agent_valid(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()

        code = "def agent_main(): return 0.5"
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")

        version_id = uuid4()
        agent_data = MinerAgentWithCode(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            created_at=datetime.now(timezone.utc),
            code=code_b64,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            agent_model = await task.process_agent(agent_data)

            assert agent_model.version_id == str(version_id)
            assert agent_model.miner_uid == 42
            assert agent_model.miner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            assert agent_model.agent_name == "TestAgent"
            assert agent_model.version_number == 1
            assert agent_model.pulled_at is not None

            file_path = Path(agent_model.file_path)
            assert file_path.exists()
            assert file_path.read_text() == code

    async def test_process_agent_invalid_base64(self):
        logger = MagicMock(spec=NuminousLogger)
        db_operations = MagicMock(spec=DatabaseOperations)
        api_client = AsyncMock()

        version_id = uuid4()
        agent_data = MinerAgentWithCode(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            created_at=datetime.now(timezone.utc),
            code="INVALID_BASE64!!!",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            task = PullAgents(
                interval_seconds=300.0,
                api_client=api_client,
                db_operations=db_operations,
                agents_base_dir=base_dir,
                page_size=50,
                logger=logger,
            )

            with pytest.raises(ValueError, match="Failed to decode base64"):
                await task.process_agent(agent_data)
