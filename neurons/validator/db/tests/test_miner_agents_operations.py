import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.utils.logger.logger import NuminousLogger


class TestMinerAgentsOperations(TestDbOperationsBase):
    @pytest.fixture
    async def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)

        db_operations = DatabaseOperations(db_client=db_client, logger=logger)

        return db_operations

    async def test_get_last_agent_pulled_at_no_agents(self, db_operations: DatabaseOperations):
        result = await db_operations.get_last_agent_pulled_at()

        assert result is None

    async def test_get_last_agent_pulled_at(self, db_operations: DatabaseOperations):
        pulled_at_1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        pulled_at_2 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        pulled_at_3 = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        agents = [
            MinerAgentsModel(
                version_id=str(uuid4()),
                miner_uid=1,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=1,
                file_path="/data/agents/1/agent.py",
                pulled_at=pulled_at_1,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id=str(uuid4()),
                miner_uid=2,
                miner_hotkey="hotkey2",
                agent_name="Agent2",
                version_number=1,
                file_path="/data/agents/2/agent.py",
                pulled_at=pulled_at_2,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id=str(uuid4()),
                miner_uid=3,
                miner_hotkey="hotkey3",
                agent_name="Agent3",
                version_number=1,
                file_path="/data/agents/3/agent.py",
                pulled_at=pulled_at_3,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_operations.get_last_agent_pulled_at()

        assert result == "2024-01-01 11:00:00+00:00"

    async def test_upsert_miner_agents_empty_list(self, db_operations: DatabaseOperations):
        await db_operations.upsert_miner_agents([])

    async def test_upsert_miner_agents_insert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        version_id_1 = str(uuid4())
        version_id_2 = str(uuid4())

        agents = [
            MinerAgentsModel(
                version_id=version_id_1,
                miner_uid=42,
                miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                agent_name="TestAgent1",
                version_number=1,
                file_path="/data/agents/42/test1.py",
                pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id=version_id_2,
                miner_uid=43,
                miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                agent_name="TestAgent2",
                version_number=1,
                file_path="/data/agents/43/test2.py",
                pulled_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_client.many(
            """
                SELECT
                    version_id,
                    miner_uid,
                    miner_hotkey,
                    agent_name,
                    version_number,
                    file_path
                FROM
                    miner_agents
                ORDER BY
                    miner_uid
            """
        )

        assert len(result) == 2
        assert result[0][0] == version_id_1
        assert result[0][1] == 42
        assert result[0][2] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert result[0][3] == "TestAgent1"
        assert result[0][4] == 1
        assert result[0][5] == "/data/agents/42/test1.py"

        assert result[1][0] == version_id_2
        assert result[1][1] == 43
        assert result[1][3] == "TestAgent2"

    async def test_upsert_miner_agents_update(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        version_id = str(uuid4())

        agent_v1 = MinerAgentsModel(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            file_path="/data/agents/42/old_path.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )

        await db_operations.upsert_miner_agents([agent_v1])

        result = await db_client.one(
            """
                SELECT file_path, pulled_at FROM miner_agents WHERE version_id = ?
            """,
            [version_id],
        )

        original_file_path = result[0]
        original_pulled_at = result[1]

        assert original_file_path == "/data/agents/42/old_path.py"

        await asyncio.sleep(1)

        agent_v2 = MinerAgentsModel(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            file_path="/data/agents/42/new_path.py",
            pulled_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )

        await db_operations.upsert_miner_agents([agent_v2])

        result = await db_client.many(
            """
                SELECT version_id, file_path, pulled_at FROM miner_agents
            """
        )

        assert len(result) == 1
        assert result[0][0] == version_id
        assert result[0][1] == "/data/agents/42/new_path.py"
        assert result[0][2] != original_pulled_at

    async def test_get_agent_by_version_not_found(self, db_operations: DatabaseOperations):
        version_id = str(uuid4())

        result = await db_operations.get_agent_by_version(version_id)

        assert result is None

    async def test_get_agent_by_version_found(self, db_operations: DatabaseOperations):
        version_id = str(uuid4())

        agent = MinerAgentsModel(
            version_id=version_id,
            miner_uid=42,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            agent_name="TestAgent",
            version_number=1,
            file_path="/data/agents/42/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )

        await db_operations.upsert_miner_agents([agent])

        result = await db_operations.get_agent_by_version(version_id)

        assert result is not None
        assert isinstance(result, MinerAgentsModel)
        assert result.version_id == version_id
        assert result.miner_uid == 42
        assert result.agent_name == "TestAgent"
        assert result.file_path == "/data/agents/42/test.py"

    async def test_get_active_agents_empty(self, db_operations: DatabaseOperations):
        result = await db_operations.get_active_agents()

        assert len(result) == 0
        assert isinstance(result, list)

    async def test_get_active_agents_ordered_by_version_id(self, db_operations: DatabaseOperations):
        pulled_at_1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        pulled_at_2 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        pulled_at_3 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        agents = [
            MinerAgentsModel(
                version_id="c-agent",
                miner_uid=1,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=1,
                file_path="/data/agents/1/agent.py",
                pulled_at=pulled_at_1,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="a-agent",
                miner_uid=2,
                miner_hotkey="hotkey2",
                agent_name="Agent2",
                version_number=1,
                file_path="/data/agents/2/agent.py",
                pulled_at=pulled_at_2,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="b-agent",
                miner_uid=3,
                miner_hotkey="hotkey3",
                agent_name="Agent3",
                version_number=1,
                file_path="/data/agents/3/agent.py",
                pulled_at=pulled_at_3,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_operations.get_active_agents()

        assert len(result) == 3
        assert isinstance(result[0], MinerAgentsModel)
        assert result[0].version_id == "a-agent"
        assert result[1].version_id == "b-agent"
        assert result[2].version_id == "c-agent"

    async def test_get_active_agents_with_limit(self, db_operations: DatabaseOperations):
        agents = [
            MinerAgentsModel(
                version_id=f"{i:02d}-agent",
                miner_uid=i,
                miner_hotkey=f"hotkey{i}",
                agent_name=f"Agent{i}",
                version_number=1,
                file_path=f"/data/agents/{i}/agent.py",
                pulled_at=datetime(2024, 1, 1, 10, i, 0, tzinfo=timezone.utc),
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(10)
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_operations.get_active_agents(limit=3)

        assert len(result) == 3
        assert result[0].miner_uid == 0
        assert result[1].miner_uid == 1
        assert result[2].miner_uid == 2

    async def test_get_active_agents_returns_latest_version_per_miner(
        self, db_operations: DatabaseOperations
    ):
        pulled_at_1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        pulled_at_2 = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        pulled_at_3 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        pulled_at_4 = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        agents = [
            MinerAgentsModel(
                version_id="1-miner1-v1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=1,
                file_path="/data/agents/1/v1/agent.py",
                pulled_at=pulled_at_1,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="1-miner1-v3",
                miner_uid=1,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=3,
                file_path="/data/agents/1/v3/agent.py",
                pulled_at=pulled_at_3,
                created_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="1-miner1-v2",
                miner_uid=1,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=2,
                file_path="/data/agents/1/v2/agent.py",
                pulled_at=pulled_at_2,
                created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="2-miner2-v1",
                miner_uid=2,
                miner_hotkey="hotkey2",
                agent_name="Agent2",
                version_number=1,
                file_path="/data/agents/2/v1/agent.py",
                pulled_at=pulled_at_1,
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="2-miner2-v2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                agent_name="Agent2",
                version_number=2,
                file_path="/data/agents/2/v2/agent.py",
                pulled_at=pulled_at_4,
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id="3-miner3-v1",
                miner_uid=3,
                miner_hotkey="hotkey3",
                agent_name="Agent3",
                version_number=1,
                file_path="/data/agents/3/v1/agent.py",
                pulled_at=pulled_at_2,
                created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_operations.get_active_agents()

        assert len(result) == 3

        miner1_agent = next((a for a in result if a.miner_uid == 1), None)
        assert miner1_agent is not None
        assert miner1_agent.version_number == 3
        assert miner1_agent.file_path == "/data/agents/1/v3/agent.py"

        miner2_agent = next((a for a in result if a.miner_uid == 2), None)
        assert miner2_agent is not None
        assert miner2_agent.version_number == 2
        assert miner2_agent.file_path == "/data/agents/2/v2/agent.py"

        miner3_agent = next((a for a in result if a.miner_uid == 3), None)
        assert miner3_agent is not None
        assert miner3_agent.version_number == 1
        assert miner3_agent.file_path == "/data/agents/3/v1/agent.py"

        assert result[0].version_id == "1-miner1-v3"
        assert result[1].version_id == "2-miner2-v2"
        assert result[2].version_id == "3-miner3-v1"

    async def test_upsert_miner_agents_unique_constraint(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        version_id_1 = str(uuid4())
        version_id_2 = str(uuid4())

        agents = [
            MinerAgentsModel(
                version_id=version_id_1,
                miner_uid=42,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=1,
                file_path="/data/agents/42/v1.py",
                pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
            MinerAgentsModel(
                version_id=version_id_2,
                miner_uid=42,
                miner_hotkey="hotkey1",
                agent_name="Agent1",
                version_number=2,
                file_path="/data/agents/42/v2.py",
                pulled_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        await db_operations.upsert_miner_agents(agents)

        result = await db_client.many(
            """
                SELECT version_id, version_number FROM miner_agents ORDER BY version_number
            """
        )

        assert len(result) == 2
        assert result[0][0] == version_id_1
        assert result[0][1] == 1
        assert result[1][0] == version_id_2
        assert result[1][1] == 2
