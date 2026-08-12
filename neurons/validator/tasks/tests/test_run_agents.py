import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import (
    CreateAgentRunRequest,
    CreateAgentRunResponse,
    MemoryEntry,
    MemoryPullResponse,
)
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.sandbox import SandboxManager
from neurons.validator.sandbox.models import SandboxErrorType
from neurons.validator.tasks.run_agents import MAX_TIMEOUT_RETRIES, RunAgents
from neurons.validator.utils.common.interval import (
    get_interval_iso_datetime,
    get_interval_start_minutes,
)
from neurons.validator.utils.logger.logger import NuminousLogger


@pytest.fixture
def mock_logger():
    return MagicMock(spec=NuminousLogger)


@pytest.fixture
def mock_db_operations():
    return AsyncMock(spec=DatabaseOperations)


@pytest.fixture
def mock_sandbox_manager():
    return MagicMock(spec=SandboxManager)


@pytest.fixture
def mock_api_client():
    client = MagicMock(spec=NuminousClient)
    client.post_agent_logs = AsyncMock()
    client.create_agent_run = AsyncMock(
        return_value=CreateAgentRunResponse(run_id=UUID("123e4567-e89b-12d3-a456-426614174000"))
    )
    return client


@pytest.fixture
def mock_subtensor_cm():
    metagraph = MagicMock()
    metagraph.block = 12345
    metagraph.num_uids = 0
    metagraph.neurons = []

    chain_client = AsyncMock()
    chain_client.subnets.metagraph = AsyncMock(return_value=metagraph)

    subtensor = MagicMock()
    subtensor.return_value.__aenter__ = AsyncMock(return_value=chain_client)
    subtensor.return_value.__aexit__ = AsyncMock(return_value=False)
    subtensor.chain_client = chain_client

    return subtensor


@pytest.fixture
def sample_event_tuple():
    return EventsModel(
        unique_event_id="event_123",
        event_id="external_event_123",
        market_type="polymarket",
        event_type="llm_generated",
        title="Will it rain?",
        description="Weather forecast unclear",
        status=EventStatus.PENDING,
        metadata="{}",
        cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_agent():
    return MinerAgentsModel(
        version_id="a23e4567-e89b-12d3-a456-426614174000",
        miner_uid=42,
        miner_hotkey="5HotKey123",
        track="MAIN",
        agent_name="test_agent",
        version_number=1,
        file_path="/tmp/test_agent.py",
        pulled_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


class TestRunAgentsInit:
    def test_valid_initialization(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )
        assert task.name == "run-agents"
        assert task.interval_seconds == 600.0
        assert task.netuid == 99
        assert task.network == "test"

    def test_invalid_interval_negative(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            RunAgents(
                interval_seconds=-1.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_interval_zero(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            RunAgents(
                interval_seconds=0.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_db_operations_type(
        self, mock_sandbox_manager, mock_subtensor_cm, mock_api_client, mock_logger
    ):
        with pytest.raises(TypeError, match="db_operations must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations="not_db_ops",
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_sandbox_manager_type(
        self, mock_db_operations, mock_subtensor_cm, mock_api_client, mock_logger
    ):
        with pytest.raises(TypeError, match="sandbox_manager must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager="not_sandbox",
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_netuid_type(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="netuid must be a non-negative integer."):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=-1,
                network="",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_network(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="network must be a non-empty string."):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_api_client_type(
        self, mock_db_operations, mock_sandbox_manager, mock_subtensor_cm, mock_logger
    ):
        with pytest.raises(TypeError, match="api_client must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client="not_api_client",
                logger=mock_logger,
            )

    def test_invalid_logger_type(
        self, mock_db_operations, mock_sandbox_manager, mock_subtensor_cm, mock_api_client
    ):
        with pytest.raises(TypeError, match="logger must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger="not_logger",
            )

    def test_invalid_max_concurrent_negative(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="max_concurrent_sandboxes must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                max_concurrent_sandboxes=-1,
            )

    def test_invalid_max_concurrent_zero(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="max_concurrent_sandboxes must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                max_concurrent_sandboxes=0,
            )

    def test_invalid_timeout_negative(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="timeout_seconds must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                timeout_seconds=-1,
            )

    def test_invalid_timeout_zero(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="timeout_seconds must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                timeout_seconds=0,
            )

    def test_invalid_validator_uid_negative(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="validator_uid must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                validator_uid=-1,
            )

    def test_invalid_validator_uid_too_large(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(ValueError, match="validator_uid must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                validator_uid=257,
            )

    def test_invalid_validator_hotkey_type(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        with pytest.raises(TypeError, match="validator_hotkey must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                netuid=99,
                network="test",
                api_client=mock_api_client,
                logger=mock_logger,
                validator_hotkey=123,
            )

    def test_valid_initialization_with_validator_params(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            validator_uid=42,
            validator_hotkey="5ValidatorHotkey123",
        )
        assert task.validator_uid == 42
        assert task.validator_hotkey == "5ValidatorHotkey123"


class TestRunAgentsRun:
    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_no_events(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        # Mock datetime.now() to return hour >= 4 to pass sync_hour check
        mock_datetime.now.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
            await task.run()

        mock_subtensor_cm.assert_called_once_with("test")
        mock_subtensor_cm.return_value.__aenter__.assert_awaited_once()
        mock_subtensor_cm.return_value.__aexit__.assert_awaited_once()
        mock_subtensor_cm.chain_client.subnets.metagraph.assert_awaited_once_with(task.netuid)

        mock_logger.debug.assert_called_with("No events to predict")
        mock_db_operations.get_active_agents.assert_not_called()

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_no_agents(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        # Mock datetime.now() to return hour >= 4 to pass sync_hour check
        mock_datetime.now.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = [
            (
                "event_1",
                "external_event_1",
                "polymarket",
                "llm",
                "Some title",
                "desc",
                None,
                "{}",
            )
        ]
        mock_db_operations.get_active_agents.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
            await task.run()

        mock_subtensor_cm.assert_called_once_with("test")
        mock_subtensor_cm.return_value.__aenter__.assert_awaited_once()
        mock_subtensor_cm.return_value.__aexit__.assert_awaited_once()
        mock_subtensor_cm.chain_client.subnets.metagraph.assert_awaited_once_with(task.netuid)

        mock_logger.warning.assert_called_with("No agents available for execution")


class TestRunAgentsFiltering:
    def test_filter_agent_uid_not_in_metagraph(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        metagraph = MagicMock()
        metagraph.neurons = []

        task.metagraph = metagraph

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 0

    def test_filter_agent_hotkey_mismatch(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        metagraph = MagicMock()
        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "different_hotkey"
        metagraph.neurons = [neuron]

        task.metagraph = metagraph

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 0

    def test_keep_agent_without_served_axon(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        metagraph = MagicMock()
        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "5HotKey123"
        neuron.axon = None
        metagraph.neurons = [neuron]

        task.metagraph = metagraph

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 1
        assert result[0] == sample_agent

    def test_keep_valid_agent(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        metagraph = MagicMock()
        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "5HotKey123"
        metagraph.neurons = [neuron]

        task.metagraph = metagraph

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 1
        assert result[0] == sample_agent

    def test_mixed_filtering(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        agent1 = MinerAgentsModel(
            version_id="v1",
            miner_uid=42,
            miner_hotkey="hotkey1",
            track="MAIN",
            agent_name="agent1",
            version_number=1,
            file_path="/tmp/a1.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        agent2 = MinerAgentsModel(
            version_id="v2",
            miner_uid=99,
            miner_hotkey="hotkey2",
            track="MAIN",
            agent_name="agent2",
            version_number=1,
            file_path="/tmp/a2.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        agent3 = MinerAgentsModel(
            version_id="v3",
            miner_uid=100,
            miner_hotkey="hotkey3",
            track="MAIN",
            agent_name="agent3",
            version_number=1,
            file_path="/tmp/a3.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        metagraph = MagicMock()
        neuron1 = MagicMock()
        neuron1.uid = 42
        neuron1.hotkey = "hotkey1"
        neuron3 = MagicMock()
        neuron3.uid = 100
        neuron3.hotkey = "hotkey3"
        metagraph.neurons = [neuron1, neuron3]

        task.metagraph = metagraph

        result = task.filter_agents_by_metagraph([agent1, agent2, agent3])

        assert len(result) == 2
        assert result[0] == agent1
        assert result[1] == agent3


class TestRunAgentsTrackFiltering:
    async def test_skip_agent_when_track_not_in_event(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        main_event = EventsModel(
            unique_event_id="event_main_only",
            event_id="ext_1",
            market_type="polymarket",
            event_type="llm_generated",
            title="Main only event",
            description="desc",
            status=EventStatus.PENDING,
            metadata="{}",
            cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
            tracks='["MAIN"]',
        )

        signal_agent = MinerAgentsModel(
            version_id="v_signal",
            miner_uid=42,
            miner_hotkey="5HotKey123",
            track="SIGNAL",
            agent_name="signal_agent",
            version_number=1,
            file_path="/tmp/signal.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.execute_all([main_event], [signal_agent], interval_start_minutes=1000)

        mock_db_operations.get_latest_prediction_for_event_and_miner.assert_not_called()

    async def test_execute_agent_when_track_matches(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        event = EventsModel(
            unique_event_id="event_main",
            event_id="ext_1",
            market_type="polymarket",
            event_type="llm_generated",
            title="Main event",
            description="desc",
            status=EventStatus.PENDING,
            metadata="{}",
            cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
            tracks='["MAIN"]',
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_with_semaphore = AsyncMock()

        await task.execute_all([event], [sample_agent], interval_start_minutes=1000)

        task.execute_with_semaphore.assert_called_once()

    async def test_mixed_tracks_filters_correctly(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        main_only_event = EventsModel(
            unique_event_id="event_main_only",
            event_id="ext_1",
            market_type="polymarket",
            event_type="llm_generated",
            title="Main only",
            description="desc",
            status=EventStatus.PENDING,
            metadata="{}",
            cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
            tracks='["MAIN"]',
        )

        both_tracks_event = EventsModel(
            unique_event_id="event_both",
            event_id="ext_2",
            market_type="polymarket",
            event_type="llm_generated",
            title="Both tracks",
            description="desc",
            status=EventStatus.PENDING,
            metadata="{}",
            cutoff=datetime(2025, 12, 31, tzinfo=timezone.utc),
            tracks='["MAIN", "SIGNAL"]',
        )

        main_agent = MinerAgentsModel(
            version_id="v_main",
            miner_uid=42,
            miner_hotkey="5HotKey123",
            track="MAIN",
            agent_name="main_agent",
            version_number=1,
            file_path="/tmp/main.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        signal_agent = MinerAgentsModel(
            version_id="v_signal",
            miner_uid=43,
            miner_hotkey="5HotKey456",
            track="SIGNAL",
            agent_name="signal_agent",
            version_number=1,
            file_path="/tmp/signal.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_with_semaphore = AsyncMock()

        await task.execute_all(
            [main_only_event, both_tracks_event],
            [main_agent, signal_agent],
            interval_start_minutes=1000,
        )

        # main_agent runs on both events (2), signal_agent only on both_tracks_event (1) = 3 total
        assert task.execute_with_semaphore.call_count == 3


class TestRunAgentsParsing:
    def test_parse_event_description_with_separator(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        full_desc = "Will it rain? ==Further Information==: Weather forecast unclear"
        title, description = task.parse_event_description(full_desc)

        assert title == "Will it rain?"
        assert description == "Weather forecast unclear"

    def test_parse_event_description_without_separator(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        full_desc = "Will it rain tomorrow?"
        title, description = task.parse_event_description(full_desc)

        assert title == "Will it rain tomorrow?"
        assert description == "Will it rain tomorrow?"


class TestRunAgentsIdempotency:
    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_skip_when_prediction_exists(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.now() to return hour >= 4 to pass sync_hour check
        mock_datetime.now.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "5HotKey123"

        mock_metagraph = mock_subtensor_cm.chain_client.subnets.metagraph.return_value
        mock_metagraph.neurons = [neuron]

        # Prediction already exists in current interval
        current_interval = get_interval_start_minutes()
        existing_prediction = PredictionsModel(
            unique_event_id="event_123",
            miner_uid=42,
            miner_hotkey="5HotKey123",
            track="MAIN",
            latest_prediction=0.75,
            interval_start_minutes=current_interval,
            interval_agg_prediction=0.75,
        )
        mock_db_operations.get_latest_prediction_for_event_and_miner.return_value = (
            existing_prediction
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
            await task.run()

        mock_db_operations.get_latest_prediction_for_event_and_miner.assert_called_once()
        mock_logger.debug.assert_any_call(
            "Skipping execution - prediction exists for interval",
            extra={
                "event_id": "event_123",
                "agent_version_id": "a23e4567-e89b-12d3-a456-426614174000",
                "miner_uid": 42,
                "interval_start_minutes": current_interval,
            },
        )

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_execute_when_prediction_not_exists(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.now() to return hour >= 4 to pass sync_hour check
        mock_datetime.now.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "5HotKey123"

        mock_metagraph = mock_subtensor_cm.chain_client.subnets.metagraph.return_value
        mock_metagraph.neurons = [neuron]
        axon = MagicMock()
        axon.hotkey = "5HotKey123"
        mock_metagraph.axons = {42: axon}

        # No prediction exists
        mock_db_operations.get_latest_prediction_for_event_and_miner.return_value = None
        mock_db_operations.has_final_run = AsyncMock(return_value=False)
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_agent_for_event = AsyncMock()

        with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
            await task.run()

        mock_db_operations.get_latest_prediction_for_event_and_miner.assert_called_once()
        task.execute_agent_for_event.assert_called_once()

        call_args = task.execute_agent_for_event.call_args[1]
        assert call_args["event"] == sample_event_tuple
        assert call_args["agent"] == sample_agent

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_reruns_when_prediction_exists_in_earlier_interval(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.now() to return hour >= 4 to pass sync_hour check
        mock_datetime.now.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        neuron = MagicMock()
        neuron.uid = 42
        neuron.hotkey = "5HotKey123"

        mock_metagraph = mock_subtensor_cm.chain_client.subnets.metagraph.return_value
        mock_metagraph.neurons = [neuron]

        # Existing prediction in interval 100
        existing_prediction = PredictionsModel(
            unique_event_id="event_123",
            miner_uid=42,
            miner_hotkey="5HotKey123",
            track="MAIN",
            latest_prediction=0.75,
            interval_start_minutes=100,
            interval_agg_prediction=0.75,
            run_id="original_run_id",
            version_id="a23e4567-e89b-12d3-a456-426614174000",
        )
        mock_db_operations.get_latest_prediction_for_event_and_miner.return_value = (
            existing_prediction
        )
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.has_final_run = AsyncMock(return_value=False)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_agent_for_event = AsyncMock()

        with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
            await task.run()

        # A prediction from an earlier interval is history: the agent re-runs for
        # the current interval instead of it being replicated forward.
        task.execute_agent_for_event.assert_called_once()
        assert (
            task.execute_agent_for_event.call_args[1]["interval_start_minutes"]
            == get_interval_start_minutes()
        )
        mock_db_operations.upsert_predictions.assert_not_called()


class TestRunAgentsFileLoading:
    async def test_load_agent_file_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        tmp_path,
    ):
        agent_file = tmp_path / "test_agent.py"
        agent_file.write_text("def agent_main(): pass")
        sample_agent.file_path = str(agent_file)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        code = await task.load_agent_code(sample_agent)
        assert code == "def agent_main(): pass"

    async def test_load_agent_file_not_found(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        sample_agent.file_path = "/nonexistent/path/agent.py"

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        code = await task.load_agent_code(sample_agent)
        assert code is None
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Failed to load agent code" in call_args[0][0]

    async def test_load_agent_file_permission_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        tmp_path,
        monkeypatch,
    ):
        agent_file = tmp_path / "restricted_agent.py"
        agent_file.write_text("def agent_main(): pass")
        sample_agent.file_path = str(agent_file)

        def mock_read_text():
            raise PermissionError("Permission denied")

        monkeypatch.setattr(Path, "read_text", lambda self: mock_read_text())

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        code = await task.load_agent_code(sample_agent)
        assert code is None
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Failed to load agent code" in call_args[0][0]


class TestRunAgentsSandbox:
    async def test_run_sandbox_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        event_data = {"event_id": "event_123", "title": "Test", "description": "Test event"}
        agent_code = "def agent_main(): return {'prediction': 0.75}"

        def mock_create_sandbox(agent_code, event_data, run_id, on_finish, timeout):
            on_finish({"event_id": "event_123", "prediction": 0.75})
            return "sandbox_123"

        mock_sandbox_manager.create_sandbox = mock_create_sandbox

        result = await task.run_sandbox(agent_code, event_data, "run_123")

        assert result == {"event_id": "event_123", "prediction": 0.75}

    async def test_run_sandbox_failure(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        event_data = {"event_id": "event_123", "title": "Test", "description": "Test event"}
        agent_code = "def agent_main(): return {'prediction': 0.75}"

        def mock_create_sandbox(agent_code, event_data, run_id, on_finish, timeout):
            on_finish({"SUCCESS": False, "error": "Execution failed"})
            return "sandbox_123"

        mock_sandbox_manager.create_sandbox = mock_create_sandbox

        result = await task.run_sandbox(agent_code, event_data, "run_123")

        assert result["SUCCESS"] is False
        assert "error" in result

    async def test_run_sandbox_timeout(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )

        event_data = {"event_id": "event_123", "title": "Test", "description": "Test event"}
        agent_code = "def agent_main(): return {'prediction': 0.75}"

        def mock_create_sandbox(agent_code, event_data, run_id, on_finish, timeout):
            return "sandbox_123"

        mock_sandbox_manager.create_sandbox = mock_create_sandbox

        result = await task.run_sandbox(agent_code, event_data, "run_123")

        assert result is None
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args
        assert "timeout" in str(call_args).lower()


class TestRunAgentsPredictionStorage:
    async def test_store_prediction_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.store_prediction(
            event_id="event_123",
            agent=sample_agent,
            prediction_value=0.75,
            run_id="run_123",
            interval_start_minutes=0,
        )

        mock_db_operations.upsert_predictions.assert_called_once()
        call_args = mock_db_operations.upsert_predictions.call_args[0][0]

        assert len(call_args) == 1
        prediction = call_args[0]
        assert prediction.unique_event_id == "event_123"
        assert prediction.miner_uid == 42
        assert prediction.miner_hotkey == "5HotKey123"
        assert prediction.latest_prediction == 0.75
        assert prediction.interval_agg_prediction == 0.75
        assert prediction.version_id == "a23e4567-e89b-12d3-a456-426614174000"

    async def test_store_prediction_clips_values(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.store_prediction(
            event_id="event_123",
            agent=sample_agent,
            prediction_value=1.5,
            run_id="run_123",
            interval_start_minutes=0,
        )

        call_args = mock_db_operations.upsert_predictions.call_args[0][0]
        prediction = call_args[0]
        assert prediction.latest_prediction == 1.0
        assert prediction.interval_agg_prediction == 1.0

    async def test_store_prediction_handles_failure(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock(side_effect=Exception("Database error"))

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.store_prediction(
            event_id="event_123",
            agent=sample_agent,
            prediction_value=0.75,
            run_id="run_123",
            interval_start_minutes=0,
        )

        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args
        assert "Failed to store prediction" in call_args[0][0]


class TestRunAgentsStoreSources:
    def _make_task(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        return RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

    def _valid_source(self, url: str = "https://example.com") -> dict:
        return {
            "url": url,
            "source_type": "news",
            "direction": "up",
            "source_timestamp": "2024-01-01T12:00:00+00:00",
            "impact_bucket": "high",
            "persistence_bucket": "medium",
            "reasoning": "supports outcome",
        }

    async def test_stores_valid_sources(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        result = {
            "output": {
                "sources": [
                    self._valid_source("https://a.com"),
                    self._valid_source("https://b.com"),
                ]
            }
        }

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, result)

        mock_db_operations.insert_sources.assert_called_once()
        run_id, sources = mock_db_operations.insert_sources.call_args[0]
        assert run_id == "run_1"
        assert len(sources) == 2
        assert sources[0].url == "https://a.com"
        assert sources[1].url == "https://b.com"

    async def test_skips_when_status_not_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        result = {"output": {"sources": [self._valid_source()]}}

        await task._store_sources("run_1", AgentRunStatus.INVALID_SANDBOX_OUTPUT, result)

        mock_db_operations.insert_sources.assert_not_called()

    async def test_skips_when_result_is_none(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, None)

        mock_db_operations.insert_sources.assert_not_called()

    async def test_skips_when_sources_missing(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        result = {"output": {"prediction": 0.5}}

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, result)

        mock_db_operations.insert_sources.assert_not_called()

    async def test_skips_when_sources_empty(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        result = {"output": {"sources": []}}

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, result)

        mock_db_operations.insert_sources.assert_not_called()

    async def test_truncates_to_max_per_run(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        from neurons.validator.models.sources import MAX_SOURCES_PER_RUN

        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        raw_sources = [
            self._valid_source(f"https://{i}.com") for i in range(MAX_SOURCES_PER_RUN + 5)
        ]
        result = {"output": {"sources": raw_sources}}

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, result)

        _, stored = mock_db_operations.insert_sources.call_args[0]
        assert len(stored) == MAX_SOURCES_PER_RUN
        assert stored[0].url == "https://0.com"
        assert stored[-1].url == f"https://{MAX_SOURCES_PER_RUN - 1}.com"

    async def test_handles_insert_failure(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        mock_db_operations.insert_sources = AsyncMock(side_effect=Exception("db broken"))
        task = self._make_task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        result = {"output": {"sources": [self._valid_source()]}}

        await task._store_sources("run_1", AgentRunStatus.SUCCESS, result)

        mock_logger.error.assert_called()
        assert "Failed to store sources" in mock_logger.error.call_args[0][0]


class TestRunAgentsErrorLogging:
    async def test_logs_exported_on_agent_execution_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        error_result = {
            "status": "error",
            "error": "agent_main() must return a dict, got NoneType.",
            "traceback": "Traceback (most recent call last):\n  File ...\nException: ...",
            "logs": "[AGENT_RUNNER] Starting\n[AGENT_RUNNER] Error occurred",
        }
        task.run_sandbox = AsyncMock(return_value=error_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.insert_agent_run_log.assert_called_once()
        call_args = mock_db_operations.insert_agent_run_log.call_args
        run_id = call_args[0][0]
        logs = call_args[0][1]

        assert run_id is not None
        assert "[AGENT_RUNNER] Starting" in logs
        assert "ERROR DETAILS" in logs
        assert "agent_main() must return a dict" in logs
        assert "Traceback" in logs

    async def test_logs_exported_on_timeout(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        timeout_result = {
            "status": "error",
            "error": "Timeout exceeded",
            "logs": "[AGENT_RUNNER] Starting\n[AGENT_RUNNER] Processing...\n<execution stopped>",
        }
        task.run_sandbox = AsyncMock(return_value=timeout_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.insert_agent_run_log.assert_called_once()
        call_args = mock_db_operations.insert_agent_run_log.call_args
        run_id = call_args[0][0]
        logs = call_args[0][1]

        assert run_id is not None
        assert "[AGENT_RUNNER] Starting" in logs
        assert "TIMEOUT" in logs
        assert "Execution exceeded timeout limit" in logs

    async def test_logs_exported_on_validation_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        invalid_result = {
            "status": "SUCCESS",
            "output": {"event_id": "event_123"},
            "logs": "[AGENT_RUNNER] Starting\n[AGENT_RUNNER] Completed",
        }
        task.run_sandbox = AsyncMock(return_value=invalid_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.insert_agent_run_log.assert_called_once()
        call_args = mock_db_operations.insert_agent_run_log.call_args
        run_id = call_args[0][0]
        logs = call_args[0][1]

        assert run_id is not None
        assert "[AGENT_RUNNER] Starting" in logs
        assert "[AGENT_RUNNER] Completed" in logs

    async def test_logs_exported_on_result_none(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        task.run_sandbox = AsyncMock(return_value=None)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.insert_agent_run_log.assert_called_once()
        call_args = mock_db_operations.insert_agent_run_log.call_args
        run_id = call_args[0][0]
        logs = call_args[0][1]

        assert run_id is not None
        assert "Sandbox timeout - no logs" in logs


class TestRunAgentsSyncHour:
    async def test_run_skips_when_before_sync_hour(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 5
            mock_datetime.now.return_value = mock_now

            with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
                await task.run()

            mock_datetime.now.assert_called_once_with(timezone.utc)

        mock_logger.debug.assert_called_with(
            "Before execution window",
            extra={"current_hour": 5, "sync_hour": 10},
        )
        mock_db_operations.get_events_to_predict.assert_not_called()

    async def test_run_executes_when_at_sync_hour(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 10
            mock_datetime.now.return_value = mock_now

            with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
                await task.run()

            mock_datetime.now.assert_called_once_with(timezone.utc)

        mock_db_operations.get_events_to_predict.assert_called_once()

    async def test_run_executes_when_after_sync_hour(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 15
            mock_datetime.now.return_value = mock_now

            with patch("neurons.validator.tasks.run_agents.Subtensor", mock_subtensor_cm):
                await task.run()

            mock_datetime.now.assert_called_once_with(timezone.utc)

        mock_db_operations.get_events_to_predict.assert_called_once()


class TestRunAgentsDetermineRunStatus:
    def test_timeout_at_run_agents_level(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        status, prediction = task._determine_status_and_extract_prediction(
            result=None, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.SANDBOX_TIMEOUT
        assert prediction is None

    def test_success_with_valid_prediction(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {"status": "SUCCESS", "output": {"prediction": 0.75}}
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.SUCCESS
        assert prediction == 0.75

    def test_success_but_invalid_prediction_type(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {"status": "SUCCESS", "output": {"prediction": "invalid"}}
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None

    def test_success_but_missing_prediction_field(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {"status": "SUCCESS", "output": {"event_id": "event1"}}
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None

    def test_success_but_invalid_output_format(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {"status": "SUCCESS", "output": "not a dict"}
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None

    def test_error_timeout(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {
            "status": "error",
            "error": "Timeout exceeded",
            "error_type": SandboxErrorType.TIMEOUT,
        }
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.SANDBOX_TIMEOUT
        assert prediction is None

    def test_error_container_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {
            "status": "error",
            "error": "Container error: Failed to start",
            "error_type": SandboxErrorType.CONTAINER_ERROR,
        }
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.SANDBOX_TIMEOUT
        assert prediction is None

    def test_error_invalid_output(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {
            "status": "error",
            "error": "Failed to read output.json",
            "error_type": SandboxErrorType.INVALID_OUTPUT,
        }
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None

    def test_error_agent_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {
            "status": "error",
            "error": "agent_main() must return a dict",
            "error_type": SandboxErrorType.AGENT_ERROR,
        }
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INTERNAL_AGENT_ERROR
        assert prediction is None

    def test_error_unknown_type(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = {"status": "error", "error": "Unknown error", "error_type": "unknown_type"}
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INTERNAL_AGENT_ERROR
        assert prediction is None
        mock_logger.warning.assert_called_once()

    def test_invalid_result_type(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = "not a dict"
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None


class TestRunAgentsCreateAgentRun:
    async def test_create_agent_run_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run_id = "test-run-123"
        event_id = "event-456"

        run = await task._create_agent_run(
            interval_start_minutes=1000,
            run_id=run_id,
            event_id=event_id,
            agent=sample_agent,
            status=AgentRunStatus.SUCCESS,
        )

        assert isinstance(run, AgentRunsModel)
        assert run.run_id == run_id
        assert run.unique_event_id == event_id
        assert run.agent_version_id == sample_agent.version_id
        assert run.miner_uid == sample_agent.miner_uid
        assert run.miner_hotkey == sample_agent.miner_hotkey
        assert run.status == AgentRunStatus.SUCCESS
        assert run.exported is False
        assert run.is_final is True

    async def test_create_agent_run_internal_agent_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run_id = "test-run-789"
        event_id = "event-abc"

        run = await task._create_agent_run(
            interval_start_minutes=1000,
            run_id=run_id,
            event_id=event_id,
            agent=sample_agent,
            status=AgentRunStatus.INTERNAL_AGENT_ERROR,
        )

        assert run.status == AgentRunStatus.INTERNAL_AGENT_ERROR
        assert run.is_final is True

    @patch("neurons.validator.tasks.run_agents.MAX_TIMEOUT_RETRIES", 2)
    async def test_create_agent_run_sandbox_timeout_not_final(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
            interval_start_minutes=1000,
            run_id="run-timeout",
            event_id="event-timeout",
            agent=sample_agent,
            status=AgentRunStatus.SANDBOX_TIMEOUT,
        )

        assert run.status == AgentRunStatus.SANDBOX_TIMEOUT
        assert run.is_final is False

    async def test_create_agent_run_sandbox_timeout_final(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(
            return_value=MAX_TIMEOUT_RETRIES
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
            interval_start_minutes=1000,
            run_id="run-timeout-final",
            event_id="event-timeout",
            agent=sample_agent,
            status=AgentRunStatus.SANDBOX_TIMEOUT,
        )

        assert run.status == AgentRunStatus.SANDBOX_TIMEOUT
        assert run.is_final is True

    async def test_create_agent_run_invalid_sandbox_output(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
            interval_start_minutes=1000,
            run_id="run-invalid",
            event_id="event-invalid",
            agent=sample_agent,
            status=AgentRunStatus.INVALID_SANDBOX_OUTPUT,
        )

        assert run.status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert run.is_final is True


class TestRunAgentsRunCreation:
    async def test_creates_run_on_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.upsert_agent_runs = AsyncMock()
        mock_db_operations.insert_agent_run_log = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        success_result = {
            "status": "SUCCESS",
            "output": {"event_id": "external_event_123", "prediction": 0.75},
            "logs": "[AGENT_RUNNER] Success",
        }
        task.run_sandbox = AsyncMock(return_value=success_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        # Verify run was created
        mock_db_operations.upsert_agent_runs.assert_called_once()
        runs = mock_db_operations.upsert_agent_runs.call_args[0][0]
        assert len(runs) == 1
        run = runs[0]
        assert run.status == AgentRunStatus.SUCCESS
        assert run.is_final is True
        assert run.exported is False
        assert run.unique_event_id == "event_123"
        assert run.agent_version_id == sample_agent.version_id

        # Verify prediction was stored
        mock_db_operations.upsert_predictions.assert_called_once()

    @patch("neurons.validator.tasks.run_agents.MAX_TIMEOUT_RETRIES", 2)
    async def test_creates_run_on_timeout(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_agent_runs = AsyncMock()
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        task.run_sandbox = AsyncMock(return_value=None)  # Timeout

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.upsert_agent_runs.assert_called_once()
        runs = mock_db_operations.upsert_agent_runs.call_args[0][0]
        assert len(runs) == 1
        run = runs[0]
        assert run.status == AgentRunStatus.SANDBOX_TIMEOUT
        assert run.is_final is False
        assert run.exported is False

    async def test_creates_run_on_agent_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_agent_runs = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        error_result = {
            "status": "error",
            "error": "agent_main() must return a dict",
            "error_type": SandboxErrorType.AGENT_ERROR,
            "traceback": "Traceback...",
            "logs": "[AGENT_RUNNER] Error",
        }
        task.run_sandbox = AsyncMock(return_value=error_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.upsert_agent_runs.assert_called_once()
        runs = mock_db_operations.upsert_agent_runs.call_args[0][0]
        assert len(runs) == 1
        run = runs[0]
        assert run.status == AgentRunStatus.INTERNAL_AGENT_ERROR
        assert run.is_final is True
        assert run.exported is False

    async def test_creates_run_on_invalid_output(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_agent_runs = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        # Success status but missing prediction field
        invalid_result = {
            "status": "SUCCESS",
            "output": {"event_id": "external_event_123"},  # Missing prediction!
            "logs": "[AGENT_RUNNER] Success but invalid",
        }
        task.run_sandbox = AsyncMock(return_value=invalid_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.upsert_agent_runs.assert_called_once()
        runs = mock_db_operations.upsert_agent_runs.call_args[0][0]
        assert len(runs) == 1
        run = runs[0]
        assert run.status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert run.is_final is True
        assert run.exported is False

    async def test_run_links_to_prediction_on_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.upsert_agent_runs = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        success_result = {
            "status": "SUCCESS",
            "output": {"event_id": "external_event_123", "prediction": 0.85},
            "logs": "[AGENT_RUNNER] Success",
        }
        task.run_sandbox = AsyncMock(return_value=success_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        # Verify run and prediction share same run_id
        runs = mock_db_operations.upsert_agent_runs.call_args[0][0]
        predictions = mock_db_operations.upsert_predictions.call_args[0][0]

        assert len(runs) == 1
        assert len(predictions) == 1

        run_id = runs[0].run_id
        prediction_run_id = predictions[0].run_id

        assert run_id == prediction_run_id
        assert runs[0].status == AgentRunStatus.SUCCESS
        assert predictions[0].latest_prediction == 0.85

    async def test_no_prediction_stored_on_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.upsert_agent_runs = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        error_result = {
            "status": "error",
            "error": "Something went wrong",
            "error_type": SandboxErrorType.AGENT_ERROR,
            "logs": "[AGENT_RUNNER] Error",
        }
        task.run_sandbox = AsyncMock(return_value=error_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.upsert_agent_runs.assert_called_once()
        mock_db_operations.upsert_predictions.assert_not_called()

        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if len(call[0]) > 0
            and "Agent execution completed with non-success status" in call[0][0]
        ]
        assert len(debug_calls) == 1

    async def test_api_create_agent_run_called_with_correct_params(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.upsert_agent_runs = AsyncMock()
        mock_db_operations.insert_agent_run_log = AsyncMock()

        mock_api_client.create_agent_run = AsyncMock(
            return_value=CreateAgentRunResponse(run_id=UUID("223e4567-e89b-12d3-a456-426614174001"))
        )

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
            validator_uid=99,
            validator_hotkey="5ValidatorHotkey999",
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        success_result = {
            "status": "SUCCESS",
            "output": {"event_id": "external_event_123", "prediction": 0.75},
            "logs": "[AGENT_RUNNER] Success",
        }
        task.run_sandbox = AsyncMock(return_value=success_result)

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_api_client.create_agent_run.assert_called_once()
        call_args = mock_api_client.create_agent_run.call_args[0][0]

        assert isinstance(call_args, CreateAgentRunRequest)
        assert call_args.miner_uid == sample_agent.miner_uid
        assert call_args.miner_hotkey == sample_agent.miner_hotkey
        assert call_args.vali_uid == 99
        assert call_args.vali_hotkey == "5ValidatorHotkey999"
        assert call_args.event_id == "event_123"
        assert str(call_args.version_id) == sample_agent.version_id
        # The backend keys the run on the interval the validator is running for.
        assert call_args.interval_datetime == datetime.fromisoformat(
            get_interval_iso_datetime(1000)
        )

    async def test_api_create_agent_run_failure_returns_early(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()
        mock_db_operations.upsert_agent_runs = AsyncMock()

        mock_api_client.create_agent_run = AsyncMock(side_effect=Exception("API Error"))

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
            validator_uid=99,
            validator_hotkey="5ValidatorHotkey999",
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        task.run_sandbox = AsyncMock()

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_api_client.create_agent_run.assert_called_once()
        task.run_sandbox.assert_not_called()
        mock_db_operations.upsert_agent_runs.assert_not_called()
        mock_db_operations.upsert_predictions.assert_not_called()

        error_calls = [
            call
            for call in mock_logger.error.call_args_list
            if len(call[0]) > 0 and "Failed to create agent run via API" in call[0][0]
        ]
        assert len(error_calls) == 1


class TestRunAgentsMaxRetries:
    async def test_first_timeout_allows_execution(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.get_latest_prediction_for_event_and_miner = AsyncMock(return_value=None)
        mock_db_operations.has_final_run = AsyncMock(return_value=False)
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )
        task.execute_agent_for_event = AsyncMock()

        semaphore = asyncio.Semaphore(1)
        await task.execute_with_semaphore(
            semaphore=semaphore,
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        mock_db_operations.has_final_run.assert_called_once_with(
            unique_event_id="event_123",
            agent_version_id="a23e4567-e89b-12d3-a456-426614174000",
            interval_start_minutes=1000,
        )
        task.execute_agent_for_event.assert_called_once()

    async def test_second_timeout_allows_execution(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.get_latest_prediction_for_event_and_miner = AsyncMock(return_value=None)
        mock_db_operations.has_final_run = AsyncMock(return_value=False)
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=1)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )
        task.execute_agent_for_event = AsyncMock()

        semaphore = asyncio.Semaphore(1)
        await task.execute_with_semaphore(
            semaphore=semaphore,
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        task.execute_agent_for_event.assert_called_once()

    async def test_third_timeout_allows_execution_creates_final_run(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.get_latest_prediction_for_event_and_miner = AsyncMock(return_value=None)
        mock_db_operations.has_final_run = AsyncMock(return_value=False)
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=2)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )
        task.execute_agent_for_event = AsyncMock()

        semaphore = asyncio.Semaphore(1)
        await task.execute_with_semaphore(
            semaphore=semaphore,
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        task.execute_agent_for_event.assert_called_once()

    async def test_fourth_call_skips_when_final_run_exists(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        mock_db_operations.get_latest_prediction_for_event_and_miner = AsyncMock(return_value=None)
        mock_db_operations.has_final_run = AsyncMock(return_value=True)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )
        task.execute_agent_for_event = AsyncMock()

        semaphore = asyncio.Semaphore(1)
        await task.execute_with_semaphore(
            semaphore=semaphore,
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        task.execute_agent_for_event.assert_not_called()
        mock_logger.debug.assert_called_with(
            "Skipping execution - final run exists for interval",
            extra={
                "event_id": "event_123",
                "agent_version_id": "a23e4567-e89b-12d3-a456-426614174000",
                "interval_start_minutes": 1000,
            },
        )


class TestRunAgentsMemory:
    def _task(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        return RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            netuid=99,
            network="test",
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

    async def test_pull_memory_builds_lookup(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_api_client.pull_memory = AsyncMock(
            return_value=MemoryPullResponse(
                items=[
                    MemoryEntry(
                        miner_uid=42,
                        miner_hotkey="5HotKey123",
                        event_id="event_123",
                        interval_datetime="2026-07-24T00:00:00+00:00",
                        memory="prior blob",
                    )
                ],
                count=1,
            )
        )

        lookup = await task._pull_memory([(sample_event_tuple, sample_agent)], 1000)

        assert lookup == {(42, "5HotKey123", "event_123"): "prior blob"}

        body = mock_api_client.pull_memory.call_args.args[0]
        assert body.pairs[0].miner_uid == 42
        assert body.pairs[0].miner_hotkey == "5HotKey123"
        assert body.pairs[0].event_id == "event_123"
        assert body.interval_datetime == datetime.fromisoformat(get_interval_iso_datetime(1000))

    async def test_pull_memory_empty_pairs_no_call(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_api_client.pull_memory = AsyncMock()

        lookup = await task._pull_memory([], 1000)

        assert lookup == {}
        mock_api_client.pull_memory.assert_not_called()

    async def test_pull_memory_failure_returns_empty(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_api_client.pull_memory = AsyncMock(side_effect=Exception("backend down"))

        lookup = await task._pull_memory([(sample_event_tuple, sample_agent)], 1000)

        assert lookup == {}

    async def test_memory_injected_into_event_data(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_db_operations.upsert_agent_runs = AsyncMock()
        mock_db_operations.insert_agent_run_log = AsyncMock()
        task.load_agent_code = AsyncMock(return_value="def agent_main(e): pass")
        task.run_sandbox = AsyncMock(return_value={"status": "error", "error": "x", "logs": ""})

        memory_by_pair = {(42, "5HotKey123", "event_123"): "prior blob"}

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair=memory_by_pair,
        )

        event_data = task.run_sandbox.call_args.args[1]
        assert event_data["memory"] == "prior blob"

    async def test_memory_injected_none_when_absent(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_db_operations.upsert_agent_runs = AsyncMock()
        mock_db_operations.insert_agent_run_log = AsyncMock()
        task.load_agent_code = AsyncMock(return_value="def agent_main(e): pass")
        task.run_sandbox = AsyncMock(return_value={"status": "error", "error": "x", "logs": ""})

        await task.execute_agent_for_event(
            event=sample_event_tuple,
            agent=sample_agent,
            interval_start_minutes=1000,
            memory_by_pair={},
        )

        event_data = task.run_sandbox.call_args.args[1]
        assert event_data["memory"] is None

    async def test_store_memory_on_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_db_operations.insert_reforecast_memory = AsyncMock()

        result = {
            "status": "SUCCESS",
            "output": {"event_id": "e", "prediction": 0.5, "memory": "new blob"},
        }

        await task._store_memory("run-1", AgentRunStatus.SUCCESS, result, 1000)

        mock_db_operations.insert_reforecast_memory.assert_awaited_once_with(
            "run-1", "new blob", 1000
        )

    async def test_store_memory_skipped_when_no_memory_key(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_db_operations.insert_reforecast_memory = AsyncMock()

        result = {"status": "SUCCESS", "output": {"event_id": "e", "prediction": 0.5}}

        await task._store_memory("run-1", AgentRunStatus.SUCCESS, result, 1000)

        mock_db_operations.insert_reforecast_memory.assert_not_called()

    async def test_store_memory_skipped_on_non_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_subtensor_cm,
        mock_api_client,
        mock_logger,
    ):
        task = self._task(
            mock_db_operations,
            mock_sandbox_manager,
            mock_subtensor_cm,
            mock_api_client,
            mock_logger,
        )
        mock_db_operations.insert_reforecast_memory = AsyncMock()

        await task._store_memory("run-1", AgentRunStatus.SANDBOX_TIMEOUT, None, 1000)

        mock_db_operations.insert_reforecast_memory.assert_not_called()
