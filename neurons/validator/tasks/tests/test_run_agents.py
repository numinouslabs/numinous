import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.sandbox import SandboxManager
from neurons.validator.sandbox.models import SandboxErrorType
from neurons.validator.tasks.run_agents import RunAgents
from neurons.validator.utils.if_metagraph import IfMetagraph
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
    client = AsyncMock(spec=NuminousClient)
    client.post_agent_logs = AsyncMock()
    return client


@pytest.fixture
def mock_metagraph():
    metagraph = MagicMock(spec=IfMetagraph)
    metagraph.sync = AsyncMock()
    metagraph.block = np.array([12345])
    metagraph.uids = []
    metagraph.axons = []

    return metagraph


@pytest.fixture
def sample_event_tuple():
    return (
        "event_123",
        "external_event_123",
        "polymarket",
        "llm_generated",
        "Will it rain?",
        "Weather forecast unclear",
        datetime(2025, 12, 31, tzinfo=timezone.utc),
        "{}",
    )


@pytest.fixture
def sample_agent():
    return MinerAgentsModel(
        version_id="agent_v1",
        miner_uid=42,
        miner_hotkey="5HotKey123",
        agent_name="test_agent",
        version_number=1,
        file_path="/tmp/test_agent.py",
        pulled_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


class TestRunAgentsInit:
    def test_valid_initialization(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )
        assert task.name == "run-agents"
        assert task.interval_seconds == 600.0

    def test_invalid_interval_negative(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            RunAgents(
                interval_seconds=-1.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_interval_zero(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="interval_seconds must be a positive"):
            RunAgents(
                interval_seconds=0.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_db_operations_type(
        self, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(TypeError, match="db_operations must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations="not_db_ops",
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_sandbox_manager_type(
        self, mock_db_operations, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(TypeError, match="sandbox_manager must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager="not_sandbox",
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_metagraph_type(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger
    ):
        with pytest.raises(TypeError, match="metagraph must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph="not_metagraph",
                api_client=mock_api_client,
                logger=mock_logger,
            )

    def test_invalid_api_client_type(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_logger
    ):
        with pytest.raises(TypeError, match="api_client must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client="not_api_client",
                logger=mock_logger,
            )

    def test_invalid_logger_type(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client
    ):
        with pytest.raises(TypeError, match="logger must be an instance"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger="not_logger",
            )

    def test_invalid_max_concurrent_negative(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="max_concurrent_sandboxes must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
                max_concurrent_sandboxes=-1,
            )

    def test_invalid_max_concurrent_zero(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="max_concurrent_sandboxes must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
                max_concurrent_sandboxes=0,
            )

    def test_invalid_timeout_negative(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="timeout_seconds must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
                timeout_seconds=-1,
            )

    def test_invalid_timeout_zero(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        with pytest.raises(ValueError, match="timeout_seconds must be"):
            RunAgents(
                interval_seconds=600.0,
                db_operations=mock_db_operations,
                sandbox_manager=mock_sandbox_manager,
                metagraph=mock_metagraph,
                api_client=mock_api_client,
                logger=mock_logger,
                timeout_seconds=0,
            )


@pytest.mark.asyncio
class TestRunAgentsRun:
    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_no_events(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
    ):
        # Mock datetime.utcnow() to return hour >= 4 to pass sync_hour check
        mock_datetime.utcnow.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.run()

        mock_metagraph.sync.assert_called_once_with()
        mock_logger.debug.assert_called_with("No events to predict")
        mock_db_operations.get_active_agents.assert_not_called()

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_no_agents(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
    ):
        # Mock datetime.utcnow() to return hour >= 4 to pass sync_hour check
        mock_datetime.utcnow.return_value = datetime(2025, 12, 3, 10, 0, 0)

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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.run()

        mock_logger.warning.assert_called_with("No agents available for execution")


class TestRunAgentsFiltering:
    def test_filter_agent_uid_not_in_metagraph(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger, sample_agent
    ):
        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.uids = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 0

    def test_filter_agent_hotkey_mismatch(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger, sample_agent
    ):
        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.uids = [np.array([42])]

        axon = MagicMock()
        axon.hotkey = "different_hotkey"
        metagraph.axons = {42: axon}

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 0

    def test_filter_agent_no_axon(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger, sample_agent
    ):
        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.uids = [np.array([42])]
        metagraph.axons = {42: None}

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 0

    def test_keep_valid_agent(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger, sample_agent
    ):
        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.uids = [np.array([42])]

        axon = MagicMock()
        axon.hotkey = "5HotKey123"
        metagraph.axons = {42: axon}

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = task.filter_agents_by_metagraph([sample_agent])

        assert len(result) == 1
        assert result[0] == sample_agent

    def test_mixed_filtering(
        self, mock_db_operations, mock_sandbox_manager, mock_api_client, mock_logger
    ):
        agent1 = MinerAgentsModel(
            version_id="v1",
            miner_uid=42,
            miner_hotkey="hotkey1",
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
            agent_name="agent3",
            version_number=1,
            file_path="/tmp/a3.py",
            pulled_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        metagraph = MagicMock(spec=IfMetagraph)
        metagraph.uids = [np.array([42]), np.array([100])]

        axon1 = MagicMock()
        axon1.hotkey = "hotkey1"
        axon3 = MagicMock()
        axon3.hotkey = "hotkey3"
        metagraph.axons = {42: axon1, 100: axon3}

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = task.filter_agents_by_metagraph([agent1, agent2, agent3])

        assert len(result) == 2
        assert result[0] == agent1
        assert result[1] == agent3


class TestRunAgentsParsing:
    def test_parse_event_description_with_separator(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        full_desc = "Will it rain? ==Further Information==: Weather forecast unclear"
        title, description = task.parse_event_description(full_desc)

        assert title == "Will it rain?"
        assert description == "Weather forecast unclear"

    def test_parse_event_description_without_separator(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        full_desc = "Will it rain tomorrow?"
        title, description = task.parse_event_description(full_desc)

        assert title == "Will it rain tomorrow?"
        assert description == "Will it rain tomorrow?"


@pytest.mark.asyncio
class TestRunAgentsIdempotency:
    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_skip_when_prediction_exists(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.utcnow() to return hour >= 4 to pass sync_hour check
        mock_datetime.utcnow.return_value = datetime(2025, 12, 3, 10, 0, 0)

        from neurons.validator.models.prediction import PredictionsModel
        from neurons.validator.utils.common.interval import get_interval_start_minutes

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        mock_metagraph.uids = [np.array([42])]
        axon = MagicMock()
        axon.hotkey = "5HotKey123"
        mock_metagraph.axons = {42: axon}

        # Prediction already exists in current interval
        current_interval = get_interval_start_minutes()
        existing_prediction = PredictionsModel(
            unique_event_id="event_123",
            miner_uid=42,
            miner_hotkey="5HotKey123",
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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        await task.run()

        mock_db_operations.get_latest_prediction_for_event_and_miner.assert_called_once()
        mock_logger.debug.assert_any_call(
            "Skipping execution - prediction exists",
            extra={"event_id": "event_123", "agent_version_id": "agent_v1", "miner_uid": 42},
        )

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_execute_when_prediction_not_exists(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.utcnow() to return hour >= 4 to pass sync_hour check
        mock_datetime.utcnow.return_value = datetime(2025, 12, 3, 10, 0, 0)

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        mock_metagraph.uids = [np.array([42])]
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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_agent_for_event = AsyncMock()

        await task.run()

        mock_db_operations.get_latest_prediction_for_event_and_miner.assert_called_once()
        task.execute_agent_for_event.assert_called_once()

        call_args = task.execute_agent_for_event.call_args[1]
        assert call_args["event_id"] == "event_123"
        assert call_args["agent"] == sample_agent
        assert call_args["event_tuple"] == sample_event_tuple

    @patch("neurons.validator.tasks.run_agents.datetime")
    async def test_replicate_when_prediction_exists_in_different_interval(
        self,
        mock_datetime,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_event_tuple,
        sample_agent,
    ):
        # Mock datetime.utcnow() to return hour >= 4 to pass sync_hour check
        mock_datetime.utcnow.return_value = datetime(2025, 12, 3, 10, 0, 0)

        from neurons.validator.models.prediction import PredictionsModel

        mock_db_operations.get_events_to_predict.return_value = [sample_event_tuple]
        mock_db_operations.get_active_agents.return_value = [sample_agent]

        mock_metagraph.uids = [np.array([42])]
        axon = MagicMock()
        axon.hotkey = "5HotKey123"
        mock_metagraph.axons = {42: axon}

        # Existing prediction in interval 100
        existing_prediction = PredictionsModel(
            unique_event_id="event_123",
            miner_uid=42,
            miner_hotkey="5HotKey123",
            latest_prediction=0.75,
            interval_start_minutes=100,
            interval_agg_prediction=0.75,
            run_id="original_run_id",
            version_id="agent_v1",
        )
        mock_db_operations.get_latest_prediction_for_event_and_miner.return_value = (
            existing_prediction
        )
        mock_db_operations.upsert_predictions = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )
        task.execute_agent_for_event = AsyncMock()

        await task.run()

        # Should not execute agent
        task.execute_agent_for_event.assert_not_called()

        # Should replicate prediction
        mock_db_operations.upsert_predictions.assert_called_once()
        replicated_predictions = mock_db_operations.upsert_predictions.call_args[0][0]
        assert len(replicated_predictions) == 1
        replicated = replicated_predictions[0]
        assert replicated.unique_event_id == "event_123"
        assert replicated.miner_uid == 42
        assert replicated.latest_prediction == 0.75
        assert replicated.run_id == "original_run_id"
        assert replicated.version_id == "agent_v1"

        # Verify info log was called with replication message
        info_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call[0][0] == "Replicated existing prediction to new interval"
        ]
        assert len(info_calls) == 1
        log_extra = info_calls[0][1]["extra"]
        assert log_extra["event_id"] == "event_123"
        assert log_extra["agent_version_id"] == "agent_v1"
        assert log_extra["miner_uid"] == 42
        assert log_extra["from_interval"] == 100
        # to_interval should be different from from_interval
        assert log_extra["to_interval"] != 100


@pytest.mark.asyncio
class TestRunAgentsFileLoading:
    async def test_load_agent_file_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
        tmp_path,
    ):
        agent_file = tmp_path / "test_agent.py"
        agent_file.write_text("def agent_main(): pass")
        sample_agent.file_path = str(agent_file)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        code = await task.load_agent_code(sample_agent)
        assert code == "def agent_main(): pass"

    async def test_load_agent_file_not_found(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        sample_agent.file_path = "/nonexistent/path/agent.py"

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
        tmp_path,
        monkeypatch,
    ):
        from pathlib import Path

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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        code = await task.load_agent_code(sample_agent)
        assert code is None
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Failed to load agent code" in call_args[0][0]


@pytest.mark.asyncio
class TestRunAgentsSandbox:
    async def test_run_sandbox_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args
        assert "timeout" in str(call_args).lower()


@pytest.mark.asyncio
class TestRunAgentsPredictionStorage:
    async def test_store_prediction_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        assert prediction.version_id == "agent_v1"

    async def test_store_prediction_clips_values(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock()

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.upsert_predictions = AsyncMock(side_effect=Exception("Database error"))

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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


@pytest.mark.asyncio
class TestRunAgentsErrorLogging:
    async def test_logs_exported_on_agent_execution_error(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
        sample_event_tuple,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=120,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        task.run_sandbox = AsyncMock(return_value=None)

        await task.execute_agent_for_event(
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
        )

        mock_db_operations.insert_agent_run_log.assert_called_once()
        call_args = mock_db_operations.insert_agent_run_log.call_args
        run_id = call_args[0][0]
        logs = call_args[0][1]

        assert run_id is not None
        assert "Sandbox timeout - no logs" in logs


@pytest.mark.asyncio
class TestRunAgentsSyncHour:
    async def test_run_skips_when_before_sync_hour(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        from unittest.mock import patch

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 5
            mock_datetime.utcnow.return_value = mock_now

            await task.run()

        mock_logger.debug.assert_called_with(
            "Before execution window",
            extra={"current_hour": 5, "sync_hour": 10},
        )
        mock_metagraph.sync.assert_not_called()
        mock_db_operations.get_events_to_predict.assert_not_called()

    async def test_run_executes_when_at_sync_hour(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        from unittest.mock import patch

        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 10
            mock_datetime.utcnow.return_value = mock_now

            await task.run()

        mock_metagraph.sync.assert_called_once()
        mock_db_operations.get_events_to_predict.assert_called_once()

    async def test_run_executes_when_after_sync_hour(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        from unittest.mock import patch

        mock_db_operations.get_events_to_predict.return_value = []

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
            sync_hour=10,
        )

        with patch("neurons.validator.tasks.run_agents.datetime") as mock_datetime:
            mock_now = MagicMock()
            mock_now.hour = 15
            mock_datetime.utcnow.return_value = mock_now

            await task.run()

        mock_metagraph.sync.assert_called_once()
        mock_db_operations.get_events_to_predict.assert_called_once()


class TestRunAgentsDetermineRunStatus:
    def test_timeout_at_run_agents_level(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        status, prediction = task._determine_status_and_extract_prediction(
            result=None, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.SANDBOX_TIMEOUT
        assert prediction is None

    def test_success_with_valid_prediction(
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
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
        self, mock_db_operations, mock_sandbox_manager, mock_metagraph, mock_api_client, mock_logger
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        result = "not a dict"
        status, prediction = task._determine_status_and_extract_prediction(
            result=result, event_id="event1", agent_version_id="agent1", run_id="run1"
        )
        assert status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert prediction is None


@pytest.mark.asyncio
class TestRunAgentsCreateAgentRun:
    async def test_create_agent_run_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run_id = "test-run-123"
        event_id = "event-456"

        run = await task._create_agent_run(
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run_id = "test-run-789"
        event_id = "event-abc"

        run = await task._create_agent_run(
            run_id=run_id,
            event_id=event_id,
            agent=sample_agent,
            status=AgentRunStatus.INTERNAL_AGENT_ERROR,
        )

        assert run.status == AgentRunStatus.INTERNAL_AGENT_ERROR
        assert run.is_final is True

    async def test_create_agent_run_sandbox_timeout_not_final(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=0)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        mock_db_operations.count_runs_for_event_and_agent = AsyncMock(return_value=2)

        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
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
        mock_metagraph,
        mock_api_client,
        mock_logger,
        sample_agent,
    ):
        task = RunAgents(
            interval_seconds=600.0,
            db_operations=mock_db_operations,
            sandbox_manager=mock_sandbox_manager,
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
        )

        run = await task._create_agent_run(
            run_id="run-invalid",
            event_id="event-invalid",
            agent=sample_agent,
            status=AgentRunStatus.INVALID_SANDBOX_OUTPUT,
        )

        assert run.status == AgentRunStatus.INVALID_SANDBOX_OUTPUT
        assert run.is_final is True


@pytest.mark.asyncio
class TestRunAgentsRunCreation:
    async def test_creates_run_on_success(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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

    async def test_creates_run_on_timeout(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
            api_client=mock_api_client,
            logger=mock_logger,
            timeout_seconds=1,
        )

        task.load_agent_code = AsyncMock(return_value="def agent_main(): pass")
        task.run_sandbox = AsyncMock(return_value=None)  # Timeout

        await task.execute_agent_for_event(
            event_id="event_timeout",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
            event_id="event_error",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
            event_id="event_invalid",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
            event_id="event_123",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
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
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
            event_id="event_error",
            agent=sample_agent,
            event_tuple=sample_event_tuple,
            interval_start_minutes=1000,
        )

        mock_db_operations.upsert_agent_runs.assert_called_once()
        mock_db_operations.upsert_predictions.assert_not_called()

        info_calls = [
            call
            for call in mock_logger.info.call_args_list
            if len(call[0]) > 0
            and "Agent execution completed with non-success status" in call[0][0]
        ]
        assert len(info_calls) == 1


@pytest.mark.asyncio
class TestRunAgentsMaxRetries:
    async def test_first_timeout_allows_execution(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
        )

        mock_db_operations.has_final_run.assert_called_once_with(
            unique_event_id="event_123",
            agent_version_id="agent_v1",
        )
        task.execute_agent_for_event.assert_called_once()

    async def test_second_timeout_allows_execution(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
        )

        task.execute_agent_for_event.assert_called_once()

    async def test_third_timeout_allows_execution_creates_final_run(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
        )

        task.execute_agent_for_event.assert_called_once()

    async def test_fourth_call_skips_when_final_run_exists(
        self,
        mock_db_operations,
        mock_sandbox_manager,
        mock_metagraph,
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
            metagraph=mock_metagraph,
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
        )

        task.execute_agent_for_event.assert_not_called()
        mock_logger.debug.assert_called_with(
            "Skipping execution - final run exists",
            extra={
                "event_id": "event_123",
                "agent_version_id": "agent_v1",
            },
        )
