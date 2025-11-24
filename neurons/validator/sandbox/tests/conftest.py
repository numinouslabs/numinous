import logging
import shutil
import tempfile
from typing import Dict
from unittest.mock import MagicMock

import docker
import pytest
from bittensor_wallet import Wallet

from neurons.validator.utils.logger.logger import NuminousLogger


@pytest.fixture
def mock_docker_client() -> MagicMock:
    mock_client = MagicMock(spec=docker.DockerClient)
    mock_container = MagicMock()
    mock_container.wait = MagicMock(return_value={"StatusCode": 0})
    mock_container.logs = MagicMock(return_value=b"test logs")
    mock_container.remove = MagicMock()
    mock_container.kill = MagicMock()
    mock_client.containers.run = MagicMock(return_value=mock_container)
    return mock_client


@pytest.fixture
def mock_wallet() -> MagicMock:
    return MagicMock(spec=Wallet)


@pytest.fixture
def mock_logger() -> NuminousLogger:
    logging.setLoggerClass(NuminousLogger)
    logger = logging.getLogger("test_sandbox_manager")
    logger.setLevel(logging.DEBUG)
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def temp_dir() -> str:
    temp_dir = tempfile.mkdtemp(prefix="test_sandbox_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_on_finish() -> MagicMock:
    return MagicMock()


@pytest.fixture
def sample_agent_code() -> str:
    return """
def agent_main(event_data):
    return {"event_id": event_data["event_id"], "prediction": 0.5}
"""


@pytest.fixture
def sample_env_vars() -> Dict[str, str]:
    return {"SANDBOX_PROXY_URL": "http://localhost:8080", "RUN_ID": "test-run-id"}


@pytest.fixture
def mock_docker_setup(monkeypatch):
    mock_docker_client = MagicMock(spec=docker.DockerClient)

    mock_containers_list = []
    mock_docker_client.containers.list = MagicMock(return_value=mock_containers_list)

    mock_network = MagicMock()
    mock_docker_client.networks.get = MagicMock(return_value=mock_network)
    mock_docker_client.networks.create = MagicMock(return_value=mock_network)

    mock_proxy_container = MagicMock()
    mock_proxy_container.status = "running"
    mock_docker_client.containers.get = MagicMock(side_effect=docker.errors.NotFound("Not found"))
    mock_docker_client.containers.run = MagicMock(return_value=mock_proxy_container)

    monkeypatch.setattr("docker.from_env", lambda: mock_docker_client)

    return mock_docker_client
