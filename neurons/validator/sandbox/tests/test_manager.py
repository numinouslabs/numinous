import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

from neurons.validator.sandbox.manager import SandboxManager
from neurons.validator.sandbox.models import SandboxState


class TestSandboxManagerInit:
    def test_invalid_gateway_url_empty_string(self, mock_wallet, mock_logger):
        with pytest.raises(ValueError, match="gateway_url must be a non-empty string"):
            with patch("neurons.validator.sandbox.manager.docker"):
                SandboxManager(mock_wallet, "", mock_logger)

    def test_invalid_logger_type(self, mock_wallet):
        with pytest.raises(TypeError, match="logger must be an instance of NuminousLogger"):
            with patch("neurons.validator.sandbox.manager.docker"):
                SandboxManager(mock_wallet, "http://gateway", "not_a_logger")


class TestSandboxManagerCore:
    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_context_manager_calls_cleanup(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        manager.cleanup_all_sandboxes = MagicMock()
        with manager:
            pass
        manager.cleanup_all_sandboxes.assert_called_once()

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_native_docker_timeout_used(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        mock_container = MagicMock()
        mock_container.wait = MagicMock(return_value={"StatusCode": 0})
        mock_container.logs = MagicMock(return_value=b"test logs")
        mock_container.remove = MagicMock()

        manager.docker_client.containers.run = MagicMock(return_value=mock_container)
        temp_dir = tempfile.mkdtemp(prefix="test_sandbox_")
        (Path(temp_dir) / "output.json").write_text(
            json.dumps({"status": "success", "output": {"event_id": "test", "prediction": 0.5}})
        )

        sandbox_id = "sandbox_test"
        manager.sandboxes[sandbox_id] = SandboxState(
            temp_dir=temp_dir,
            run_id="test-run",
            env_vars={"RUN_ID": "test"},
            on_finish=MagicMock(),
            timeout=60,
            start_time=time.time(),
            container=None,
        )

        manager._run_sandbox(sandbox_id)
        mock_container.wait.assert_called_once_with(timeout=60)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_timeout_error_kills_container(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        mock_container = MagicMock()
        mock_container.wait = MagicMock(side_effect=requests.exceptions.ReadTimeout("Timeout"))
        mock_container.kill = MagicMock()

        manager.docker_client.containers.run = MagicMock(return_value=mock_container)
        temp_dir = tempfile.mkdtemp(prefix="test_sandbox_")
        on_finish = MagicMock()

        manager.sandboxes["sandbox_test"] = SandboxState(
            temp_dir=temp_dir,
            run_id="test-run",
            env_vars={"RUN_ID": "test"},
            on_finish=on_finish,
            timeout=60,
            start_time=time.time(),
            container=None,
        )

        manager._run_sandbox("sandbox_test")

        mock_container.kill.assert_called_once()
        on_finish.assert_called_once()
        assert on_finish.call_args[0][0]["status"] == "error"
        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_log_exception_handled_gracefully(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        mock_container = MagicMock()
        mock_container.wait = MagicMock(return_value={"StatusCode": 0})
        mock_container.logs = MagicMock(side_effect=Exception("Log read failed"))
        mock_container.remove = MagicMock()

        manager.docker_client.containers.run = MagicMock(return_value=mock_container)
        temp_dir = tempfile.mkdtemp(prefix="test_sandbox_")
        (Path(temp_dir) / "output.json").write_text(
            json.dumps({"status": "success", "output": {"event_id": "test", "prediction": 0.5}})
        )

        manager.sandboxes["sandbox_test"] = SandboxState(
            temp_dir=temp_dir,
            run_id="test-run",
            env_vars={"RUN_ID": "test"},
            on_finish=MagicMock(),
            timeout=60,
            start_time=time.time(),
            container=None,
        )

        manager._run_sandbox("sandbox_test")

        mock_logger.warning.assert_called()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("neurons.validator.sandbox.manager.threading.Thread")
    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_create_sandbox_creates_required_files(
        self,
        mock_image_exists,
        mock_build_image,
        mock_thread,
        mock_wallet,
        mock_logger,
        mock_docker_setup,
        sample_agent_code,
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        sandbox_id = manager.create_sandbox(
            agent_code=sample_agent_code,
            event_data={"event_id": "test"},
            run_id="run1",
            on_finish=MagicMock(),
        )

        temp_dir = Path(manager.sandboxes[sandbox_id].temp_dir)
        assert (temp_dir / "agent.py").exists()
        assert (temp_dir / "input.json").exists()

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_cleanup_removes_temp_directory(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        temp_dir = tempfile.mkdtemp(prefix="test_sandbox_")

        manager.sandboxes["sandbox_test"] = SandboxState(
            temp_dir=temp_dir,
            run_id="test-run",
            env_vars={"RUN_ID": "test"},
            on_finish=MagicMock(),
            timeout=60,
            start_time=time.time(),
            container=None,
        )

        manager.cleanup_sandbox("sandbox_test")

        assert not Path(temp_dir).exists()
        assert "sandbox_test" not in manager.sandboxes


class TestSandboxManagerValidation:
    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_create_sandbox_invalid_agent_code(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        with pytest.raises(ValueError, match="agent_code must be a non-empty string"):
            manager.create_sandbox(
                agent_code="",
                event_data={"event_id": "test"},
                run_id="run1",
                on_finish=MagicMock(),
            )

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_create_sandbox_invalid_timeout(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            manager.create_sandbox(
                agent_code="def test(): pass",
                event_data={"event_id": "test"},
                run_id="run1",
                on_finish=MagicMock(),
                timeout=-1,
            )
