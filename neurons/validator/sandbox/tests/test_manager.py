import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import docker.errors
import pytest
import requests.exceptions

from neurons.validator.models.track import TrackEnum
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
            json.dumps({"status": "SUCCESS", "output": {"event_id": "test", "prediction": 0.5}})
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
    def test_memory_flows_into_result_output(
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
            json.dumps(
                {
                    "status": "success",
                    "output": {
                        "event_id": "test",
                        "prediction": 0.5,
                        "memory": "updated belief blob",
                    },
                }
            )
        )

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

        on_finish.assert_called_once()
        result = on_finish.call_args[0][0]
        assert result["status"] == "success"
        assert result["output"]["memory"] == "updated belief blob"
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
            json.dumps({"status": "SUCCESS", "output": {"event_id": "test", "prediction": 0.5}})
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


class TestSigningProxyLifecycle:
    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_removes_existing_proxy_before_creating_new_one(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger
    ):
        mock_docker_client = MagicMock()
        mock_old_proxy = MagicMock()
        mock_old_proxy.status = "running"
        mock_old_proxy.short_id = "abc123"
        mock_docker_client.containers.get = MagicMock(return_value=mock_old_proxy)
        mock_docker_client.containers.run = MagicMock()
        mock_docker_client.networks.get = MagicMock()
        mock_docker_client.networks.create = MagicMock()

        with patch("docker.from_env", return_value=mock_docker_client):
            SandboxManager(mock_wallet, "http://gateway", mock_logger)

        mock_old_proxy.remove.assert_called_once_with(force=True)
        mock_docker_client.containers.run.assert_called_once()
        call_kwargs = mock_docker_client.containers.run.call_args[1]
        assert "ulimits" in call_kwargs
        assert len(call_kwargs["ulimits"]) == 1
        assert call_kwargs["ulimits"][0]["Name"] == "nofile"
        assert call_kwargs["ulimits"][0]["Soft"] == 65536
        assert call_kwargs["ulimits"][0]["Hard"] == 65536

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_creates_new_proxy_when_none_exists(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger
    ):
        mock_docker_client = MagicMock()
        mock_docker_client.containers.get = MagicMock(
            side_effect=docker.errors.NotFound("proxy not found")
        )
        mock_docker_client.containers.run = MagicMock()
        mock_docker_client.networks.get = MagicMock()
        mock_docker_client.networks.create = MagicMock()

        with patch("docker.from_env", return_value=mock_docker_client):
            SandboxManager(mock_wallet, "http://gateway", mock_logger)

        mock_docker_client.containers.run.assert_called_once()
        call_kwargs = mock_docker_client.containers.run.call_args[1]
        assert "ulimits" in call_kwargs
        assert len(call_kwargs["ulimits"]) == 1
        assert call_kwargs["ulimits"][0]["Name"] == "nofile"
        assert call_kwargs["ulimits"][0]["Soft"] == 65536
        assert call_kwargs["ulimits"][0]["Hard"] == 65536


class TestRunRegistry:
    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_register_run_creates_file(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        manager.register_run("run-123", TrackEnum.SIGNAL)

        registry_file = manager.run_registry_dir / "run-123"
        assert registry_file.exists()
        assert registry_file.read_text() == "SIGNAL"

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_register_run_main_track(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        manager.register_run("run-456", TrackEnum.MAIN)

        registry_file = manager.run_registry_dir / "run-456"
        assert registry_file.exists()
        assert registry_file.read_text() == "MAIN"

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_unregister_run_removes_file(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        manager.register_run("run-789", TrackEnum.SIGNAL)

        registry_file = manager.run_registry_dir / "run-789"
        assert registry_file.exists()

        manager.unregister_run("run-789")
        assert not registry_file.exists()

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_unregister_nonexistent_run_no_error(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)
        manager.unregister_run("nonexistent-run")

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_multiple_concurrent_registrations(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)

        manager.register_run("run-a", TrackEnum.MAIN)
        manager.register_run("run-b", TrackEnum.SIGNAL)
        manager.register_run("run-c", TrackEnum.SIGNAL)

        assert (manager.run_registry_dir / "run-a").read_text() == "MAIN"
        assert (manager.run_registry_dir / "run-b").read_text() == "SIGNAL"
        assert (manager.run_registry_dir / "run-c").read_text() == "SIGNAL"

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_proxy_container_gets_registry_volume(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        manager = SandboxManager(mock_wallet, "http://gateway", mock_logger)

        call_kwargs = mock_docker_setup.containers.run.call_args[1]
        volumes = call_kwargs["volumes"]
        assert str(manager.run_registry_dir) in volumes
        assert volumes[str(manager.run_registry_dir)]["bind"] == "/run_registry"
        assert volumes[str(manager.run_registry_dir)]["mode"] == "ro"

    @patch("neurons.validator.sandbox.manager.build_docker_image")
    @patch("neurons.validator.sandbox.manager.image_exists", return_value=True)
    def test_proxy_container_gets_registry_env_var(
        self, mock_image_exists, mock_build_image, mock_wallet, mock_logger, mock_docker_setup
    ):
        SandboxManager(mock_wallet, "http://gateway", mock_logger)

        call_kwargs = mock_docker_setup.containers.run.call_args[1]
        environment = call_kwargs["environment"]
        assert environment["RUN_REGISTRY_DIR"] == "/run_registry"
