import json
import os
import shutil
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Callable, Dict, Optional, Type

import docker
import docker.errors
import requests.exceptions
import urllib3.exceptions
from bittensor_wallet import Wallet

from neurons.validator.sandbox.agent_models import AgentOutput, AgentRunnerOutput, RunStatus
from neurons.validator.sandbox.models import SandboxErrorType, SandboxResult, SandboxState
from neurons.validator.sandbox.utils.docker import build_docker_image, image_exists
from neurons.validator.sandbox.utils.temp import cleanup_temp_dir, create_temp_dir
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __version__

# Constants
SANDBOX_NETWORK_NAME = "ig-validator-sandbox-network"
SANDBOX_SIGNING_PROXY_HOST = "ig_validator_signing_proxy"
SANDBOX_SIGNING_PROXY_PORT = 8888
SANDBOX_SIGNING_PROXY_URL = f"http://{SANDBOX_SIGNING_PROXY_HOST}:{SANDBOX_SIGNING_PROXY_PORT}"


class SandboxManager:
    docker_client: docker.DockerClient
    logger: NuminousLogger
    bt_wallet: Wallet
    gateway_url: str
    signing_proxy_container: Optional[docker.models.containers.Container]
    sandboxes: Dict[str, SandboxState]
    temp_base_dir: Optional[Path]

    def __init__(
        self,
        bt_wallet: Wallet,
        gateway_url: str,
        logger: NuminousLogger,
        *,
        force_rebuild: bool = False,
        temp_base_dir: Optional[Path] = None,
    ) -> None:
        # Validate bt_wallet
        if not isinstance(bt_wallet, Wallet):
            raise TypeError("bt_wallet must be an instance of Wallet.")

        # Validate gateway_url
        if not isinstance(gateway_url, str) or not gateway_url:
            raise ValueError("gateway_url must be a non-empty string.")

        # Validate logger
        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        # Validate force_rebuild
        if not isinstance(force_rebuild, bool):
            raise TypeError("force_rebuild must be a boolean.")

        # Validate temp_base_dir
        if temp_base_dir is not None and not isinstance(temp_base_dir, Path):
            raise TypeError("temp_base_dir must be a Path instance or None.")

        self.bt_wallet = bt_wallet
        self.gateway_url = gateway_url
        self.logger = logger
        self.temp_base_dir = temp_base_dir

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Connected to Docker daemon")
        except Exception as e:
            self.logger.error("Failed to create Docker client", extra={"error": str(e)})
            raise RuntimeError(f"Failed to connect to Docker: {e}.") from e

        # Clean up any existing sandbox containers
        self._cleanup_old_containers()

        # Build Docker images
        self._build_images(force_rebuild)

        # Create isolated network
        self._create_sandbox_network()

        # Start signing proxy
        self.signing_proxy_container = None
        self._create_signing_proxy()

        self.sandboxes: Dict[str, SandboxState] = {}

        self.logger.info(
            "Sandbox Manager initialized",
            extra={"gateway_url": self.gateway_url, "force_rebuild": force_rebuild},
        )

    def __enter__(self) -> "SandboxManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        self.cleanup_all_sandboxes()
        self.logger.info("Sandbox Manager closed")

    def _cleanup_old_containers(self) -> None:
        self.logger.debug("Cleaning up existing sandbox containers")

        for container in self.docker_client.containers.list(all=True):
            if container.name.startswith("sandbox_ig_validator_sandbox_"):
                try:
                    self.logger.debug(
                        "Removing old container", extra={"container_name": container.name}
                    )
                    container.stop(timeout=3)
                    container.remove(force=True)
                except Exception as e:
                    self.logger.warning(
                        "Could not clean up container",
                        extra={"container_name": container.name, "error": str(e)},
                    )

    def _build_images(self, force_rebuild: bool) -> None:
        sandbox_dir = Path(__file__).parent

        # Build sandbox image
        sandbox_image = "ig-validator-sandbox-image"
        if force_rebuild or not image_exists(self.docker_client, sandbox_image):
            build_docker_image(self.docker_client, sandbox_dir, sandbox_image, self.logger)
        else:
            self.logger.info(
                "Docker image already exists, skipping build",
                extra={"image": sandbox_image, "hint": "use force_rebuild=True to rebuild"},
            )

        # Build signing proxy image
        signing_proxy_image = "ig-validator-signing-proxy-image"
        signing_proxy_dir = sandbox_dir / "signing_proxy"
        if force_rebuild or not image_exists(self.docker_client, signing_proxy_image):
            build_docker_image(
                self.docker_client, signing_proxy_dir, signing_proxy_image, self.logger
            )
        else:
            self.logger.info(
                "Docker image already exists, skipping build",
                extra={"image": signing_proxy_image, "hint": "use force_rebuild=True to rebuild"},
            )

    def _create_sandbox_network(self) -> None:
        try:
            try:
                self.docker_client.networks.get(SANDBOX_NETWORK_NAME)
                self.logger.debug(
                    "Sandbox network already exists", extra={"network": SANDBOX_NETWORK_NAME}
                )
            except docker.errors.NotFound:
                self.docker_client.networks.create(
                    SANDBOX_NETWORK_NAME,
                    driver="bridge",
                    internal=True,  # No external access
                )
                self.logger.info(
                    "Created isolated network", extra={"network": SANDBOX_NETWORK_NAME}
                )
        except Exception as e:
            self.logger.error("Failed to create network", extra={"error": str(e)})
            raise RuntimeError(f"Failed to create sandbox network: {e}.") from e

    def _create_signing_proxy(self) -> None:
        try:
            existing_proxy = self.docker_client.containers.get(SANDBOX_SIGNING_PROXY_HOST)
            if existing_proxy.status == "running":
                self.logger.info(
                    "Reusing existing signing proxy container",
                    extra={"container_name": SANDBOX_SIGNING_PROXY_HOST},
                )
                self.signing_proxy_container = existing_proxy
                return
            else:
                self.logger.debug("Removing stopped signing proxy container")
                try:
                    existing_proxy.remove(force=True)
                except Exception as e:
                    self.logger.warning(
                        "Failed to remove stopped signing proxy", extra={"error": str(e)}
                    )
        except docker.errors.NotFound:
            self.logger.debug("No existing signing proxy found, creating new one")
        except Exception as e:
            self.logger.warning(
                "Error checking for existing signing proxy", extra={"error": str(e)}
            )

        # Start new signing proxy
        self.logger.info("Starting new signing proxy")

        # Get host wallet path from environment variable
        host_wallet_path = os.environ.get("HOST_WALLET_PATH")
        if host_wallet_path:
            wallet_path = Path(host_wallet_path)
            self.logger.info(
                "Using explicit host wallet path from environment",
                extra={"host_path": str(wallet_path)},
            )
        else:
            # Fallback for backward compatibility
            wallet_path = Path(self.bt_wallet.path).expanduser()
            self.logger.warning(
                "HOST_WALLET_PATH not set, using fallback. "
                "This may fail if wallets are not in /root/. "
                "Please set HOST_WALLET_PATH in docker-compose environment.",
                extra={"host_path": str(wallet_path)},
            )

        try:
            self.signing_proxy_container = self.docker_client.containers.run(
                "ig-validator-signing-proxy-image",
                name=SANDBOX_SIGNING_PROXY_HOST,
                network=SANDBOX_NETWORK_NAME,
                environment={
                    "VALIDATOR_WALLET_NAME": self.bt_wallet.name,
                    "VALIDATOR_WALLET_PATH": "/wallet",
                    "VALIDATOR_WALLET_HOTKEY": self.bt_wallet.hotkey_str,
                    "GATEWAY_URL": self.gateway_url,
                    "VALIDATOR_VERSION": __version__,
                },
                volumes={str(wallet_path): {"bind": "/wallet", "mode": "ro"}},
                remove=False,
                detach=True,
            )
        except docker.errors.APIError as e:
            if "bind source path does not exist" in str(e):
                self.logger.error(
                    f"Wallet directory not found on HOST: {wallet_path}. "
                    f"Check HOST_WALLET_PATH in .env.validator or create wallet with btcli.",
                    extra={"host_path": str(wallet_path)},
                )
                raise FileNotFoundError(
                    f"Wallet directory not found: {wallet_path}. " f"Verify: ls -la {wallet_path}"
                ) from e
            raise

        try:
            bridge_network = self.docker_client.networks.get("bridge")
            bridge_network.connect(self.signing_proxy_container)
            self.logger.debug("Connected signing proxy to bridge network (internet access)")
        except Exception as e:
            self.logger.warning(
                "Failed to connect signing proxy to bridge", extra={"error": str(e)}
            )

        self.logger.info(
            "Signing proxy running",
            extra={"container_name": SANDBOX_SIGNING_PROXY_HOST},
        )

    def create_sandbox(
        self,
        *,
        agent_code: str,
        event_data: dict,
        run_id: str,
        env_vars: Optional[dict[str, str]] = None,
        on_finish: Callable[[dict], None],
        timeout: int = 120,
    ) -> str:
        # Validate agent_code
        if not isinstance(agent_code, str) or not agent_code:
            raise ValueError("agent_code must be a non-empty string.")

        # Validate event_data
        if not isinstance(event_data, dict):
            raise TypeError("event_data must be a dictionary.")

        # Validate run_id
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string.")

        # Validate env_vars
        if env_vars is not None and not isinstance(env_vars, dict):
            raise TypeError("env_vars must be a dictionary or None.")

        # Validate on_finish
        if not callable(on_finish):
            raise TypeError("on_finish must be callable.")

        # Validate timeout
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer.")

        temp_dir = create_temp_dir(base_dir=self.temp_base_dir)
        sandbox_id = f"sandbox_{temp_dir.name}"
        self.logger.debug(
            "Created temp dir for sandbox", extra={"sandbox_id": sandbox_id, "path": str(temp_dir)}
        )

        try:
            # Write agent code to temp directory
            agent_path = temp_dir / "agent.py"
            agent_path.write_text(agent_code)
            self.logger.debug("Wrote agent.py", extra={"sandbox_id": sandbox_id})

            # Write event data as input.json
            input_path = temp_dir / "input.json"
            input_path.write_text(json.dumps(event_data, indent=2))
            self.logger.debug("Wrote input.json", extra={"sandbox_id": sandbox_id})

            # Copy agent_runner.py to temp directory
            runner_path = Path(__file__).parent / "agent_runner.py"
            target_path = temp_dir / "agent_runner.py"
            self.logger.debug(
                "Copying agent_runner.py",
                extra={
                    "sandbox_id": sandbox_id,
                    "source": str(runner_path),
                    "target": str(target_path),
                    "source_exists": runner_path.exists(),
                },
            )
            shutil.copy2(runner_path, target_path)
            self.logger.debug(
                "Copied agent_runner.py",
                extra={"sandbox_id": sandbox_id, "target_exists": target_path.exists()},
            )

            temp_files = list(temp_dir.iterdir())
            self.logger.debug(
                "Temp directory contents",
                extra={
                    "sandbox_id": sandbox_id,
                    "files": [f.name for f in temp_files],
                    "count": len(temp_files),
                },
            )

        except Exception as e:
            self.logger.warning(
                "Sandbox setup failed", extra={"sandbox_id": sandbox_id, "error": str(e)}
            )
            cleanup_temp_dir(temp_dir)
            on_finish({"status": "error", "error": str(e), "traceback": traceback.format_exc()})
            return sandbox_id

        self.sandboxes[sandbox_id] = SandboxState(
            temp_dir=str(temp_dir),
            run_id=run_id,
            env_vars=env_vars or {},
            on_finish=on_finish,
            timeout=timeout,
            start_time=time.time(),
            container=None,
        )

        # Start runner thread
        thread = threading.Thread(target=self._run_sandbox, args=(sandbox_id,), daemon=True)
        thread.start()
        self.logger.debug("Started runner thread", extra={"sandbox_id": sandbox_id})

        return sandbox_id

    def _run_sandbox(self, sandbox_id: str) -> None:
        self.logger.debug("Running sandbox", extra={"sandbox_id": sandbox_id})

        sandbox = self.sandboxes.get(sandbox_id)
        if not sandbox:
            self.logger.warning("Sandbox not found", extra={"sandbox_id": sandbox_id})
            return

        def finish_with_error(
            error_msg: str, result: SandboxResult, error_type: SandboxErrorType
        ) -> None:
            self.logger.warning(
                "Sandbox failed", extra={"sandbox_id": sandbox_id, "error": error_msg}
            )
            result.status = "error"
            result.error = error_msg
            result.error_type = error_type
            try:
                sandbox.on_finish(result.model_dump())
            except Exception as e:
                self.logger.warning(
                    "on_finish callback failed",
                    extra={"sandbox_id": sandbox_id, "error": str(e)},
                )
            finally:
                self.cleanup_sandbox(sandbox_id)

        temp_dir = Path(sandbox.temp_dir)
        result = SandboxResult(status="success", logs="")

        try:
            container_args = {
                "image": "ig-validator-sandbox-image",
                "command": "python /sandbox/agent_runner.py",
                "name": sandbox_id,
                "volumes": {str(temp_dir): {"bind": "/sandbox", "mode": "rw"}},
                "environment": {
                    "SANDBOX_PROXY_URL": SANDBOX_SIGNING_PROXY_URL,
                    "RUN_ID": sandbox.run_id,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **sandbox.env_vars,
                },
                "network": SANDBOX_NETWORK_NAME,
                "remove": False,
                "detach": True,
                "mem_limit": "768m",  # 0.75GB RAM per sandbox
                "memswap_limit": "768m",  # No swap
                "cpu_quota": 50000,  # 0.5 CPU (50000 = 50% of 1 core)
                "cpu_period": 100000,
            }

            sandbox.container = self.docker_client.containers.run(**container_args)

            try:
                # Wait for container to finish with timeout
                sandbox.container.wait(timeout=sandbox.timeout)
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ReadTimeoutError,
            ):
                sandbox.container.kill()

                try:
                    result.logs = sandbox.container.logs(stderr=False).decode("utf-8")
                    self.logger.debug(
                        "Captured partial logs on timeout",
                        extra={"sandbox_id": sandbox_id, "lines": len(result.logs.splitlines())},
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to capture logs on timeout",
                        extra={"sandbox_id": sandbox_id, "error": str(e)},
                    )
                    result.logs = f"Failed to capture partial logs on timeout: {e}"

                finish_with_error("Timeout exceeded", result, error_type=SandboxErrorType.TIMEOUT)
                return

            self.logger.debug("Sandbox finished", extra={"sandbox_id": sandbox_id})

            try:
                logs = sandbox.container.logs(stderr=False).decode("utf-8")
                result.logs = logs
                self.logger.debug(
                    "Captured logs",
                    extra={"sandbox_id": sandbox_id, "lines": len(logs.splitlines())},
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to capture container logs",
                    extra={"sandbox_id": sandbox_id, "error": str(e)},
                )
                result.logs = ""

            # Remove container
            sandbox.container.remove()
            sandbox.container = None

        except Exception as e:
            # Try to capture logs even on container error
            try:
                if sandbox.container:
                    result.logs = sandbox.container.logs(stderr=False).decode("utf-8")
                    self.logger.debug(
                        "Captured logs on container error",
                        extra={"sandbox_id": sandbox_id, "lines": len(result.logs.splitlines())},
                    )
            except Exception as log_err:
                self.logger.debug(
                    "Failed to capture logs on container error",
                    extra={"sandbox_id": sandbox_id, "error": str(log_err)},
                )
                result.logs = ""

            result.traceback = traceback.format_exc()
            finish_with_error(
                f"Container error: {e}", result, error_type=SandboxErrorType.CONTAINER_ERROR
            )
            return

        # Read output.json
        output_path = temp_dir / "output.json"
        try:
            output_dict = json.loads(output_path.read_text())
            self.logger.debug("Read output.json", extra={"sandbox_id": sandbox_id})
        except Exception as e:
            finish_with_error(
                f"Failed to read output.json: {e}",
                result,
                error_type=SandboxErrorType.INVALID_OUTPUT,
            )
            return

        # Validate output structure
        try:
            output = AgentRunnerOutput(**output_dict)
            self.logger.debug("Validated output with Pydantic", extra={"sandbox_id": sandbox_id})
        except Exception as e:
            finish_with_error(
                f"Invalid output.json structure: {e}",
                result,
                error_type=SandboxErrorType.INVALID_OUTPUT,
            )
            return

        # Handle success or error status
        if output.status == RunStatus.SUCCESS:
            if output.output is None:
                finish_with_error(
                    "output.json has status='success' but no 'output' field",
                    result,
                    error_type=SandboxErrorType.INVALID_OUTPUT,
                )
                return

            try:
                agent_output = AgentOutput(**output.output)
                result.output = agent_output.model_dump()
                self.logger.debug(
                    "Validated agent output",
                    extra={
                        "sandbox_id": sandbox_id,
                        "prediction": round(agent_output.prediction, 4),
                    },
                )
            except Exception as e:
                finish_with_error(
                    f"Invalid agent output structure: {e}",
                    result,
                    error_type=SandboxErrorType.INVALID_OUTPUT,
                )
                return

        elif output.status == RunStatus.ERROR:
            if output.error is None:
                finish_with_error(
                    "output.json has status='error' but no 'error' field",
                    result,
                    error_type=SandboxErrorType.INVALID_OUTPUT,
                )
                return
            result.traceback = output.traceback
            finish_with_error(output.error, result, error_type=SandboxErrorType.AGENT_ERROR)
            return
        else:
            finish_with_error(
                f"Invalid status in output.json: {output.status}",
                result,
                error_type=SandboxErrorType.INVALID_OUTPUT,
            )
            return

        # Success - call on_finish
        try:
            sandbox.on_finish(result.model_dump())
        except Exception as e:
            self.logger.warning(
                "on_finish callback failed",
                extra={"sandbox_id": sandbox_id, "error": str(e)},
            )
        finally:
            self.cleanup_sandbox(sandbox_id)

    def cleanup_sandbox(self, sandbox_id: str) -> None:
        sandbox = self.sandboxes.get(sandbox_id)
        if not sandbox:
            return

        # Stop and remove container
        if sandbox.container:
            try:
                sandbox.container.stop()
                sandbox.container.remove()
                self.logger.debug("Removed container", extra={"sandbox_id": sandbox_id})
            except Exception as e:
                self.logger.debug(
                    "Container cleanup error",
                    extra={"sandbox_id": sandbox_id, "error": str(e)},
                )

        # Clean up temp directory
        try:
            cleanup_temp_dir(Path(sandbox.temp_dir))
        except Exception as e:
            self.logger.debug(
                "Temp dir cleanup error",
                extra={"sandbox_id": sandbox_id, "error": str(e)},
            )

        # Remove from tracking
        if sandbox_id in self.sandboxes:
            del self.sandboxes[sandbox_id]
            self.logger.debug("Cleaned up sandbox", extra={"sandbox_id": sandbox_id})

    def cleanup_all_sandboxes(self) -> None:
        for sandbox_id in list(self.sandboxes.keys()):
            self.cleanup_sandbox(sandbox_id)

    def get_num_sandboxes(self) -> int:
        return len(self.sandboxes)
