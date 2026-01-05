import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from bittensor import AsyncSubtensor

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import CreateAgentRunRequest
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.sandbox import SandboxManager
from neurons.validator.sandbox.models import SandboxErrorType
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.interval import (
    SCORING_WINDOW_INTERVALS,
    get_interval_start_minutes,
)
from neurons.validator.utils.logger.logger import NuminousLogger

TITLE_SEPARATOR = " ==Further Information==: "
MAX_LOG_CHARS = 25_000
MAX_TIMEOUT_RETRIES = 3


class RunAgents(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    sandbox_manager: SandboxManager
    subtensor_cm: AsyncSubtensor
    api_client: NuminousClient
    logger: NuminousLogger
    max_concurrent_sandboxes: int
    timeout_seconds: int
    sync_hour: int
    validator_uid: int
    validator_hotkey: str

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        sandbox_manager: SandboxManager,
        netuid: int,
        subtensor: AsyncSubtensor,
        api_client: NuminousClient,
        logger: NuminousLogger,
        max_concurrent_sandboxes: int = 5,
        timeout_seconds: int = 600,
        sync_hour: int = 4,
        validator_uid: int = 0,
        validator_hotkey: str = "",
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(sandbox_manager, SandboxManager):
            raise TypeError("sandbox_manager must be an instance of SandboxManager.")

        if not isinstance(netuid, int) or netuid < 0:
            raise ValueError("netuid must be a non-negative integer.")

        if not isinstance(subtensor, AsyncSubtensor):
            raise TypeError("subtensor must be an instance of AsyncSubtensor.")
        if not isinstance(api_client, NuminousClient):
            raise TypeError("api_client must be an instance of NuminousClient.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        if not isinstance(max_concurrent_sandboxes, int) or max_concurrent_sandboxes < 1:
            raise ValueError("max_concurrent_sandboxes must be an integer >= 1.")

        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer.")

        if not isinstance(validator_uid, int) or validator_uid < 0 or validator_uid > 256:
            raise ValueError("validator_uid must be a positive integer.")

        if not isinstance(validator_hotkey, str):
            raise TypeError("validator_hotkey must be a string.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.sandbox_manager = sandbox_manager
        self.netuid = netuid
        self.subtensor_cm = subtensor
        self.api_client = api_client
        self.logger = logger
        self.max_concurrent_sandboxes = max_concurrent_sandboxes
        self.timeout_seconds = timeout_seconds
        self.sync_hour = sync_hour
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey

        self.logger.debug(
            "RunAgents task initialized",
            extra={
                "max_concurrent": self.max_concurrent_sandboxes,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    @property
    def name(self) -> str:
        return "run-agents"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        async with self.subtensor_cm as subtensor:
            self.metagraph = await subtensor.metagraph(netuid=self.netuid, lite=True)

        current_hour_utc = datetime.now(timezone.utc).hour

        if current_hour_utc < self.sync_hour:
            self.logger.debug(
                "Before execution window",
                extra={"current_hour": current_hour_utc, "sync_hour": self.sync_hour},
            )
            return

        block = self.metagraph.block.item()
        self.logger.debug(
            "Synced metagraph", extra={"block": block, "neurons": len(self.metagraph.uids)}
        )

        events = await self.db_operations.get_events_to_predict(
            days_until_cutoff=SCORING_WINDOW_INTERVALS
        )
        if not len(events):
            self.logger.debug("No events to predict")
            return

        agents = await self.db_operations.get_active_agents()
        if not len(agents):
            self.logger.warning("No agents available for execution")
            return

        valid_agents = self.filter_agents_by_metagraph(agents)
        if not len(valid_agents):
            self.logger.warning("No valid agents after metagraph filtering")
            return

        interval_start_minutes = get_interval_start_minutes()

        self.logger.info(
            "Starting to run agents",
            extra={
                "scoring_window_intervals": SCORING_WINDOW_INTERVALS,
                "total_events": len(events),
                "total_agents": len(valid_agents),
                "interval_start_minutes": interval_start_minutes,
            },
        )

        await self.execute_all(events, valid_agents, interval_start_minutes)

    def filter_agents_by_metagraph(self, agents: List[MinerAgentsModel]) -> List[MinerAgentsModel]:
        valid_agents = []
        metagraph_uids = {int(uid): uid for uid in self.metagraph.uids}

        for agent in agents:
            if agent.miner_uid not in metagraph_uids:
                self.logger.debug(
                    "Skipping agent - UID not in metagraph",
                    extra={"agent_version_id": agent.version_id, "miner_uid": agent.miner_uid},
                )
                continue

            axon = self.metagraph.axons[agent.miner_uid]
            if axon is None:
                self.logger.debug(
                    "Skipping agent - no axon found",
                    extra={"agent_version_id": agent.version_id, "miner_uid": agent.miner_uid},
                )
                continue

            if axon.hotkey != agent.miner_hotkey:
                self.logger.debug(
                    "Skipping agent - hotkey mismatch",
                    extra={
                        "agent_version_id": agent.version_id,
                        "agent_hotkey": agent.miner_hotkey,
                        "metagraph_hotkey": axon.hotkey,
                    },
                )
                continue

            valid_agents.append(agent)

        self.logger.debug(
            "Filtered agents by metagraph",
            extra={
                "total_agents": len(agents),
                "valid_agents": len(valid_agents),
                "filtered_out": len(agents) - len(valid_agents),
            },
        )

        return valid_agents

    def parse_event_description(self, full_description: str) -> tuple[str, str]:
        # Fallback parser used only when legacy merged description format is present
        if TITLE_SEPARATOR in full_description:
            parts = full_description.split(TITLE_SEPARATOR, 1)
            return parts[0], parts[1]
        return full_description, full_description

    async def load_agent_code(self, agent: MinerAgentsModel) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            code = await loop.run_in_executor(None, Path(agent.file_path).read_text)
            return code
        except Exception as e:
            self.logger.error(
                "Failed to load agent code",
                extra={
                    "agent_version_id": agent.version_id,
                    "file_path": agent.file_path,
                    "error": str(e),
                },
            )
            return None

    async def run_sandbox(self, agent_code: str, event_data: dict, run_id: str) -> Optional[dict]:
        result_future = asyncio.Future()

        def on_finish(result):
            if not result_future.done():
                result_future.set_result(result)

        self.sandbox_manager.create_sandbox(
            agent_code=agent_code,
            event_data=event_data,
            run_id=run_id,
            on_finish=on_finish,
            timeout=self.timeout_seconds,
        )

        try:
            # Add 5s buffer to allow sandbox cleanup before timeout
            result = await asyncio.wait_for(result_future, timeout=self.timeout_seconds + 5)
            return result
        except asyncio.TimeoutError:
            self.logger.warning(
                "Sandbox execution timeout",
                extra={"run_id": run_id, "timeout": self.timeout_seconds},
            )
            return None

    async def store_prediction(
        self,
        event_id: str,
        agent: MinerAgentsModel,
        prediction_value: float,
        run_id: str,
        interval_start_minutes: int,
    ) -> None:
        try:
            clipped_value = max(0.0, min(1.0, prediction_value))

            prediction = PredictionsModel(
                unique_event_id=event_id,
                miner_uid=agent.miner_uid,
                miner_hotkey=agent.miner_hotkey,
                latest_prediction=clipped_value,
                interval_start_minutes=interval_start_minutes,
                interval_agg_prediction=clipped_value,
                run_id=run_id,
                version_id=agent.version_id,
            )

            await self.db_operations.upsert_predictions([prediction])

            self.logger.debug(
                "Stored prediction",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent.version_id,
                    "prediction": clipped_value,
                },
            )
        except Exception as e:
            self.logger.error(
                "Failed to store prediction",
                extra={"event_id": event_id, "agent_version_id": agent.version_id, "error": str(e)},
            )

    def _build_error_logs(self, logs: str, error_msg: str, traceback: Optional[str] = None) -> str:
        if "Timeout" in error_msg:
            logs += f"\n\n{'='*50}\nTIMEOUT\n{'='*50}\n"
            logs += "Execution exceeded timeout limit\n"
        else:
            logs += f"\n\n{'='*50}\nERROR DETAILS\n{'='*50}\n"
            logs += f"Error: {error_msg}\n"
            if traceback:
                logs += f"\nTraceback:\n{traceback}"
        return logs

    def _determine_status_and_extract_prediction(
        self,
        result: Optional[dict],
        event_id: str,
        agent_version_id: str,
        run_id: str,
    ) -> tuple[AgentRunStatus, Optional[float]]:
        if result is None:
            return (AgentRunStatus.SANDBOX_TIMEOUT, None)

        if not isinstance(result, dict):
            self.logger.warning(
                "Invalid result type from sandbox",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent_version_id,
                    "result_type": type(result).__name__,
                },
            )
            return (AgentRunStatus.INVALID_SANDBOX_OUTPUT, None)

        # Handle error status from sandbox
        if result.get("status") == "error":
            error_type = result.get("error_type")
            self.logger.warning(
                "Agent execution failed",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent_version_id,
                    "run_id": run_id,
                    "error": result.get("error", "Unknown error"),
                    "error_type": error_type,
                },
            )

            if error_type == SandboxErrorType.TIMEOUT:
                return (AgentRunStatus.SANDBOX_TIMEOUT, None)
            elif error_type == SandboxErrorType.CONTAINER_ERROR:
                return (AgentRunStatus.SANDBOX_TIMEOUT, None)
            elif error_type == SandboxErrorType.INVALID_OUTPUT:
                return (AgentRunStatus.INVALID_SANDBOX_OUTPUT, None)
            elif error_type == SandboxErrorType.AGENT_ERROR:
                return (AgentRunStatus.INTERNAL_AGENT_ERROR, None)
            else:
                self.logger.error(
                    "Unknown error_type from sandbox, defaulting to INTERNAL_AGENT_ERROR",
                    extra={"error_type": error_type, "error": result.get("error")},
                )
                return (AgentRunStatus.INTERNAL_AGENT_ERROR, None)

        # Handle success status - validate output structure
        output = result.get("output")
        if not isinstance(output, dict):
            self.logger.warning(
                "Invalid output format from sandbox",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent_version_id,
                    "result": str(result),
                },
            )
            return (AgentRunStatus.INVALID_SANDBOX_OUTPUT, None)

        if "prediction" not in output:
            self.logger.warning(
                "Missing prediction field in output",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent_version_id,
                    "output": str(output),
                },
            )
            return (AgentRunStatus.INVALID_SANDBOX_OUTPUT, None)

        prediction_value = output["prediction"]
        if not isinstance(prediction_value, (int, float)):
            self.logger.warning(
                "Invalid prediction value type",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent_version_id,
                    "prediction_value": str(prediction_value),
                    "type": type(prediction_value).__name__,
                },
            )
            return (AgentRunStatus.INVALID_SANDBOX_OUTPUT, None)

        return (AgentRunStatus.SUCCESS, float(prediction_value))

    async def _create_agent_run(
        self,
        run_id: str,
        event_id: str,
        agent: MinerAgentsModel,
        status: AgentRunStatus,
    ) -> AgentRunsModel:
        if status != AgentRunStatus.SANDBOX_TIMEOUT:
            is_final = True
        else:
            timeout_count = await self.db_operations.count_runs_for_event_and_agent(
                unique_event_id=event_id,
                agent_version_id=agent.version_id,
                status=AgentRunStatus.SANDBOX_TIMEOUT,
                is_final=False,
            )
            is_final = timeout_count >= MAX_TIMEOUT_RETRIES - 1

        return AgentRunsModel(
            run_id=run_id,
            unique_event_id=event_id,
            agent_version_id=agent.version_id,
            miner_uid=agent.miner_uid,
            miner_hotkey=agent.miner_hotkey,
            status=status,
            exported=False,
            is_final=is_final,
        )

    async def execute_agent_for_event(
        self,
        event_id: str,
        agent: MinerAgentsModel,
        event_tuple: tuple,
        interval_start_minutes: int,
    ) -> None:
        (
            unique_event_id_to_remove,
            external_event_id,
            market_type_to_remove,
            event_type,
            title,
            description,
            cutoff,
            metadata,
        ) = event_tuple

        try:
            create_run_request = CreateAgentRunRequest(
                miner_uid=agent.miner_uid,
                miner_hotkey=agent.miner_hotkey,
                vali_uid=self.validator_uid,
                vali_hotkey=self.validator_hotkey,
                event_id=event_id,
                version_id=agent.version_id,
            )

            create_run_response = await self.api_client.create_agent_run(create_run_request)
            run_id = str(create_run_response.run_id)

            self.logger.debug(
                "Created agent run via API",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent.version_id,
                    "miner_uid": agent.miner_uid,
                    "run_id": run_id,
                },
            )
        except Exception as e:
            self.logger.error(
                "Failed to create agent run via API",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent.version_id,
                    "error": str(e),
                },
            )
            return

        agent_code = await self.load_agent_code(agent)
        if agent_code is None:
            self.logger.error(
                "Cannot execute agent - failed to load code",
                extra={"event_id": event_id, "agent_version_id": agent.version_id},
            )
            return

        # Backward compatibility: if title is missing/empty, try to parse it from description
        if not title:
            title, description = self.parse_event_description(description)
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata

        event_data = {
            "event_id": external_event_id,
            "event_type": event_type,
            "title": title,
            "description": description,
            "cutoff": cutoff,
            "metadata": metadata,
        }

        result = await self.run_sandbox(agent_code, event_data, run_id)

        if result is None:
            logs = "Sandbox timeout - no logs"
        else:
            logs = result.get("logs", "No logs available")
            if result.get("status") == "error":
                logs = self._build_error_logs(
                    logs, result.get("error", "Unknown error"), result.get("traceback")
                )

        original_length = len(logs)
        if original_length > MAX_LOG_CHARS:
            truncation_msg = (
                f"[LOG TRUNCATED: Original {original_length:,} chars, "
                f"showing last {MAX_LOG_CHARS:,} chars]\n\n"
            )
            logs = truncation_msg + logs[-MAX_LOG_CHARS:]

        run_status, prediction_value = self._determine_status_and_extract_prediction(
            result, event_id, agent.version_id, run_id
        )

        agent_run = await self._create_agent_run(
            run_id=run_id,
            event_id=event_id,
            agent=agent,
            status=run_status,
        )
        await self.db_operations.upsert_agent_runs([agent_run])

        try:
            await self.db_operations.insert_agent_run_log(run_id, logs)
        except Exception as e:
            self.logger.error(
                "Failed to store agent run log",
                extra={"run_id": run_id, "error": str(e), "log_content": logs},
                exc_info=True,
            )

        if run_status == AgentRunStatus.SUCCESS and prediction_value is not None:
            await self.store_prediction(
                event_id, agent, prediction_value, run_id, interval_start_minutes
            )
        else:
            self.logger.debug(
                "Agent execution completed with non-success status",
                extra={
                    "event_id": event_id,
                    "agent_version_id": agent.version_id,
                    "run_id": run_id,
                    "status": run_status.value,
                },
            )

    async def execute_all(
        self, events: List[tuple], agents: List[MinerAgentsModel], interval_start_minutes: int
    ) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrent_sandboxes)

        tasks = []
        for event in events:
            for agent in agents:
                task = self.execute_with_semaphore(semaphore, event, agent, interval_start_minutes)
                tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def replicate_prediction_to_interval(
        self,
        existing_prediction: PredictionsModel,
        interval_start_minutes: int,
    ) -> None:
        new_prediction = PredictionsModel(
            unique_event_id=existing_prediction.unique_event_id,
            miner_uid=existing_prediction.miner_uid,
            miner_hotkey=existing_prediction.miner_hotkey,
            latest_prediction=existing_prediction.latest_prediction,
            interval_start_minutes=interval_start_minutes,
            interval_agg_prediction=existing_prediction.latest_prediction,
            interval_count=1,
            run_id=existing_prediction.run_id,
            version_id=existing_prediction.version_id,
        )

        await self.db_operations.upsert_predictions([new_prediction])

    async def execute_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        event: tuple,
        agent: MinerAgentsModel,
        interval_start_minutes: int,
    ) -> None:
        async with semaphore:
            event_id = event[0]

            # Check if prediction exists for this (event, miner) in ANY interval
            existing_prediction = (
                await self.db_operations.get_latest_prediction_for_event_and_miner(
                    unique_event_id=event_id,
                    miner_uid=agent.miner_uid,
                    miner_hotkey=agent.miner_hotkey,
                )
            )

            if existing_prediction is not None:
                # Prediction exists - check if already in current interval
                if existing_prediction.interval_start_minutes == interval_start_minutes:
                    self.logger.debug(
                        "Skipping execution - prediction exists",
                        extra={
                            "event_id": event_id,
                            "agent_version_id": agent.version_id,
                            "miner_uid": agent.miner_uid,
                        },
                    )
                    return

                # Replicate to current interval
                await self.replicate_prediction_to_interval(
                    existing_prediction=existing_prediction,
                    interval_start_minutes=interval_start_minutes,
                )
                self.logger.debug(
                    "Replicated existing prediction to new interval",
                    extra={
                        "event_id": event_id,
                        "agent_version_id": agent.version_id,
                        "miner_uid": agent.miner_uid,
                        "from_interval": existing_prediction.interval_start_minutes,
                        "to_interval": interval_start_minutes,
                    },
                )
                return

            has_final_run = await self.db_operations.has_final_run(
                unique_event_id=event_id,
                agent_version_id=agent.version_id,
            )

            if has_final_run:
                self.logger.debug(
                    "Skipping execution - final run exists",
                    extra={
                        "event_id": event_id,
                        "agent_version_id": agent.version_id,
                    },
                )
                return

            await self.execute_agent_for_event(
                event_id=event_id,
                agent=agent,
                event_tuple=event,
                interval_start_minutes=interval_start_minutes,
            )
