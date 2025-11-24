import asyncio
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.numinous_client import GetAgentsResponse, MinerAgentWithCode
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.agent_storage import get_agent_file_path, save_agent_code
from neurons.validator.utils.logger.logger import NuminousLogger


class AgentAPIClient(Protocol):
    async def get_agents(self, offset: int, limit: int) -> GetAgentsResponse:
        ...


class PullAgents(AbstractTask):
    """Pull agent code from backend API and store locally."""

    interval: float
    api_client: AgentAPIClient
    db_operations: DatabaseOperations
    agents_base_dir: Path
    page_size: int
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        api_client: AgentAPIClient,
        db_operations: DatabaseOperations,
        agents_base_dir: Path,
        page_size: int,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive float")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(agents_base_dir, Path):
            raise TypeError("agents_base_dir must be an instance of Path.")

        if not isinstance(page_size, int) or page_size <= 0 or page_size > 100:
            raise ValueError("page_size must be 1-100")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.interval = interval_seconds
        self.api_client = api_client
        self.db_operations = db_operations
        self.agents_base_dir = agents_base_dir
        self.page_size = page_size
        self.logger = logger

        self.agents_base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "pull-agents"

    @property
    def interval_seconds(self) -> float:
        return self.interval

    async def run(self) -> None:
        total_pulled = 0
        total_failed = 0

        offset = 0

        while True:
            response = await self.api_client.get_agents(offset=offset, limit=self.page_size)

            agents_data = response.items

            if len(agents_data) == 0:
                break

            processed_agents = []
            for agent_data in agents_data:
                try:
                    agent_model = await self.process_agent(agent_data)
                    processed_agents.append(agent_model)
                    total_pulled += 1

                except Exception as e:
                    total_failed += 1
                    self.logger.error(
                        "Failed to process agent",
                        extra={
                            "version_id": str(agent_data.version_id),
                            "miner_uid": agent_data.miner_uid,
                            "error": str(e),
                        },
                    )
                    continue

            # Batch upsert (one DB call per page, not per agent)
            if processed_agents:
                await self.db_operations.upsert_miner_agents(processed_agents)

            self.logger.debug(
                "Agents batch processed",
                extra={
                    "batch_size": len(agents_data),
                    "offset": offset,
                    "pulled": total_pulled,
                    "failed": total_failed,
                },
            )

            if len(agents_data) < self.page_size:
                break

            offset += self.page_size

        if total_pulled > 0 or total_failed > 0:
            self.logger.info(
                "Pull agents completed",
                extra={
                    "agents_pulled": total_pulled,
                    "agents_failed": total_failed,
                },
            )

    async def process_agent(self, agent_data: MinerAgentWithCode) -> MinerAgentsModel:
        try:
            code_bytes = base64.b64decode(agent_data.code)
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {e}")

        file_path = get_agent_file_path(
            self.agents_base_dir,
            agent_data.miner_uid,
            agent_data.miner_hotkey,
            agent_data.version_id,
        )

        # Run sync I/O in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_agent_code, file_path, code_bytes)

        return MinerAgentsModel(
            version_id=str(agent_data.version_id),
            miner_uid=agent_data.miner_uid,
            miner_hotkey=agent_data.miner_hotkey,
            agent_name=agent_data.agent_name,
            version_number=agent_data.version_number,
            file_path=str(file_path),
            pulled_at=datetime.now(timezone.utc),
            created_at=agent_data.created_at,
        )
