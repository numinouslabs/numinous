import asyncio
import sqlite3
import sys
from pathlib import Path

from bittensor import AsyncSubtensor
from bittensor_wallet import Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.sandbox import SandboxManager
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.tasks.delete_events import DeleteEvents
from neurons.validator.tasks.export_agent_run_logs import ExportAgentRunLogs
from neurons.validator.tasks.export_agent_runs import ExportAgentRuns
from neurons.validator.tasks.export_predictions import ExportPredictions
from neurons.validator.tasks.export_scores import ExportScores
from neurons.validator.tasks.metagraph_scoring import MetagraphScoring
from neurons.validator.tasks.pull_agents import PullAgents
from neurons.validator.tasks.pull_events import PullEvents
from neurons.validator.tasks.resolve_events import ResolveEvents
from neurons.validator.tasks.run_agents import RunAgents
from neurons.validator.tasks.scoring import Scoring
from neurons.validator.tasks.set_weights import SetWeights
from neurons.validator.tasks.sync_miners_metadata import SyncMinersMetadata
from neurons.validator.utils.common.event_loop import measure_event_loop_lag
from neurons.validator.utils.config import get_config
from neurons.validator.utils.env import assert_requirements
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import (
    logger,
    override_loggers_level,
    set_bittensor_logger,
)


async def main():
    # Assert system requirements
    assert_requirements()

    # Start session id
    logger.start_session()

    config, numinous_env, db_path, logging_level, gateway_url, validator_sync_hour = get_config()

    # Loggers
    override_loggers_level(logging_level)
    set_bittensor_logger()

    # Bittensor stuff
    bt_netuid = config.get("netuid")
    bt_network = config.get("subtensor").get("network")
    bt_wallet = Wallet(config=config)
    bt_subtensor = AsyncSubtensor(config=config)
    bt_metagraph = IfMetagraph(netuid=bt_netuid, network=bt_network)

    # Sync metagraph for reading validator info
    await bt_metagraph.sync()

    validator_hotkey = bt_wallet.hotkey.ss58_address
    validator_uid = bt_metagraph.hotkeys.index(validator_hotkey)

    # Components
    db_client = DatabaseClient(db_path=db_path, logger=logger)
    db_operations = DatabaseOperations(db_client=db_client, logger=logger)
    numinous_api_client = NuminousClient(env=numinous_env, logger=logger, bt_wallet=bt_wallet)

    # Migrate db
    await db_client.migrate()

    # Tasks
    pull_events_task = PullEvents(
        interval_seconds=50.0,
        page_size=50,
        db_operations=db_operations,
        api_client=numinous_api_client,
    )

    resolve_events_task = ResolveEvents(
        interval_seconds=900.0,
        db_operations=db_operations,
        api_client=numinous_api_client,
        page_size=100,
        logger=logger,
    )

    delete_events_task = DeleteEvents(
        interval_seconds=1800.0,
        db_operations=db_operations,
        api_client=numinous_api_client,
        page_size=100,
        logger=logger,
    )

    pull_agents_task = PullAgents(
        interval_seconds=300.0,
        api_client=numinous_api_client,
        db_operations=db_operations,
        agents_base_dir=Path(__file__).parent.parent.parent / "data" / "agents",
        page_size=50,
        logger=logger,
    )

    sync_miners_task = SyncMinersMetadata(
        interval_seconds=300.0,
        db_operations=db_operations,
        metagraph=bt_metagraph,
        logger=logger,
    )

    sandbox_temp_dir = Path("/tmp") / "ig_validator_sandboxes"
    sandbox_temp_dir.mkdir(parents=True, exist_ok=True)

    sandbox_manager = SandboxManager(
        bt_wallet=bt_wallet,
        gateway_url=gateway_url,
        logger=logger,
        log_docker_to_stdout=True,
        temp_base_dir=sandbox_temp_dir,
    )

    run_agents_task = RunAgents(
        interval_seconds=600.0,
        db_operations=db_operations,
        api_client=numinous_api_client,
        sandbox_manager=sandbox_manager,
        metagraph=bt_metagraph,
        logger=logger,
        max_concurrent_sandboxes=config.get("sandbox", {}).get("max_concurrent", 50),
        timeout_seconds=config.get("sandbox", {}).get("timeout_seconds", 180),
        sync_hour=validator_sync_hour,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
    )

    export_predictions_task = ExportPredictions(
        interval_seconds=180.0,
        db_operations=db_operations,
        api_client=numinous_api_client,
        batch_size=300,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
        logger=logger,
    )

    scoring_task = Scoring(
        interval_seconds=307.0,
        db_operations=db_operations,
        metagraph=bt_metagraph,
        logger=logger,
        page_size=100,
    )

    metagraph_scoring_task = MetagraphScoring(
        interval_seconds=347.0,
        db_operations=db_operations,
        page_size=1000,
        logger=logger,
        metagraph=bt_metagraph,
    )

    export_scores_task = ExportScores(
        interval_seconds=373.0,
        page_size=500,
        db_operations=db_operations,
        api_client=numinous_api_client,
        logger=logger,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
    )

    export_agent_runs_task = ExportAgentRuns(
        interval_seconds=300.0,
        batch_size=500,
        db_operations=db_operations,
        api_client=numinous_api_client,
        logger=logger,
        validator_uid=validator_uid,
        validator_hotkey=validator_hotkey,
    )

    export_agent_run_logs_task = ExportAgentRunLogs(
        interval_seconds=600.0,
        batch_size=500,
        db_operations=db_operations,
        api_client=numinous_api_client,
        logger=logger,
    )

    set_weights_task = SetWeights(
        interval_seconds=379.0,
        db_operations=db_operations,
        logger=logger,
        metagraph=bt_metagraph,
        netuid=bt_netuid,
        subtensor=bt_subtensor,
        wallet=bt_wallet,
        api_client=numinous_api_client,
    )

    db_cleaner_task = DbCleaner(
        interval_seconds=53.0, db_operations=db_operations, batch_size=4000, logger=logger
    )

    vacuum_task = DbVacuum(
        interval_seconds=300.0,
        db_operations=db_operations,
        logger=logger,
        pages=500,
    )

    # Add tasks to scheduler
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=pull_events_task)
    scheduler.add(task=pull_agents_task)
    scheduler.add(task=sync_miners_task)
    scheduler.add(task=resolve_events_task)
    scheduler.add(task=delete_events_task)
    scheduler.add(task=run_agents_task)
    scheduler.add(task=export_predictions_task)
    scheduler.add(task=export_agent_runs_task)
    scheduler.add(task=export_agent_run_logs_task)
    scheduler.add(task=scoring_task)
    scheduler.add(task=metagraph_scoring_task)
    scheduler.add(task=export_scores_task)
    scheduler.add(task=set_weights_task)
    scheduler.add(task=db_cleaner_task)
    scheduler.add(task=vacuum_task)

    # Start scheduler
    scheduler_task = asyncio.create_task(scheduler.start())

    # Measure event loop lag
    loop_lag_task = asyncio.create_task(
        measure_event_loop_lag(
            measuring_frequency_seconds=0.5, lag_threshold_seconds=1.0, logger=logger
        )
    )

    logger.info(
        "Validator started",
        extra={
            "validator_uid": validator_uid,
            "validator_hotkey": validator_hotkey,
            "bt_network": bt_network,
            "bt_netuid": bt_netuid,
            "numinous_env": numinous_env,
            "db_path": db_path,
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
        },
    )

    await asyncio.gather(scheduler_task, loop_lag_task)
