import asyncio
import sqlite3
import sys

from bittensor import Subtensor, Wallet

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.tasks.db_cleaner import DbCleaner
from neurons.validator.tasks.db_vacuum import DbVacuum
from neurons.validator.tasks.post_dkg_scores import PostDkgScores
from neurons.validator.tasks.pull_dkg_results import PullDkgResults
from neurons.validator.tasks.pull_dkg_rounds import PullDkgRounds
from neurons.validator.tasks.score_dkg_miners import ScoreDkgMiners
from neurons.validator.tasks.set_weights import SetWeights
from neurons.validator.tasks.sync_miners_metadata import SyncMinersMetadata
from neurons.validator.utils.common.event_loop import measure_event_loop_lag
from neurons.validator.utils.config import get_config
from neurons.validator.utils.logger.logger import (
    logger,
    override_loggers_level,
    set_bittensor_logger,
)


async def main():
    # Start session id
    logger.start_session()

    settings = get_config()

    # Loggers
    override_loggers_level(settings.logging_level)
    set_bittensor_logger()

    # Bittensor stuff
    bt_netuid = settings.netuid
    bt_network = settings.subtensor_network
    bt_wallet = Wallet(
        name=settings.wallet_name, hotkey=settings.wallet_hotkey, path=settings.wallet_path
    )

    async with Subtensor(settings.subtensor_connection) as chain_client:
        metagraph = await chain_client.subnets.metagraph(bt_netuid)

        validator_hotkey = bt_wallet.hotkey.ss58_address
        validator_uid = metagraph.hotkeys.index(validator_hotkey)

    # Components
    db_client = DatabaseClient(db_path=settings.db_path, logger=logger)
    dkg_operations = DkgDatabaseOperations(db_client=db_client)
    dkg_client = DkgClient(env=settings.numinous_env, logger=logger, bt_wallet=bt_wallet)

    # Migrate db
    await db_client.migrate()

    # Tasks
    sync_miners_task = SyncMinersMetadata(
        interval_seconds=300.0,
        dkg_operations=dkg_operations,
        netuid=bt_netuid,
        network=settings.subtensor_connection,
        logger=logger,
    )

    pull_dkg_rounds_task = PullDkgRounds(
        interval_seconds=600.0,
        dkg_operations=dkg_operations,
        dkg_client=dkg_client,
        logger=logger,
    )

    pull_dkg_results_task = PullDkgResults(
        interval_seconds=900.0,
        dkg_operations=dkg_operations,
        dkg_client=dkg_client,
        logger=logger,
    )

    score_dkg_miners_task = ScoreDkgMiners(
        interval_seconds=300.0,
        dkg_operations=dkg_operations,
        logger=logger,
    )

    post_dkg_scores_task = PostDkgScores(
        interval_seconds=300.0,
        dkg_operations=dkg_operations,
        dkg_client=dkg_client,
        logger=logger,
    )

    set_weights_task = SetWeights(
        interval_seconds=379.0,
        dkg_operations=dkg_operations,
        logger=logger,
        netuid=bt_netuid,
        network=settings.subtensor_connection,
        wallet=bt_wallet,
    )

    db_cleaner_task = DbCleaner(
        interval_seconds=3600.0,
        dkg_operations=dkg_operations,
        batch_size=4000,
        logger=logger,
    )

    vacuum_task = DbVacuum(
        interval_seconds=300.0,
        dkg_operations=dkg_operations,
        logger=logger,
        pages=500,
    )

    # Add tasks to scheduler
    scheduler = TasksScheduler(logger=logger)

    scheduler.add(task=sync_miners_task)
    scheduler.add(task=pull_dkg_rounds_task)
    scheduler.add(task=pull_dkg_results_task)
    scheduler.add(task=score_dkg_miners_task)
    scheduler.add(task=post_dkg_scores_task)
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
            "numinous_env": settings.numinous_env,
            "db_path": settings.db_path,
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
        },
    )

    await asyncio.gather(scheduler_task, loop_lag_task)
