import argparse
import logging
from pathlib import Path
from typing import Literal
from unittest.mock import ANY

from bittensor import AsyncSubtensor
from bittensor.core.config import Config
from bittensor.utils.btlogging import LoggingMachine
from bittensor_wallet.wallet import Wallet

NuminousEnvType = Literal["test", "prod"]

VALID_NETWORK_CONFIGS = [
    {"subtensor.network": "finney", "netuid": 6, "ifgames.env": None},
    {"subtensor.network": "test", "netuid": 155, "ifgames.env": None},
    {"subtensor.network": "local", "netuid": 6, "ifgames.env": "prod"},
    {"subtensor.network": "local", "netuid": 155, "ifgames.env": "test"},
    {"subtensor.network": ANY, "netuid": 6, "ifgames.env": "prod"},
    {"subtensor.network": ANY, "netuid": 155, "ifgames.env": "test"},
]


def get_config():
    # Build parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--netuid", type=int, help="Subnet netuid", choices=[6, 155], required=False
    )
    parser.add_argument(
        "--ifgames.env", type=str, help="IFGames env", choices=["prod", "test"], required=False
    )
    parser.add_argument(
        "--db.directory",
        type=str,
        help="Directory where the database file is located (default: ./). This must be an absolute directory path that exists",
        required=False,
    )
    parser.add_argument(
        "--backend.gateway_url",
        type=str,
        help="Backend gateway URL for sandbox API access (optional, defaults to ifgames backend based on env)",
        required=False,
    )
    parser.add_argument(
        "--sandbox.max_concurrent",
        type=int,
        default=50,
        help="Maximum concurrent sandbox executions (default: 50)",
    )
    parser.add_argument(
        "--sandbox.timeout_seconds",
        type=int,
        default=180,
        help="Timeout for agent execution in seconds (default: 180)",
    )

    AsyncSubtensor.add_args(parser=parser)
    LoggingMachine.add_args(parser=parser)
    Wallet.add_args(parser=parser)

    # Read args
    args = parser.parse_args()
    netuid = args.__getattribute__("netuid")
    network = args.__getattribute__("subtensor.network")
    numinous_env = args.__getattribute__("ifgames.env")
    logging_trace = args.__getattribute__("logging.trace")
    logging_debug = args.__getattribute__("logging.debug")
    logging_info = args.__getattribute__("logging.info")

    # Set default, __getattribute__ doesn't return arguments defaults
    db_directory = args.__getattribute__("db.directory") or str(Path.cwd())

    # Validate network config
    if not any(
        [
            netuid == config["netuid"]
            and network == config["subtensor.network"]
            and numinous_env == config["ifgames.env"]
            for config in VALID_NETWORK_CONFIGS
        ]
    ):
        raise ValueError(
            (
                f"Invalid netuid {netuid}, subtensor.network '{network}' and ifgames.env '{numinous_env}' combination.\n"
                f"Valid combinations are:\n"
                f"{chr(10).join(map(str, VALID_NETWORK_CONFIGS))}"
            )
        )

    # Validate db directory
    if not Path(db_directory).is_absolute():
        raise ValueError(f"Invalid db.directory '{db_directory}' must be an absolute path.")

    config = Config(parser=parser, strict=True)
    env = "prod" if netuid == 6 else "test"
    db_filename = "validator.db" if env == "prod" else "validator_test.db"
    db_path = str(Path(db_directory) / db_filename)

    gateway_url = config.get("backend", {}).get("gateway_url")
    if not gateway_url:
        gateway_url = "https://ifgames.win" if env == "prod" else "http://host.docker.internal:8000"

    logging_level: int = logging.WARNING
    if logging_trace or logging_debug:
        logging_level = logging.DEBUG
    elif logging_info:
        logging_level = logging.INFO

    return config, env, db_path, logging_level, gateway_url
