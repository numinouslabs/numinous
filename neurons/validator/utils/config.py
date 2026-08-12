import argparse
import logging
import os
import platform
from pathlib import Path
from typing import Literal
from unittest.mock import ANY

from pydantic import BaseModel

NuminousEnvType = Literal["test", "prod"]

VALID_NETWORK_CONFIGS = [
    {"subtensor.network": "finney", "netuid": 6, "numinous.env": None},
    {"subtensor.network": "test", "netuid": 155, "numinous.env": None},
    {"subtensor.network": "local", "netuid": 6, "numinous.env": "prod"},
    {"subtensor.network": "local", "netuid": 155, "numinous.env": "test"},
    {"subtensor.network": ANY, "netuid": 6, "numinous.env": "prod"},
    {"subtensor.network": ANY, "netuid": 155, "numinous.env": "test"},
]


class ValidatorSettings(BaseModel):
    netuid: int
    subtensor_network: str
    subtensor_chain_endpoint: str | None
    wallet_name: str
    wallet_hotkey: str
    wallet_path: str
    numinous_env: NuminousEnvType
    db_path: str
    logging_level: int
    gateway_url: str
    validator_sync_hour: int
    sandbox_max_concurrent: int
    sandbox_timeout_seconds: int

    @property
    def subtensor_connection(self) -> str:
        return self.subtensor_chain_endpoint or self.subtensor_network


def _environment_flag(variable_name: str) -> bool:
    return os.getenv(variable_name, "").lower() in ("1", "true", "yes", "on")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--netuid", type=int, help="Subnet netuid", choices=[6, 155], required=False
    )
    parser.add_argument(
        "--numinous.env", type=str, help="Numinous env", choices=["prod", "test"], required=False
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
        help="Backend gateway URL for sandbox API access (optional, defaults to numinous backend based on env)",
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
        default=240,
        help="Timeout for agent execution in seconds (default: 240)",
    )
    parser.add_argument(
        "--validator.sync_hour",
        type=int,
        default=1,
        help="Hour for validator synchronization (default: 1)",
    )
    parser.add_argument(
        "--subtensor.network",
        type=str,
        default=os.getenv("BT_SUBTENSOR_NETWORK", "finney"),
        help="The subtensor network: finney (mainnet), test, or local",
    )
    parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        default=os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT"),
        help="Websocket endpoint of a subtensor node. If set, overrides --subtensor.network",
    )
    parser.add_argument(
        "--wallet.name",
        type=str,
        default=os.getenv("BT_WALLET_NAME", "default"),
        help="The name of the wallet to use",
    )
    parser.add_argument(
        "--wallet.hotkey",
        type=str,
        default=os.getenv("BT_WALLET_HOTKEY", "default"),
        help="The name of the wallet's hotkey to use",
    )
    parser.add_argument(
        "--wallet.path",
        type=str,
        default=os.getenv("BT_WALLET_PATH", "~/.bittensor/wallets/"),
        help="The path to the directory containing the wallets",
    )
    parser.add_argument(
        "--logging.debug",
        action="store_true",
        default=_environment_flag("BT_LOGGING_DEBUG"),
        help="Enable debug logging",
    )
    parser.add_argument(
        "--logging.trace",
        action="store_true",
        default=_environment_flag("BT_LOGGING_TRACE"),
        help="Enable trace logging",
    )
    parser.add_argument(
        "--logging.info",
        action="store_true",
        default=_environment_flag("BT_LOGGING_INFO"),
        help="Enable info logging",
    )

    return parser


def _validate_network_config(netuid: int | None, network: str, numinous_env: str | None) -> None:
    if not any(
        [
            netuid == config["netuid"]
            and network == config["subtensor.network"]
            and numinous_env == config["numinous.env"]
            for config in VALID_NETWORK_CONFIGS
        ]
    ):
        raise ValueError(
            (
                f"Invalid netuid {netuid}, subtensor.network '{network}' and numinous.env '{numinous_env}' combination.\n"
                f"Valid combinations are:\n"
                f"{chr(10).join(map(str, VALID_NETWORK_CONFIGS))}"
            )
        )


def _resolve_db_path(db_directory_argument: str | None, numinous_env: NuminousEnvType) -> str:
    db_directory = db_directory_argument or str(Path.cwd())

    if not Path(db_directory).is_absolute():
        raise ValueError(f"Invalid db.directory '{db_directory}' must be an absolute path.")

    db_filename = "validator.db" if numinous_env == "prod" else "validator_test.db"

    return str(Path(db_directory) / db_filename)


def _resolve_gateway_url(gateway_url_argument: str | None, numinous_env: NuminousEnvType) -> str:
    if gateway_url_argument:
        return gateway_url_argument

    if numinous_env == "prod":
        return "https://numinous.earth"

    return (
        "http://172.17.0.1:8000"
        if platform.system() == "Linux"
        else "http://host.docker.internal:8000"
    )


def _resolve_logging_level(args: argparse.Namespace) -> int:
    if getattr(args, "logging.trace") or getattr(args, "logging.debug"):
        return logging.DEBUG

    if getattr(args, "logging.info"):
        return logging.INFO

    return logging.WARNING


def get_config() -> ValidatorSettings:
    parser = _build_parser()
    args = parser.parse_args()

    netuid = getattr(args, "netuid")
    network = getattr(args, "subtensor.network")

    _validate_network_config(
        netuid=netuid, network=network, numinous_env=getattr(args, "numinous.env")
    )

    numinous_env: NuminousEnvType = "prod" if netuid == 6 else "test"

    return ValidatorSettings(
        netuid=netuid,
        subtensor_network=network,
        subtensor_chain_endpoint=getattr(args, "subtensor.chain_endpoint"),
        wallet_name=getattr(args, "wallet.name"),
        wallet_hotkey=getattr(args, "wallet.hotkey"),
        wallet_path=getattr(args, "wallet.path"),
        numinous_env=numinous_env,
        db_path=_resolve_db_path(getattr(args, "db.directory"), numinous_env),
        logging_level=_resolve_logging_level(args),
        gateway_url=_resolve_gateway_url(getattr(args, "backend.gateway_url"), numinous_env),
        validator_sync_hour=getattr(args, "validator.sync_hour"),
        sandbox_max_concurrent=getattr(args, "sandbox.max_concurrent"),
        sandbox_timeout_seconds=getattr(args, "sandbox.timeout_seconds"),
    )
