import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from neurons.validator.utils.config import VALID_NETWORK_CONFIGS, ValidatorSettings, get_config

DEFAULT_DB_DIRECTORY = str(Path.cwd())


def run_get_config(argv: list[str], env: dict[str, str] | None = None) -> ValidatorSettings:
    with patch.dict("os.environ", env or {}, clear=False):
        with patch("sys.argv", ["validator.py"] + argv):
            return get_config()


class TestConfig:
    def test_prod_compose_command_line(self):
        settings = run_get_config(
            (
                "--netuid 6 --subtensor.network finney"
                " --wallet.name wallet_name --wallet.hotkey wallet_hotkey"
                " --db.directory /root/infinite_games/database --numinous.env prod"
                " --sandbox.max_concurrent 50 --sandbox.timeout_seconds 240 --logging.debug"
            ).split()
        )

        assert settings.netuid == 6
        assert settings.subtensor_network == "finney"
        assert settings.subtensor_connection == "finney"
        assert settings.wallet_name == "wallet_name"
        assert settings.wallet_hotkey == "wallet_hotkey"
        assert settings.wallet_path == "~/.bittensor/wallets/"
        assert settings.numinous_env == "prod"
        assert settings.db_path == "/root/infinite_games/database/validator.db"
        assert settings.logging_level == logging.DEBUG
        assert settings.gateway_url == "https://numinous.earth"
        assert settings.validator_sync_hour == 1
        assert settings.sandbox_max_concurrent == 50
        assert settings.sandbox_timeout_seconds == 240

    def test_defaults(self):
        settings = run_get_config(["--netuid", "155", "--subtensor.network", "test"])

        assert settings.wallet_name == "default"
        assert settings.wallet_hotkey == "default"
        assert settings.wallet_path == "~/.bittensor/wallets/"
        assert settings.subtensor_chain_endpoint is None
        assert settings.db_path == str(Path(DEFAULT_DB_DIRECTORY) / "validator_test.db")
        assert settings.logging_level == logging.WARNING
        assert settings.validator_sync_hour == 1
        assert settings.sandbox_max_concurrent == 50
        assert settings.sandbox_timeout_seconds == 240

    def test_environment_variable_defaults(self):
        settings = run_get_config(
            ["--netuid", "155", "--subtensor.network", "test"],
            env={
                "BT_WALLET_NAME": "wallet_from_environment",
                "BT_WALLET_HOTKEY": "hotkey_from_environment",
                "BT_WALLET_PATH": "/wallets/from/environment",
                "BT_LOGGING_DEBUG": "1",
            },
        )

        assert settings.wallet_name == "wallet_from_environment"
        assert settings.wallet_hotkey == "hotkey_from_environment"
        assert settings.wallet_path == "/wallets/from/environment"
        assert settings.logging_level == logging.DEBUG

    def test_explicit_flag_overrides_environment_variable(self):
        settings = run_get_config(
            ["--netuid", "155", "--subtensor.network", "test", "--wallet.name", "from_flag"],
            env={"BT_WALLET_NAME": "from_environment"},
        )

        assert settings.wallet_name == "from_flag"

    def test_chain_endpoint_overrides_network_for_connection(self):
        settings = run_get_config(
            [
                "--netuid",
                "155",
                "--subtensor.network",
                "test",
                "--subtensor.chain_endpoint",
                "ws://127.0.0.1:9944",
            ]
        )

        assert settings.subtensor_network == "test"
        assert settings.subtensor_connection == "ws://127.0.0.1:9944"

    def test_test_env_gateway_url_defaults_to_local_backend(self):
        settings = run_get_config(["--netuid", "155", "--subtensor.network", "test"])

        assert settings.gateway_url in (
            "http://172.17.0.1:8000",
            "http://host.docker.internal:8000",
        )

    def test_gateway_url_argument_wins(self):
        settings = run_get_config(
            [
                "--netuid",
                "155",
                "--subtensor.network",
                "test",
                "--backend.gateway_url",
                "https://stg.numinous.earth",
            ]
        )

        assert settings.gateway_url == "https://stg.numinous.earth"

    def test_required_args_missing(self):
        with pytest.raises(ValueError):
            run_get_config([])

    def test_db_directory_validation(self):
        settings = run_get_config(
            ["--netuid", "6", "--subtensor.network", "finney", "--db.directory", "/parent/db_dir"]
        )

        assert settings.numinous_env == "prod"
        assert settings.db_path == str(Path("/parent/db_dir") / "validator.db")

        with pytest.raises(ValueError, match="Invalid db.directory './' must be an absolute path."):
            run_get_config(
                ["--netuid", "6", "--subtensor.network", "finney", "--db.directory", "./"]
            )

    @pytest.mark.parametrize("test_config", VALID_NETWORK_CONFIGS)
    def test_valid_network_configs_constant(self, test_config: dict):
        assert isinstance(test_config, dict)
        assert "subtensor.network" in test_config
        assert "netuid" in test_config
        assert "numinous.env" in test_config
        assert isinstance(test_config["netuid"], int)
        assert test_config["subtensor.network"] in ["finney", "test", "local", None]
        assert test_config["numinous.env"] in ["prod", "test", None]

    @pytest.mark.parametrize(
        "logging_flags,expected_logger_level",
        [
            (["--logging.trace", "--logging.debug", "--logging.info"], logging.DEBUG),
            (["--logging.trace"], logging.DEBUG),
            (["--logging.debug"], logging.DEBUG),
            (["--logging.info"], logging.INFO),
            ([], logging.WARNING),
        ],
    )
    def test_logger_args(self, logging_flags: list[str], expected_logger_level: int):
        settings = run_get_config(
            ["--netuid", "6", "--subtensor.network", "finney"] + logging_flags
        )

        assert settings.logging_level == expected_logger_level

    @pytest.mark.parametrize(
        "test_args,expected_env,expected_db_path",
        [
            ((6, "finney", None, "/p1/p2"), "prod", "/p1/p2/validator.db"),
            ((155, "test", None, "/p1/"), "test", "/p1/validator_test.db"),
            ((6, "local", "prod", None), "prod", str(Path(DEFAULT_DB_DIRECTORY) / "validator.db")),
            ((155, "local", "test", "/"), "test", "/validator_test.db"),
            ((6, None, "prod", None), "prod", str(Path(DEFAULT_DB_DIRECTORY) / "validator.db")),
            (
                (155, None, "test", None),
                "test",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator_test.db"),
            ),
            (
                (6, "ws://fake_ip:fake_port", "prod", None),
                "prod",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator.db"),
            ),
            (
                (155, "ws://fake_ip:fake_port", "test", None),
                "test",
                str(Path(DEFAULT_DB_DIRECTORY) / "validator_test.db"),
            ),
        ],
    )
    def test_valid_configurations(self, test_args: tuple, expected_env: str, expected_db_path: str):
        netuid, network, numinous_env, db_directory = test_args

        argv = ["--netuid", str(netuid)]
        if network is not None:
            argv += ["--subtensor.network", network]
        if numinous_env is not None:
            argv += ["--numinous.env", numinous_env]
        if db_directory is not None:
            argv += ["--db.directory", db_directory]

        settings = run_get_config(argv)

        assert settings.numinous_env == expected_env
        assert settings.db_path == expected_db_path

    @pytest.mark.parametrize(
        "test_args",
        [
            (6, "invalid", None),
            (6, "local", None),
            (155, None, "prod"),
            (6, None, "test"),
            (6, "ws://fake_ip:fake_port", "test"),
            (155, "ws://fake_ip:fake_port", "prod"),
        ],
    )
    def test_invalid_configurations(self, test_args: tuple):
        netuid, network, numinous_env = test_args

        argv = ["--netuid", str(netuid)]
        if network is not None:
            argv += ["--subtensor.network", network]
        if numinous_env is not None:
            argv += ["--numinous.env", numinous_env]

        with pytest.raises(ValueError) as exc_info:
            run_get_config(argv)

        assert "Invalid netuid" in str(exc_info.value)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--netuid", "7", "--subtensor.network", "finney"],
            ["--netuid", "155", "--subtensor.network", "local", "--numinous.env", "invalid"],
        ],
    )
    def test_rejected_by_parser(self, argv: list[str]):
        with pytest.raises(SystemExit):
            run_get_config(argv)
