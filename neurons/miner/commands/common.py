import functools
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import NoReturn, TextIO

import click
import httpx
from pydantic import JsonValue

from neurons.miner.client import (
    AlreadySubmitted,
    BackendRejected,
    DeadlinePassed,
    MinerClient,
    NoRoundAccepting,
    NotAQuestionOfTheRound,
    ReceiptInvalid,
)
from neurons.miner.models import ForecastError
from neurons.miner.wallet import DEFAULT_WALLET_PATH, HotkeyNotFound, load_hotkey


class Moment(click.ParamType):
    name = "datetime"

    def convert(
        self, value: str, parameter: click.Parameter | None, context: click.Context | None
    ) -> datetime:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            self.fail(
                f"{value!r} is not an ISO 8601 date-time, e.g. 2026-10-13T17:00:00Z",
                parameter,
                context,
            )
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class ExitCode(IntEnum):
    OK = 0
    NO_ROUND_ACCEPTING = 3
    INVALID_FORECAST = 4
    BACKEND_REJECTED = 5
    BACKEND_UNREACHABLE = 6
    HOTKEY_NOT_FOUND = 7
    ALREADY_SUBMITTED = 8
    DEADLINE_PASSED = 9
    NOT_A_QUESTION_OF_THE_ROUND = 10
    RECEIPT_INVALID = 11


_FAILURES: dict[type[Exception], ExitCode] = {
    HotkeyNotFound: ExitCode.HOTKEY_NOT_FOUND,
    NoRoundAccepting: ExitCode.NO_ROUND_ACCEPTING,
    BackendRejected: ExitCode.BACKEND_REJECTED,
    AlreadySubmitted: ExitCode.ALREADY_SUBMITTED,
    DeadlinePassed: ExitCode.DEADLINE_PASSED,
    NotAQuestionOfTheRound: ExitCode.NOT_A_QUESTION_OF_THE_ROUND,
    ReceiptInvalid: ExitCode.RECEIPT_INVALID,
}


def emit(payload: JsonValue) -> None:
    click.echo(json.dumps(payload, indent=2))


def fail(code: ExitCode, message: str) -> NoReturn:
    emit({"error": code.name.lower(), "message": message})
    sys.exit(code)


def read_forecast(forecast_file: TextIO) -> JsonValue:
    try:
        return json.load(forecast_file)
    except json.JSONDecodeError as error:
        fail(ExitCode.INVALID_FORECAST, f"not JSON: {error}")


def reject_forecast(errors: list[ForecastError]) -> NoReturn:
    emit({"valid": False, "errors": [error.model_dump() for error in errors]})
    sys.exit(ExitCode.INVALID_FORECAST)


def with_client(command: Callable[..., None]) -> Callable[..., None]:
    @click.option(
        "--env",
        type=click.Choice(["test", "prod"]),
        envvar="NUMINOUS_ENV",
        default="prod",
        show_default=True,
        help="Backend to talk to. Env var NUMINOUS_ENV.",
    )
    @click.option(
        "--wallet",
        envvar="NUMINOUS_WALLET",
        required=True,
        help="Wallet name. Env var NUMINOUS_WALLET.",
    )
    @click.option(
        "--hotkey",
        envvar="NUMINOUS_HOTKEY",
        default="default",
        show_default=True,
        help="Hotkey name in the wallet. Env var NUMINOUS_HOTKEY.",
    )
    @click.option(
        "--wallet-path",
        envvar="NUMINOUS_WALLET_PATH",
        type=click.Path(file_okay=False, path_type=Path),
        default=DEFAULT_WALLET_PATH,
        show_default=True,
        help="Folder holding the wallets. Env var NUMINOUS_WALLET_PATH.",
    )
    @functools.wraps(command)
    def wrapper(env: str, wallet: str, hotkey: str, wallet_path: Path, **arguments) -> None:
        try:
            client = MinerClient(env, load_hotkey(wallet, hotkey, wallet_path))
            command(client, **arguments)
        except tuple(_FAILURES) as error:
            fail(_FAILURES[type(error)], str(error))
        except httpx.HTTPError as error:
            fail(ExitCode.BACKEND_UNREACHABLE, f"{type(error).__name__}: {error}")

    return wrapper
