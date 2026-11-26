from typing import TextIO

import click

from neurons.miner.client import MinerClient, forecast_errors
from neurons.miner.commands.common import emit, read_forecast, reject_forecast, with_client


@click.command(name="validate")
@click.argument("forecast_file", type=click.File("r"), default="-")
@with_client
def validate_command(client: MinerClient, forecast_file: TextIO) -> None:
    """Check a forecast against the schema the backend opens it with.

    \b
    The forecast is JSON, read from FORECAST_FILE, or from stdin when no file
    is given. Prints {"valid": true}, or {"valid": false, "errors": [{"path",
    "message"}]} and exits 4.

    \b
    Each statement carries a "distribution" of 31 numbers from 0 to 1: the
    chance it first appears on day 1, 2, ... 30 after the deadline, then the
    chance it does not appear within 30 days. They are divided by their total,
    so they need not add up to 1, but at least one has to be above 0.

    \b
    A statement can carry an optional "place", the town or city where the act
    happens, e.g. {"head": "Israel", "relation": "ATTACKS", "tail": "Lebanon",
    "place": "Mansouri", "distribution": [...]}. When given, it has to match.

    \b
    Inline, no file:
      numi validate <<'EOF'
      {"version": 3, "statements": [{"head": "Apple", "relation": "INVESTS_IN",
       "tail": "TSMC", "distribution": [0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.85]}]}
      EOF

    \b
    From a file:
      numi validate forecast.json --env test --wallet my_wallet --hotkey my_hotkey
    """
    errors = forecast_errors(read_forecast(forecast_file), client.get_schema())
    if errors:
        reject_forecast(errors)

    emit({"valid": True})
