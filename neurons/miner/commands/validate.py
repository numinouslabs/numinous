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
    Inline, no file:
      numi validate <<'EOF'
      {"version": 2, "statements": [{"head": "Apple", "relation": "INVESTS_IN",
       "tail": "TSMC", "probability": 0.15}]}
      EOF

    \b
    From a file:
      numi validate forecast.json --env test --wallet my_wallet --hotkey my_hotkey
    """
    errors = forecast_errors(read_forecast(forecast_file), client.get_schema())
    if errors:
        reject_forecast(errors)

    emit({"valid": True})
