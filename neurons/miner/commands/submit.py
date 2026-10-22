import json
import uuid
from pathlib import Path
from typing import TextIO

import click
from pydantic import JsonValue

from neurons.miner.client import ForecastNotValid, MinerClient
from neurons.miner.commands.common import emit, read_forecast, reject_forecast, with_client
from neurons.miner.models import Submission

RECEIPTS_FOLDER = Path.home() / ".numinous" / "receipts"


def save_receipt(submission: Submission, forecast: JsonValue) -> Path:
    receipt = submission.receipt
    receipt_file = (
        RECEIPTS_FOLDER / receipt.miner_hotkey / submission.round_id / f"{receipt.question_id}.json"
    )
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.write_text(
        json.dumps(submission.model_dump(mode="json") | {"forecast": forecast}, indent=2)
    )
    return receipt_file


@click.command(name="submit")
@click.option(
    "--question",
    "question_id",
    type=click.UUID,
    required=True,
    help="The question this forecast answers (an id from numi round).",
)
@click.argument("forecast_file", type=click.File("r"), default="-")
@with_client
def submit_command(client: MinerClient, question_id: uuid.UUID, forecast_file: TextIO) -> None:
    """Seal and send a forecast for one question of the round accepting forecasts.

    \b
    Checks the forecast against the schema and that the question is in the
    open round, seals it to the round's reveal_round (nobody can read it before
    the deadline, the backend included), signs it with the hotkey, sends it,
    checks the receipt the backend signs, and saves forecast, sealed bytes and
    receipt under ~/.numinous/receipts/<hotkey>/<round>/<question>.json.
    One forecast per question: the first one sent counts.

    \b
    Exit codes: 0 sent, 3 no round open, 4 forecast not valid (nothing sent),
    8 already sent for this question, 9 deadline passed, 10 not a question of
    the open round, 11 the receipt does not verify.

    \b
    Inline, no file:
      numi submit --question 1d3ffd76-b934-4741-931c-582e53a47751 <<'EOF'
      {"version": 2, "statements": [{"head": "Apple", "relation": "INVESTS_IN",
       "tail": "TSMC", "probability": 0.15}]}
      EOF
    """
    forecast = read_forecast(forecast_file)
    try:
        submission = client.submit(str(question_id), forecast)
    except ForecastNotValid as invalid:
        reject_forecast(invalid.errors)

    receipt_file = save_receipt(submission, forecast)
    emit(
        {
            "sent": True,
            "round_id": submission.round_id,
            "receipt": submission.receipt.model_dump(mode="json"),
            "saved_to": str(receipt_file),
        }
    )
