import uuid
from datetime import datetime

import click

from neurons.miner.client import MinerClient
from neurons.miner.commands.common import Moment, emit, with_client


@click.command(name="forecasts")
@click.option("--round", "round_id", type=click.UUID, help="Only the forecasts sent to this round.")
@click.option(
    "--entity",
    "entity_id",
    type=click.UUID,
    help="Only the forecasts on questions about this company (entity_id from numi round).",
)
@click.option(
    "--changed-since",
    type=Moment(),
    help="Only the forecasts sent, opened, found invalid or judged after this moment (UTC when no timezone).",
)
@with_client
def forecasts_command(
    client: MinerClient,
    round_id: uuid.UUID | None,
    entity_id: uuid.UUID | None,
    changed_since: datetime | None,
) -> None:
    """Print your own forecasts and how each statement was judged, as JSON.

    \b
    Every forecast this hotkey sent, oldest first, with its round, deadline,
    horizon end, question, sealed_sha256 and state:
      sealed   received, the seal has not been opened yet (before the deadline)
      opened   opened and valid, see "opened" for the content
      invalid  did not open or did not follow the schema, see "invalid_reason"
    and per statement a status: right (with the event and its news date),
    wrong (with the judge's reason) or waiting (not judged for good yet).

    \b
    Examples:
      numi forecasts --env test --wallet my_wallet --hotkey my_hotkey
      numi forecasts --round 31554aee-295f-4524-a1fa-86de56327742
      numi forecasts --entity c4399b0f-ab16-5a64-aab3-89f18ce8e2d3 --changed-since 2026-10-13T00:00:00Z
    """
    forecasts = client.get_my_forecasts(round_id, entity_id, changed_since)
    emit({"forecasts": [forecast.model_dump(mode="json") for forecast in forecasts]})
