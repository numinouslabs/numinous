import click

from neurons.miner.client import MinerClient
from neurons.miner.commands.common import emit, with_client


@click.command(name="round")
@with_client
def round_command(client: MinerClient) -> None:
    """Print the round accepting forecasts now, as JSON.

    \b
    Fields: id, deadline_at (last moment to submit), horizon_end_at (last day a
    statement can come true), reveal_round (the timelock round every forecast of
    this round is sealed to) and questions (id, entity, optional counterparty and
    known relationship). Exits 3 when no round is accepting forecasts.

    \b
    Example:
      numi round --env test --wallet my_wallet --hotkey my_hotkey
    """
    emit(client.get_round().model_dump(mode="json"))
