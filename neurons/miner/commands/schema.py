import click

from neurons.miner.client import MinerClient
from neurons.miner.commands.common import emit, with_client


@click.command(name="schema")
@with_client
def schema_command(client: MinerClient) -> None:
    """Print the JSON Schema a forecast must follow.

    \b
    It is the exact shape the backend checks when it opens a sealed forecast:
    anything that does not match is stored as invalid.

    \b
    Example:
      numi schema --env test --wallet my_wallet --hotkey my_hotkey
    """
    emit(client.get_schema())
