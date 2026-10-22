import click

from neurons.miner.client import MinerClient
from neurons.miner.commands.common import emit, with_client


@click.command(name="entities")
@click.option(
    "--name",
    help="Keep entities whose name or one of whose aliases contains this text (at least 2 "
    "characters, case does not matter).",
)
@click.option(
    "--type",
    "entity_types",
    multiple=True,
    help="Keep only entities of this type, e.g. company, product. Repeat for several.",
)
@click.option(
    "--limit", type=click.IntRange(min=1), help="Entities per page (backend default 500)."
)
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True)
@with_client
def entities_command(
    client: MinerClient,
    name: str | None,
    entity_types: tuple[str, ...],
    limit: int | None,
    offset: int,
) -> None:
    """Print one page of the graph's entities matching a name, as JSON.

    \b
    {"entities": [...], "next_offset": n or null}. Each entity: id, canonical
    name, entity_type and aliases. Use the id with numi graph --entity. When
    next_offset is not null, ask again with --offset it.

    \b
    Examples:
      numi entities --name nvidia
      numi entities --name "hon hai" --type company
    """
    emit(client.get_entities(name, list(entity_types), limit, offset))
