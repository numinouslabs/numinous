import uuid
from datetime import datetime

import click

from neurons.miner.client import MinerClient
from neurons.miner.commands.common import Moment, emit, with_client


@click.command(name="graph")
@click.option(
    "--entity",
    "entity_id",
    type=click.UUID,
    help="This entity, its direct neighbours and the edges between them (entity_id from numi round "
    "or numi entities). Without it, the whole graph.",
)
@click.option(
    "--relation",
    "relations",
    multiple=True,
    help="Keep only edges with this relation, e.g. SUPPLIES, PARTNERS_WITH. Repeat for several.",
)
@click.option(
    "--published-by",
    type=Moment(),
    help="Only what had been published by this moment (UTC when no timezone).",
)
@click.option(
    "--with-observations",
    is_flag=True,
    help="Include the articles and quotes behind each edge. Much larger output.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1, max=2000),
    default=100,
    show_default=True,
    help="Edges per page, newest first.",
)
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True)
@with_client
def graph_command(
    client: MinerClient,
    entity_id: uuid.UUID | None,
    relations: tuple[str, ...],
    published_by: datetime | None,
    with_observations: bool,
    limit: int,
    offset: int,
) -> None:
    """Print one page of what the graph already knows, as JSON.

    \b
    {"edges": [...], "events": [...], "next_offset": n or null}. Each edge is a
    relationship: head, relation, tail (names and ids), standing, support,
    first_seen and last_seen, newest first. A statement that only repeats what
    the graph already held at the deadline cannot count, so read this before
    forecasting. When next_offset is not null, ask again with --offset it.

    \b
    standing: does it hold?  stated (not denied, or confirmed after a denial),
              disputed (denied and not firmly confirmed since), denied
    support:  what backs it?  news (reported as a new development, rumours
              included), ongoing (only reported as continuing), recalled
              (only background mentions), denied (only denials)
    An edge is already held, so forecasting it again is judged not correct,
    only when at least one of its reports is announced, completed or ongoing.
    An edge backed only by rumoured reports is not held yet. Each report's
    status comes with --with-observations.

    \b
    With --entity the page holds the entity, its direct neighbours and every
    edge among them: its own edges are those whose head_id or tail_id is the
    entity id. events (the news developments behind the articles) come only
    with --with-observations.

    \b
    Examples:
      numi graph --entity c4399b0f-ab16-5a64-aab3-89f18ce8e2d3
      numi graph --entity c4399b0f-ab16-5a64-aab3-89f18ce8e2d3 --relation SUPPLIES --relation INVESTS_IN
      numi graph --entity c4399b0f-ab16-5a64-aab3-89f18ce8e2d3 --with-observations --limit 50
    """
    emit(
        client.get_graph(entity_id, list(relations), published_by, with_observations, limit, offset)
    )
