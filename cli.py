"""Numinous CLI - Main entry point"""

import click

from neurons.miner.commands.entities import entities_command
from neurons.miner.commands.forecasts import forecasts_command
from neurons.miner.commands.graph import graph_command
from neurons.miner.commands.round import round_command
from neurons.miner.commands.schema import schema_command
from neurons.miner.commands.submit import submit_command
from neurons.miner.commands.validate import validate_command
from neurons.validator.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="numi")
def numi():
    """✨ Numinous - Forecasting Subnet CLI

    A Bittensor subnet for decentralized forecasting of future events.

    \b
    Graph forecasting (JSON output, no prompts):
      numi round           # The round accepting forecasts and its questions
      numi schema          # The JSON Schema a forecast must follow
      numi validate        # Check a forecast file against that schema
      numi submit          # Seal, sign and send a forecast for one question
      numi forecasts       # Your own forecasts and how each statement was judged
      numi graph           # What the graph already knows about a company
      numi entities        # Find an entity's id by name or alias

    \b
    For detailed help on any command, run:
      numi <command> --help
    """
    pass


# Register commands
numi.add_command(round_command)
numi.add_command(schema_command)
numi.add_command(validate_command)
numi.add_command(submit_command)
numi.add_command(forecasts_command)
numi.add_command(graph_command)
numi.add_command(entities_command)


if __name__ == "__main__":
    numi()
