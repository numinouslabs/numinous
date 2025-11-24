"""Numinous CLI - Main entry point"""

import click

from neurons.miner.scripts.fetch_agent_logs import fetch_logs
from neurons.miner.scripts.gateway import gateway
from neurons.miner.scripts.inspect_agent import inspect_agent
from neurons.miner.scripts.list_agents import list_agents
from neurons.miner.scripts.test_agent import test
from neurons.miner.scripts.upload_agent import upload


@click.group()
@click.version_option(version="2.0.0", prog_name="numi")
def numi():
    """✨ Numinous - Forecasting Subnet CLI

    A Bittensor subnet for decentralized forecasting of future events.

    \b
    Available Commands:
      numi gateway        # Manage your local miner gateway
      numi test-agent     # Test your agent locally
      numi upload-agent   # Submit your agent's code
      numi list-agents    # List your uploaded agents
      numi inspect-agent  # View/download any activated agent code
      numi fetch-logs     # Fetch agent run logs

    \b
    For detailed help on any command, run:
      numi <command> --help
    """
    pass


# Register commands
numi.add_command(gateway)
numi.add_command(test, name="test-agent")
numi.add_command(upload, name="upload-agent")
numi.add_command(list_agents, name="list-agents")
numi.add_command(inspect_agent, name="inspect-agent")
numi.add_command(fetch_logs, name="fetch-logs")
numi.add_command(list_agents, name="list-agents")


if __name__ == "__main__":
    numi()
