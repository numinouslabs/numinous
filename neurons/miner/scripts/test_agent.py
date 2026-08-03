import sys
import typing
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from neurons.miner.scripts.test_agent_lib import (  # noqa: E402
    display,
    execution,
    preflight,
    selection,
)
from neurons.miner.scripts.track_utils import resolve_agent_track  # noqa: E402
from neurons.validator.models.track import TrackEnum  # noqa: E402

console = Console()


@click.command()
@click.option(
    "--agent-file",
    "-f",
    type=str,
    help="Agent file path (interactive prompt if not provided)",
)
@click.option(
    "--track",
    "-t",
    type=str,
    help="Track to test with (controls endpoint filtering). Defaults to SIGNAL.",
)
def test(agent_file: typing.Optional[str] = None, track: typing.Optional[str] = None) -> None:
    """Test your forecasting agent locally with real events

    \b
    Examples:
      # Interactive mode
      numi test-agent

      # Test specific agent
      numi test-agent --agent-file my_agent.py
      numi test-agent -f baseline_agent.py

      # Test with SIGNAL track (restricted endpoints)
      numi test-agent -f my_signal_agent.py -t SIGNAL

    \b
    💡 Tip: Use 'numi gateway' for gateway management
      numi gateway start      # Start the gateway
      numi gateway stop       # Stop the gateway
      numi gateway status     # Check gateway status
      numi gateway logs       # View gateway logs
    """

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]✨ Numinous - Agent Testing Tool[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    if not preflight.run_preflight_checks():
        console.print(
            "[red]Pre-flight checks failed. Please fix the issues above and try again.[/red]"
        )
        console.print()
        raise click.Abort()

    console.print("[green]✓ All checks passed![/green]")
    console.print()

    agent_path = selection.select_agent(agent_file)
    events = selection.select_events()
    if not events:
        console.print("[yellow]⚠ No events selected. Exiting.[/yellow]")
        console.print()
        return

    console.print(f"[green]✓[/green] Selected [cyan]{len(events)}[/cyan] event(s)")
    console.print()

    track_enum = TrackEnum(resolve_agent_track(track, allow_dead_track=True))
    results = execution.run_tests(agent_path, events, track=track_enum)
    display.show_results(results)


if __name__ == "__main__":
    test(standalone_mode=True)
