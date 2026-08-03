import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from neurons.miner.scripts.gateway_lib import config, manager  # noqa: E402

console = Console()

_STATUS_DISPLAY = {
    config.ApiKeyStatus.SET: "[green]✓ Set[/green]",
    config.ApiKeyStatus.SKIPPED: "[dim]— skipped[/dim]",
    config.ApiKeyStatus.UNDECIDED: "[red]✗ Not set[/red]",
}


@click.group()
def gateway():
    """Manage the local miner gateway

    \b
    Examples:
      numi gateway start      # Start the gateway
      numi gateway stop       # Stop the gateway
      numi gateway status     # Check gateway status
      numi gateway logs       # View gateway logs
      numi gateway configure  # Set up API keys
    """
    pass


@gateway.command()
def start():
    """Start the gateway (with API key setup if needed)"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🚀 Starting Miner Gateway[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    pid = manager.get_gateway_pid()
    if manager.check_gateway_health():
        console.print("[yellow]⚠ Gateway is already running[/yellow]")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print()
        console.print("  [yellow]📋 View logs:[/yellow] [cyan]gateway logs[/cyan]")
        console.print("  [yellow]🛑 Stop:[/yellow] [cyan]gateway stop[/cyan]")
        console.print()
        return

    if pid:
        console.print(
            f"[yellow]⚠ Gateway process exists (PID: {pid}) but is not responding.[/yellow]"
        )
        console.print()
        if Confirm.ask("[bold cyan]Kill it and start fresh?[/bold cyan]", default=True):
            if manager.stop_gateway():
                console.print(f"  [green]✓[/green] Killed gateway process (PID: {pid})")
                console.print()
            else:
                console.print(
                    f"  [red]✗[/red] Could not kill process {pid}. Try: [cyan]kill {pid}[/cyan]"
                )
                console.print()
                return
        else:
            console.print()
            return

    if manager.is_port_in_use(manager.GATEWAY_PORT):
        port_pid = manager.get_port_pid(manager.GATEWAY_PORT)
        pid_info = f" (PID: {port_pid})" if port_pid else ""
        console.print(
            f"[yellow]⚠ Port {manager.GATEWAY_PORT} is already in use by another process{pid_info}.[/yellow]"
        )
        console.print()
        if Confirm.ask("[bold cyan]Kill it and start fresh?[/bold cyan]", default=True):
            if manager.kill_port(manager.GATEWAY_PORT):
                console.print(f"  [green]✓[/green] Killed process on port {manager.GATEWAY_PORT}")
                console.print()
            else:
                console.print(f"  [red]✗[/red] Could not kill process{pid_info}.")
                if port_pid:
                    console.print(f"  [dim]Try manually: [cyan]kill {port_pid}[/cyan][/dim]")
                console.print()
                return
        else:
            console.print()
            return

    undecided_keys = config.undecided_api_keys()

    if undecided_keys:
        console.print(f"[yellow]⚠ API keys not set up yet: {', '.join(undecided_keys)}[/yellow]")
        console.print()

        if Confirm.ask(
            "[bold cyan]Would you like to set up your API keys now?[/bold cyan]", default=True
        ):
            if not config.setup_api_keys():
                console.print("[red]✗ Failed to set up API keys[/red]")
                console.print()
                return
        else:
            console.print("[yellow]⚠ Gateway will start without API keys configured[/yellow]")
            console.print()

    success, pid, log_file = manager.start_gateway()
    if success:
        console.print()
        console.print(
            Panel.fit(
                f"[green]✓ Gateway started successfully![/green]\n\n"
                f"[dim]Process ID:[/dim] {pid}\n"
                f"[dim]URL:[/dim] {manager.GATEWAY_URL}\n"
                f"[dim]Logs:[/dim] {log_file.absolute()}\n\n"
                f"[yellow]📋 View logs:[/yellow] [cyan]numi gateway logs[/cyan]\n"
                f"[yellow]🛑 Stop gateway:[/yellow] [cyan]numi gateway stop[/cyan]",
                border_style="green",
                title="✓ Gateway Running",
            )
        )
        console.print()
    else:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ Failed to start gateway[/red]\n\n"
                "[yellow]💡 Try checking the logs:[/yellow]\n"
                "   [cyan]numi gateway logs[/cyan]",
                border_style="red",
            )
        )
        console.print()


@gateway.command()
def stop():
    """Stop the running gateway"""
    console.print()

    pid = manager.get_gateway_pid()
    if not pid:
        console.print("[yellow]⚠ Gateway is not running[/yellow]")
        console.print()
        return

    console.print(f"[cyan]Stopping gateway (PID: {pid})...[/cyan]")

    if manager.stop_gateway():
        console.print("[green]✓[/green] Gateway stopped successfully")
    else:
        console.print("[red]✗[/red] Failed to stop gateway")
        console.print(f"[dim]Try manually: kill {pid}[/dim]")

    console.print()


@gateway.command()
def status():
    """Show gateway status"""
    manager.show_gateway_status()


@gateway.command()
@click.option(
    "--no-follow",
    is_flag=True,
    help="Don't follow logs, just show last 50 lines",
)
def logs(no_follow):
    """View gateway logs (follows by default, press Ctrl+C to stop)"""
    manager.tail_logs(follow=not no_follow)


@gateway.command()
def configure():
    """Configure API keys"""
    console.print()
    console.print("[cyan]🔑 API Key Configuration[/cyan]")
    console.print()

    statuses = config.read_api_key_statuses()

    console.print("[dim]Current status:[/dim]")
    for key, status in statuses.items():
        console.print(f"  {key}: {_STATUS_DISPLAY[status]}")
    console.print()

    everything_decided = config.ApiKeyStatus.UNDECIDED not in statuses.values()

    if everything_decided:
        if not Confirm.ask(
            "[bold cyan]Every key has been set or skipped. Review them all?[/bold cyan]",
            default=False,
        ):
            console.print()
            console.print("[dim]No changes made[/dim]")
            console.print()
            return
        force_all = True
    else:
        force_all = False
        if Confirm.ask(
            "[bold cyan]Review all keys, including ones already set or skipped?[/bold cyan]",
            default=False,
        ):
            force_all = True

    if config.setup_api_keys(force_all=force_all):
        console.print("[green]✓ API keys configured![/green]")
        console.print()

        if manager.check_gateway_health():
            if Confirm.ask(
                "[cyan]Gateway is running. Restart to load new keys?[/cyan]", default=True
            ):
                console.print()
                if manager.stop_gateway():
                    console.print("[green]✓[/green] Stopped existing gateway")

                success, pid, log_file = manager.start_gateway()
                if success:
                    console.print()
                    console.print("[green]✓[/green] Gateway restarted with new keys!")
                    console.print()
                else:
                    console.print("[red]✗[/red] Failed to restart gateway")
                    console.print()


if __name__ == "__main__":
    gateway(standalone_mode=True)
