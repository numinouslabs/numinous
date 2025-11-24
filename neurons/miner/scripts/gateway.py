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

    if manager.check_gateway_health():
        pid = manager.get_gateway_pid()
        console.print("[yellow]⚠ Gateway is already running[/yellow]")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print()
        console.print("  [yellow]📋 View logs:[/yellow] [cyan]gateway logs[/cyan]")
        console.print("  [yellow]🛑 Stop:[/yellow] [cyan]gateway stop[/cyan]")
        console.print()
        return

    env_vars = config.check_env_vars()
    all_env_ok = all(env_vars.values())

    if not all_env_ok:
        missing_keys = [key for key, ok in env_vars.items() if not ok]
        console.print(f"[yellow]⚠ Missing API keys: {', '.join(missing_keys)}[/yellow]")
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

    env_vars = config.check_env_vars()

    console.print("[dim]Current status:[/dim]")
    for key, is_set in env_vars.items():
        status = "[green]✓ Set[/green]" if is_set else "[red]✗ Not set[/red]"
        console.print(f"  {key}: {status}")
    console.print()

    all_set = all(env_vars.values())

    if all_set:
        if not Confirm.ask(
            "[bold cyan]All keys are configured. Do you wish to update any of them?[/bold cyan]",
            default=False,
        ):
            console.print()
            console.print("[dim]No changes made[/dim]")
            console.print()
            return
        force_all = True
    else:
        force_all = False
        if any(env_vars.values()):
            if Confirm.ask(
                "[bold cyan]Update all keys (including existing ones)?[/bold cyan]", default=False
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
