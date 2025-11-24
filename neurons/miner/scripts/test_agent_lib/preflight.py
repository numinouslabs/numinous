import sys
from pathlib import Path

import docker
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from neurons.miner.scripts.gateway_lib import config as gateway_config  # noqa: E402
from neurons.miner.scripts.gateway_lib import manager as gateway_manager  # noqa: E402

console = Console()

GATEWAY_URL = gateway_manager.GATEWAY_URL
GATEWAY_ENV_PATH = gateway_config.GATEWAY_ENV_PATH


def check_docker() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


check_gateway_health = gateway_manager.check_gateway_health
get_gateway_pid = gateway_manager.get_gateway_pid
stop_gateway = gateway_manager.stop_gateway
check_env_vars = gateway_config.check_env_vars
setup_api_keys = gateway_config.setup_api_keys


def check_sandbox_image() -> bool:
    try:
        client = docker.from_env()
        client.images.get("ig-validator-sandbox-image")
        return True
    except Exception:
        return False


start_gateway = gateway_manager.start_gateway


def run_preflight_checks() -> bool:
    console.print()
    console.print("[cyan]Checking requirements...[/cyan]")
    console.print()

    all_passed = True

    docker_ok = check_docker()
    if docker_ok:
        console.print("  [green]✓[/green] Docker daemon running")
    else:
        console.print("  [red]✗[/red] Docker daemon not running")
        all_passed = False

    if docker_ok:
        image_ok = check_sandbox_image()
        if image_ok:
            console.print("  [green]✓[/green] Sandbox image built")
        else:
            console.print("  [yellow]⚠[/yellow] Sandbox image not built (will build on first run)")

    gateway_ok = check_gateway_health()
    if gateway_ok:
        console.print(f"  [green]✓[/green] Gateway health check ({GATEWAY_URL})")
    else:
        console.print(f"  [red]✗[/red] Gateway not running at {GATEWAY_URL}")
        all_passed = False

    if gateway_ok:
        env_vars = check_env_vars()
        all_env_ok = all(env_vars.values())

        if all_env_ok:
            console.print("  [green]✓[/green] Environment variables")
            for key in env_vars.keys():
                console.print(f"    • {key}: [green]✓[/green]")
        else:
            console.print("  [yellow]⚠[/yellow] Environment variables")
            for key, ok in env_vars.items():
                status = "[green]✓[/green]" if ok else "[red]✗[/red]"
                console.print(f"    • {key}: {status}")

            if not all_env_ok:
                all_passed = False

    console.print()

    if not docker_ok:
        console.print(
            Panel.fit(
                "[red]✗ Docker is not running[/red]\n\n"
                "[yellow]💡 Tip:[/yellow] Start Docker Desktop or Docker daemon\n"
                "   Then try again.",
                border_style="red",
                title="❌ Docker Required",
            )
        )
        console.print()

    if docker_ok and not gateway_ok:
        console.print(
            Panel.fit(
                "[red]✗ Gateway not running at localhost:8000[/red]\n\n"
                "[yellow]The gateway is required to run agent tests.[/yellow]",
                border_style="red",
                title="❌ Gateway Required",
            )
        )
        console.print()

        if Confirm.ask(
            "[bold cyan]Would you like to start the gateway now?[/bold cyan]", default=True
        ):
            console.print()
            success, pid, log_file = start_gateway()
            if success:
                console.print()
                console.print(
                    Panel.fit(
                        f"[green]✓ Gateway started successfully![/green]\n\n"
                        f"[dim]Process ID:[/dim] {pid}\n"
                        f"[dim]URL:[/dim] {GATEWAY_URL}\n"
                        f"[dim]Logs:[/dim] {log_file.absolute()}\n\n"
                        f"[yellow]📋 View logs:[/yellow] [cyan]numi gateway logs[/cyan]\n"
                        f"[yellow]🛑 Stop gateway:[/yellow] [cyan]numi gateway stop[/cyan]",
                        border_style="green",
                        title="✓ Gateway Running",
                    )
                )
                console.print()
                gateway_ok = True
                env_vars_ok = all(check_env_vars().values())
                all_passed = env_vars_ok
            else:
                console.print()
                console.print(
                    Panel.fit(
                        "[red]✗ Failed to start gateway[/red]\n\n"
                        "[yellow]💡 Try starting it manually:[/yellow]\n"
                        "   [cyan]numi gateway start[/cyan]",
                        border_style="red",
                    )
                )
                console.print()
        else:
            console.print()
            console.print(
                Panel.fit(
                    "[yellow]💡 To start the gateway manually:[/yellow]\n"
                    "   [cyan]numi gateway start[/cyan]",
                    border_style="yellow",
                )
            )
            console.print()

    if gateway_ok and not all(check_env_vars().values()):
        env_vars = check_env_vars()
        missing_keys = [key for key, ok in env_vars.items() if not ok]

        console.print(
            Panel.fit(
                f"[yellow]⚠ Missing API keys: {', '.join(missing_keys)}[/yellow]\n\n"
                "[dim]These keys are required to run agents that use LLM APIs.[/dim]",
                border_style="yellow",
                title="⚠️ API Keys Required",
            )
        )
        console.print()

        if Confirm.ask(
            "[bold cyan]Would you like to set up your API keys now?[/bold cyan]", default=True
        ):
            if setup_api_keys():
                env_vars_ok = all(check_env_vars().values())
                if env_vars_ok:
                    console.print("[green]✓[/green] All API keys configured!")
                    console.print()

                    console.print("[cyan]Restarting gateway to load new API keys...[/cyan]")

                    if stop_gateway():
                        console.print("  [green]✓[/green] Stopped existing gateway")

                    success, pid, log_file = start_gateway()

                    if success:
                        console.print()
                        console.print(
                            Panel.fit(
                                f"[green]✓ Gateway restarted successfully![/green]\n\n"
                                f"[dim]Process ID:[/dim] {pid}\n"
                                f"[dim]URL:[/dim] {GATEWAY_URL}\n"
                                f"[dim]Logs:[/dim] {log_file.absolute()}\n\n"
                                f"[yellow]📋 View logs:[/yellow] [cyan]numi gateway logs[/cyan]\n"
                                f"[yellow]🛑 Stop gateway:[/yellow] [cyan]numi gateway stop[/cyan]",
                                border_style="green",
                                title="✓ Gateway Running with New Keys",
                            )
                        )
                        console.print()
                        all_passed = True
                    else:
                        console.print("[red]✗[/red] Failed to restart gateway")
                        console.print()
                        all_passed = False
                else:
                    console.print("[red]✗[/red] Some API keys are still missing")
                    console.print()
                    all_passed = False
            else:
                all_passed = False
        else:
            console.print()
            console.print(
                Panel.fit(
                    f"[yellow]💡 To set up API keys manually, edit:[/yellow]\n"
                    f"   [cyan]{GATEWAY_ENV_PATH}[/cyan]\n\n"
                    "[yellow]Required keys:[/yellow]\n"
                    "   CHUTES_API_KEY=your_key_here\n"
                    "   DESEARCH_API_KEY=your_key_here\n\n"
                    "[yellow]Get your keys from:[/yellow]\n"
                    "   • Chutes: [link=https://chutes.ai]https://chutes.ai[/link]\n"
                    "   • Desearch: [link=https://desearch.ai]https://desearch.ai[/link]",
                    border_style="yellow",
                )
            )
            console.print()
            all_passed = False

    return all_passed
