import base64
import time
import typing
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from neurons.miner.scripts.link_desearch import link_desearch_impl
from neurons.miner.scripts.link_service import get_all_linkable_services, link_api_key_impl
from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.track_utils import prompt_track_selection
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection
from neurons.validator.models.track import TrackEnum

console = Console()


@click.group()
def services():
    """Manage linked third-party services

    \b
    Available Commands:
      numi services sources           # List all available public data sources
      numi services list              # List your linked services
      numi services link              # Link a service (interactive)
      numi services link <name>       # Link a specific service directly
      numi services unlink <name>     # Unlink a service

    \b
    Examples:
      numi services sources
      numi services list
      numi services link
      numi services link desearch
      numi services link chutes
      numi services link chutes -t SIGNAL
      numi services unlink chutes
    """
    pass


@services.command()
@click.option("--wallet", "-w", type=str, help="Wallet name")
@click.option("--hotkey", "-k", type=str, help="Hotkey name")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment",
)
@click.option(
    "--wallet-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Custom wallet directory path",
)
def list(
    wallet: Optional[str] = None,
    hotkey: Optional[str] = None,
    env: Optional[str] = None,
    wallet_path: Optional[Path] = None,
) -> None:
    """List all linked services for your miner"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🔗 Linked Services[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    if not env:
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    console.print(f"[dim]Network:[/dim] [yellow]{env.upper()}[/yellow]")
    console.print()

    if not wallet or not hotkey:
        wallet, hotkey = prompt_wallet_selection(wallet_path)

    console.print()
    with console.status(f"[cyan]Loading wallet {wallet}/{hotkey}...[/cyan]"):
        keypair = load_keypair(wallet, hotkey, wallet_path)

    if not keypair:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to load wallet:[/red] {wallet}/{hotkey}",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(f"[green]✓[/green] Loaded wallet: [yellow]{keypair.ss58_address}[/yellow]")

    console.print()
    with console.status("[cyan]Fetching linked services...[/cyan]"):
        services_list = _fetch_linked_services(env, keypair)

    if not services_list:
        console.print()
        console.print(
            Panel.fit(
                "[yellow]No services linked yet[/yellow]\n\n"
                "[dim]Link a service with:[/dim]\n"
                "[cyan]numi services link[/cyan]",
                border_style="yellow",
            )
        )
        console.print()
        return

    console.print()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Service", style="green")
    table.add_column("Track", style="magenta")
    table.add_column("Auth Type", style="cyan")
    table.add_column("Updated", style="dim")

    for service in services_list:
        table.add_row(
            service["service_name"],
            service.get("track", TrackEnum.MAIN.value),
            service["auth_type"],
            service["updated_at"][:19],
        )

    console.print(table)
    console.print()


@services.command(name="sources")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment",
)
def sources(env: Optional[str] = None) -> None:
    """List all available public data sources

    \b
    Examples:
      numi services sources
      numi services sources -e prod
    """
    if not env:
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    console.print()
    with console.status("[cyan]Fetching public data sources...[/cyan]"):
        all_sources = _fetch_public_data_sources(env)

    if not all_sources:
        console.print(
            Panel.fit(
                "[yellow]No public data sources available[/yellow]\n\n"
                "[dim]Sources are added over time. Check back later.[/dim]",
                border_style="yellow",
            )
        )
        console.print()
        return

    free_sources = [s for s in all_sources if not s.get("requires_auth")]
    auth_sources = [s for s in all_sources if s.get("requires_auth")]

    console.print()
    if free_sources:
        console.print("[bold cyan]Free Sources[/bold cyan] [dim](no API key needed)[/dim]")
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Name", style="green")
        table.add_column("Domain", style="dim")
        table.add_column("Category", style="magenta")
        for source in free_sources:
            table.add_row(source["name"], source["domain"], source["category"])
        console.print(table)
        console.print()

    if auth_sources:
        console.print(
            "[bold cyan]Auth-Required Sources[/bold cyan] [dim](link via: numi services link)[/dim]"
        )
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Name", style="green")
        table.add_column("Domain", style="dim")
        table.add_column("Category", style="magenta")
        for source in auth_sources:
            table.add_row(source["name"], source["domain"], source["category"])
        console.print(table)
        console.print()


def _fetch_public_data_sources(env: str) -> Optional[typing.List[dict]]:
    api_url = ENV_URLS[env]
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(f"{api_url}/api/v3/miner/public-data/sources")
        if response.status_code == 200:
            return response.json().get("sources", [])
        console.print(
            f"[red]✗ HTTP {response.status_code}:[/red] {api_url}/api/v3/miner/public-data/sources"
        )
        return None
    except Exception as exc:
        console.print(f"[red]✗ Failed to fetch sources:[/red] {exc}")
        return None


@services.command()
@click.argument("service_name", required=False)
@click.option("--wallet", "-w", type=str, help="Wallet name")
@click.option("--hotkey", "-k", type=str, help="Hotkey name")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment",
)
@click.option(
    "--wallet-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Custom wallet directory path",
)
@click.option(
    "--track",
    "-t",
    type=str,
    help="Track to link credentials for (e.g. MAIN, SIGNAL). Default: MAIN.",
)
def link(
    service_name: Optional[str] = None,
    wallet: Optional[str] = None,
    hotkey: Optional[str] = None,
    env: Optional[str] = None,
    wallet_path: Optional[Path] = None,
    track: Optional[str] = None,
) -> None:
    """Link a third-party service to your miner

    \b
    Examples:
      numi services link                      # Interactive mode
      numi services link desearch             # Link Desearch directly
      numi services link chutes               # Link Chutes directly
      numi services link chutes -t SIGNAL     # Link for SIGNAL track
    """
    if not env:
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    all_services = get_all_linkable_services(env)
    service_names_by_cli_name = {"desearch": None}
    for service in all_services:
        cli_name = service.name.replace("_", "-")
        service_names_by_cli_name[cli_name] = service

    if not service_name:
        cli_names = [*service_names_by_cli_name.keys()]
        console.print()
        for index, name in enumerate(cli_names, 1):
            service_config = service_names_by_cli_name[name]
            display = service_config.display_name if service_config else "Desearch"
            console.print(f"  [cyan]{index:>2}.[/cyan] {display} [dim]({name})[/dim]")
        console.print()

        selection = Prompt.ask(
            "[bold cyan]Select service number or name[/bold cyan]",
            default="1",
        )

        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(cli_names):
                service_name = cli_names[idx]
            else:
                console.print(f"[red]✗ Invalid selection:[/red] {selection}")
                raise click.Abort()
        else:
            service_name = selection.lower().strip()
        console.print()

    if not track:
        track = prompt_track_selection(show_allowlist=False, show_credential_fallback=True)
    else:
        track = track.upper()

    if service_name == "desearch":
        link_desearch_impl(wallet, hotkey, env, wallet_path, track)
    elif service_name in service_names_by_cli_name:
        service_config = service_names_by_cli_name[service_name]
        link_api_key_impl(service_config, wallet, hotkey, env, wallet_path, track)
    else:
        console.print(f"[red]✗ Unknown service:[/red] {service_name}")
        raise click.Abort()


@services.command()
@click.argument("service_name")
@click.option("--wallet", "-w", type=str, help="Wallet name")
@click.option("--hotkey", "-k", type=str, help="Hotkey name")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment",
)
@click.option(
    "--wallet-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Custom wallet directory path",
)
@click.option(
    "--track",
    "-t",
    type=str,
    help="Track to unlink credentials for (e.g. MAIN, SIGNAL). Default: MAIN.",
)
def unlink(
    service_name: str,
    wallet: Optional[str] = None,
    hotkey: Optional[str] = None,
    env: Optional[str] = None,
    wallet_path: Optional[Path] = None,
    track: Optional[str] = None,
) -> None:
    """Unlink a service from your miner

    \b
    Examples:
      numi services unlink desearch
      numi services unlink chutes -t SIGNAL
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Unlink Service: {service_name}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    if not env:
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    console.print(f"[dim]Network:[/dim] [yellow]{env.upper()}[/yellow]")
    console.print()

    if not wallet or not hotkey:
        wallet, hotkey = prompt_wallet_selection(wallet_path)

    console.print()
    with console.status(f"[cyan]Loading wallet {wallet}/{hotkey}...[/cyan]"):
        keypair = load_keypair(wallet, hotkey, wallet_path)

    if not keypair:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to load wallet:[/red] {wallet}/{hotkey}",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(f"[green]✓[/green] Loaded wallet: [yellow]{keypair.ss58_address}[/yellow]")

    if not track:
        track = prompt_track_selection(show_allowlist=False)
    else:
        track = track.upper()

    console.print()
    with console.status(f"[cyan]Unlinking {service_name} (track: {track})...[/cyan]"):
        success = _unlink_service(env, keypair, service_name, track)

    if not success:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to unlink {service_name}[/red]\n\n"
                "[yellow]Service may not be linked or network error occurred[/yellow]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✓ Successfully unlinked {service_name}[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _fetch_linked_services(env: str, keypair) -> Optional[typing.List[dict]]:
    api_url = ENV_URLS[env]
    timestamp = int(time.time())
    payload = f"{keypair.ss58_address}:{timestamp}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{api_url}/api/v3/miner/services",
                headers={
                    "Authorization": f"Bearer {signature_base64}",
                    "Miner-Public-Key": public_key_hex,
                    "Miner": keypair.ss58_address,
                    "X-Payload": payload,
                },
            )

        if response.status_code == 200:
            result = response.json()
            return result.get("credentials", [])
        return None
    except Exception:
        return None


def _unlink_service(env: str, keypair, service_name: str, track: str) -> bool:
    api_url = ENV_URLS[env]
    timestamp = int(time.time())
    payload = f"{keypair.ss58_address}:{timestamp}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(
                f"{api_url}/api/v3/miner/services/{service_name}",
                params={"track": track},
                headers={
                    "Authorization": f"Bearer {signature_base64}",
                    "Miner-Public-Key": public_key_hex,
                    "Miner": keypair.ss58_address,
                    "X-Payload": payload,
                },
            )
        return response.status_code == 204
    except Exception:
        return False


if __name__ == "__main__":
    services()
