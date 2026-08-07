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

from neurons.miner.scripts.link_service import LINKABLE_SERVICES, link_api_key_impl
from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.track_utils import prompt_credential_track
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection
from neurons.validator.models.track import TrackEnum

console = Console()


@click.group()
def services():
    """Manage linked third-party services

    \b
    Available Commands:
      numi services list              # List your linked services
      numi services link              # Link a service (interactive)
      numi services link <name>       # Link a specific service directly
      numi services unlink <name>     # Unlink a service

    \b
    Examples:
      numi services list
      numi services link
      numi services link openai
      numi services link openrouter -t SIGNAL
      numi services unlink openai
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
    help="Track to link credentials for. Default: SIGNAL, the only live track.",
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
      numi services link                       # Interactive mode
      numi services link openai                # Link OpenAI directly
      numi services link openrouter -t SIGNAL  # Link for SIGNAL track
    """
    if not env:
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    service_names_by_cli_name = {
        service.name.replace("_", "-"): service for service in LINKABLE_SERVICES
    }

    if not service_name:
        cli_names = [*service_names_by_cli_name.keys()]
        console.print()
        for index, name in enumerate(cli_names, 1):
            service_config = service_names_by_cli_name[name]
            console.print(
                f"  [cyan]{index:>2}.[/cyan] {service_config.display_name} [dim]({name})[/dim]"
            )
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
        track = prompt_credential_track(show_fallback_note=True)
    else:
        track = track.upper()

    if service_name not in service_names_by_cli_name:
        console.print(f"[red]✗ Unknown service:[/red] {service_name}")
        raise click.Abort()

    link_api_key_impl(
        service_names_by_cli_name[service_name], wallet, hotkey, env, wallet_path, track
    )


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
    help="Track to unlink credentials for. Default: SIGNAL, the only live track.",
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
      numi services unlink openai
      numi services unlink openrouter -t SIGNAL
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
        track = prompt_credential_track(show_fallback_note=False)
    else:
        track = track.upper()

    console.print()
    with console.status(f"[cyan]Unlinking {service_name} (track: {track})...[/cyan]"):
        success = _unlink_service(env, keypair, service_name, track)

    if not success:
        _report_unlink_failure(env, keypair, service_name, track)

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✓ Successfully unlinked {service_name}[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _report_unlink_failure(env: str, keypair, service_name: str, track: str) -> typing.NoReturn:
    linked_services = _fetch_linked_services(env, keypair) or []
    normalized_name = service_name.replace("-", "_")
    other_tracks = [
        service["track"]
        for service in linked_services
        if service["service_name"] == normalized_name and service.get("track") != track
    ]

    if other_tracks:
        detail = (
            f"[yellow]{service_name} is linked on:[/yellow] {', '.join(other_tracks)}\n\n"
            f"[dim]Retry with:[/dim] [cyan]numi services unlink {service_name} "
            f"-t {other_tracks[0]}[/cyan]"
        )
    else:
        detail = "[yellow]Service may not be linked or network error occurred[/yellow]"

    console.print()
    console.print(
        Panel.fit(
            f"[red]✗ Failed to unlink {service_name} on {track}[/red]\n\n{detail}",
            border_style="red",
        )
    )
    console.print()
    raise click.Abort()


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
    service_name = service_name.replace("-", "_")
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
