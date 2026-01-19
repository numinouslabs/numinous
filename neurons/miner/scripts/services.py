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

from neurons.miner.scripts.link_chutes import link_chutes_impl
from neurons.miner.scripts.link_desearch import link_desearch_impl
from neurons.miner.scripts.link_openai import link_openai_impl
from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection

console = Console()


@click.group()
def services():
    """Manage linked third-party services

    \b
    Available Commands:
      numi services list           # List all linked services
      numi services link           # Link a service (interactive)
      numi services link desearch  # Link Desearch directly
      numi services link chutes    # Link Chutes directly
      numi services link openai    # Link OpenAI directly
      numi services unlink <name>  # Unlink a service

    \b
    Examples:
      numi services list
      numi services link
      numi services link desearch
      numi services link chutes
      numi services link openai
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
    table.add_column("Auth Type", style="cyan")
    table.add_column("Updated", style="dim")

    for service in services_list:
        table.add_row(
            service["service_name"],
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
def link(
    service_name: Optional[str] = None,
    wallet: Optional[str] = None,
    hotkey: Optional[str] = None,
    env: Optional[str] = None,
    wallet_path: Optional[Path] = None,
) -> None:
    """Link a third-party service to your miner

    \b
    Available Services:
      - desearch: Link Desearch API account
      - chutes: Link Chutes API key
      - openai: Link OpenAI API key

    \b
    Examples:
      numi services link              # Interactive mode
      numi services link desearch     # Link Desearch directly
      numi services link chutes       # Link Chutes directly
      numi services link openai       # Link OpenAI directly
    """
    if not service_name:
        console.print()
        service_choice = Prompt.ask(
            "[bold cyan]Select service to link[/bold cyan]",
            choices=["desearch", "chutes", "openai"],
            default="desearch",
        )
        service_name = service_choice.lower()
        console.print()

    if service_name == "desearch":
        link_desearch_impl(wallet, hotkey, env, wallet_path)
    elif service_name == "chutes":
        link_chutes_impl(wallet, hotkey, env, wallet_path)
    elif service_name == "openai":
        link_openai_impl(wallet, hotkey, env, wallet_path)
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
def unlink(
    service_name: str,
    wallet: Optional[str] = None,
    hotkey: Optional[str] = None,
    env: Optional[str] = None,
    wallet_path: Optional[Path] = None,
) -> None:
    """Unlink a service from your miner

    \b
    Examples:
      numi services unlink desearch
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

    console.print()
    with console.status(f"[cyan]Unlinking {service_name}...[/cyan]"):
        success = _unlink_service(env, keypair, service_name)

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


def _unlink_service(env: str, keypair, service_name: str) -> bool:
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
