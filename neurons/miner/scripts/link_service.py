import base64
import time
import typing
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection

console = Console()


@dataclass
class ServiceConfig:
    name: str
    display_name: str
    key_url: str


HARDCODED_SERVICES: list[ServiceConfig] = [
    ServiceConfig(name="chutes", display_name="Chutes", key_url="https://chutes.ai/app"),
    ServiceConfig(
        name="openai", display_name="OpenAI", key_url="https://platform.openai.com/api-keys"
    ),
    ServiceConfig(
        name="openrouter", display_name="OpenRouter", key_url="https://openrouter.ai/settings/keys"
    ),
    ServiceConfig(
        name="perplexity",
        display_name="Perplexity",
        key_url="https://www.perplexity.ai/settings/api",
    ),
    ServiceConfig(name="vericore", display_name="Vericore", key_url="https://vericore.ai"),
    ServiceConfig(name="lunar_crush", display_name="LunarCrush", key_url="https://lunarcrush.com"),
    ServiceConfig(
        name="lightning_rod", display_name="Lightning Rod", key_url="https://lightningrod.ai"
    ),
    ServiceConfig(
        name="numinous_signals",
        display_name="Numinous Signals",
        key_url="https://eversight.numinouslabs.io/api-keys",
    ),
    ServiceConfig(
        name="unusual_whales",
        display_name="Unusual Whales",
        key_url="https://unusualwhales.com/pricing?product=api",
    ),
]


def fetch_public_data_sources(env: str) -> list[ServiceConfig]:
    api_url = ENV_URLS[env]

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(f"{api_url}/api/v3/miner/public-data/sources")

        if response.status_code != 200:
            return []

        sources = response.json().get("sources", [])
        return [
            ServiceConfig(
                name=source["name"],
                display_name=source["name"].replace("_", " ").title(),
                key_url=source.get("base_url") or source.get("domain", ""),
            )
            for source in sources
            if source.get("requires_auth")
        ]
    except Exception:
        return []


def get_all_linkable_services(env: str) -> list[ServiceConfig]:
    hardcoded_names = {service.name for service in HARDCODED_SERVICES}
    public_data_services = fetch_public_data_sources(env)
    new_services = [s for s in public_data_services if s.name not in hardcoded_names]
    return HARDCODED_SERVICES + new_services


def link_api_key_impl(
    service: ServiceConfig,
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
    track: str = "MAIN",
) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]🔗 Link {service.display_name} API Key[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    if not wallet or not hotkey:
        wallet, hotkey = prompt_wallet_selection(wallet_path)

    console.print()
    with console.status(f"[cyan]Loading hotkey {wallet}/{hotkey}...[/cyan]"):
        hotkey_keypair = load_keypair(wallet, hotkey, wallet_path)

    if not hotkey_keypair:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to load hotkey:[/red] {wallet}/{hotkey}",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(f"[green]✓[/green] Loaded hotkey: [yellow]{hotkey_keypair.ss58_address}[/yellow]")

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{service.display_name} API Key Setup[/bold cyan]\n\n"
            f"[dim]Get your API key from:[/dim] [cyan]{service.key_url}[/cyan]\n\n"
            "[dim]Your API key will be securely stored in AWS Secrets Manager.[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    api_key = getpass("Enter your API key: ")
    if not api_key or not api_key.strip():
        console.print("[red]✗[/red] API key cannot be empty")
        raise click.Abort()

    console.print()
    console.print("[bold cyan]Ready to link:[/bold cyan]")
    console.print(f"  [dim]Hotkey:[/dim] {hotkey_keypair.ss58_address[:16]}...")
    console.print(f"  [dim]Service:[/dim] {service.display_name}")
    console.print(f"  [dim]Network:[/dim] {env.upper()}")
    console.print(f"  [dim]Track:[/dim] {track}")
    console.print()

    if not Confirm.ask("[yellow]Proceed with linking?[/yellow]", default=True):
        console.print("[dim]Cancelled[/dim]")
        raise click.Abort()

    console.print()
    with console.status("[cyan]Storing credentials in backend...[/cyan]"):
        success, error_msg = _store_api_key_credentials(
            env, hotkey_keypair, api_key, service.name, track
        )

    if not success:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to link {service.display_name} API key[/red]\n\n"
                f"[yellow]Error:[/yellow] {error_msg}\n\n"
                "[dim]Please check:[/dim]\n"
                "  • API key is valid\n"
                "  • Wallet has been registered as miner\n"
                "  • Network connection is stable",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]✓ Successfully linked {service.display_name} API key![/bold green]\n\n"
            f"[dim]Hotkey:[/dim] {hotkey_keypair.ss58_address[:16]}...\n"
            f"[dim]Service:[/dim] {service.display_name}\n"
            f"[dim]Track:[/dim] {track}",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _store_api_key_credentials(
    env: str, keypair, api_key: str, service_name: str, track: str
) -> typing.Tuple[bool, typing.Optional[str]]:
    api_url = ENV_URLS[env]
    timestamp = int(time.time())
    payload = f"{keypair.ss58_address}:{timestamp}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/api/v3/miner/services/link",
                json={
                    "service_name": service_name,
                    "auth_type": "api_key",
                    "track": track,
                    "credential_data": {
                        "api_key": api_key,
                    },
                },
                headers={
                    "Authorization": f"Bearer {signature_base64}",
                    "Miner-Public-Key": public_key_hex,
                    "Miner": keypair.ss58_address,
                    "X-Payload": payload,
                    "Content-Type": "application/json",
                },
            )

        if response.status_code == 200:
            return True, None

        error_detail = "Unknown error"
        try:
            error_data = response.json()
            error_detail = error_data.get("detail", str(response.text))
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"

        return False, error_detail

    except httpx.TimeoutException:
        return False, "Request timed out"
    except httpx.ConnectError:
        return False, "Connection failed"
    except Exception as e:
        return False, str(e)
