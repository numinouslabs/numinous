import base64
import time
import typing
from getpass import getpass
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection

console = Console()


def link_chutes_impl(
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
) -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🔗 Link Chutes API Key[/bold cyan]",
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
            "[bold cyan]Chutes API Key Setup[/bold cyan]\n\n"
            "[dim]Get your API key from:[/dim] [cyan]https://chutes.ai/app[/cyan]\n\n"
            "[yellow]Budget Tiers:[/yellow]\n"
            "  • Free (backend key): [dim]$0.01 per agent run[/dim]\n"
            "  • Paid (your key): [green]$0.10 per agent run[/green]\n\n"
            "[dim]Your API key will be securely stored in AWS Secrets Manager.[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    api_key = getpass("Enter your Chutes API key: ")
    if not api_key or not api_key.strip():
        console.print("[red]✗[/red] API key cannot be empty")
        raise click.Abort()

    console.print()
    console.print("[bold cyan]Ready to link:[/bold cyan]")
    console.print(f"  [dim]Hotkey:[/dim] {hotkey_keypair.ss58_address[:16]}...")
    console.print("  [dim]Service:[/dim] Chutes")
    console.print("  [dim]Budget:[/dim] $0.10 per run")
    console.print(f"  [dim]Network:[/dim] {env.upper()}")
    console.print()

    if not Confirm.ask("[yellow]Proceed with linking?[/yellow]", default=True):
        console.print("[dim]Cancelled[/dim]")
        raise click.Abort()

    console.print()
    with console.status("[cyan]Storing credentials in backend...[/cyan]"):
        success, error_msg = _store_chutes_credentials(env, hotkey_keypair, api_key)

    if not success:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to link Chutes API key[/red]\n\n"
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
            "[bold green]✓ Successfully linked Chutes API key![/bold green]\n\n"
            f"[dim]Hotkey:[/dim] {hotkey_keypair.ss58_address[:16]}...\n"
            f"[dim]Service:[/dim] Chutes\n"
            f"[dim]Budget:[/dim] $0.10 per agent run\n\n"
            "[yellow]Your agent runs will now use your Chutes API key[/yellow]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _store_chutes_credentials(
    env: str, keypair, api_key: str
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
                    "service_name": "chutes",
                    "auth_type": "api_key",
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
