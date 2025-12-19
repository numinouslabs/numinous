import base64
import hashlib
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
from neurons.miner.scripts.wallet_utils import load_coldkey, load_keypair, prompt_wallet_selection

console = Console()

DESEARCH_API_URL = "https://api.desearch.ai/bt/miner/link"


def link_desearch_impl(
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
) -> None:
    """Link your Desearch account to your miner

    This command links your Desearch API account to your miner identity,
    allowing validators to attribute API costs to your account.

    \b
    Steps:
      1. Verify you have uploaded agent code
      2. Link your coldkey to Desearch account
      3. Store credentials in backend for validators

    \b
    Examples:
      # Interactive mode
      numi link-desearch

      # Specify wallet and environment
      numi link-desearch -w miner1 -k default -e prod
    """

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🔗 Link Desearch Account[/bold cyan]",
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
    with console.status("[cyan]Checking for uploaded agent code...[/cyan]"):
        latest_agent = _fetch_latest_agent(env, hotkey_keypair)

    if not latest_agent:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ No agent code found[/red]\n\n"
                "[yellow]You must upload agent code first:[/yellow]\n"
                "[cyan]numi upload-agent[/cyan]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    version_id = latest_agent["version_id"]
    agent_name = latest_agent["agent_name"]

    console.print(
        f"[green]✓[/green] Found agent: [cyan]{agent_name}[/cyan] (v{latest_agent['version_number']})"
    )
    console.print(f"[dim]Version ID:[/dim] {version_id}")

    version_hash = hashlib.sha256(version_id.encode()).hexdigest()
    console.print(f"[dim]Code hash:[/dim] {version_hash[:16]}...")

    version_signature = hotkey_keypair.sign(version_id.encode())
    version_signature_hex = version_signature.hex()

    console.print()
    console.print(
        Panel.fit(
            "[yellow]You will now link your Desearch account[/yellow]\n\n"
            "[dim]Required:[/dim] Desearch API key from https://console.desearch.ai",
            border_style="yellow",
        )
    )
    console.print()

    desearch_api_key = getpass("Enter Desearch API key: ")
    if not desearch_api_key:
        console.print("[red]✗[/red] API key is required")
        raise click.Abort()

    console.print()
    console.print("[cyan]Loading coldkey (password required)...[/cyan]")
    coldkey_keypair = load_coldkey(wallet, wallet_path)

    if not coldkey_keypair:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ Failed to load coldkey[/red]\n\n"
                "[yellow]Check your password and try again[/yellow]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(
        f"[green]✓[/green] Loaded coldkey: [yellow]{coldkey_keypair.ss58_address}[/yellow]"
    )

    coldkey_signature = coldkey_keypair.sign(desearch_api_key.encode())
    coldkey_signature_hex = coldkey_signature.hex()

    console.print()
    console.print("[bold cyan]Ready to link:[/bold cyan]")
    console.print(f"  [dim]Coldkey:[/dim] {coldkey_keypair.ss58_address[:16]}...")
    console.print(f"  [dim]Version ID:[/dim] {version_id}")
    console.print(f"  [dim]Agent:[/dim] {agent_name}")
    console.print(f"  [dim]Network:[/dim] {env.upper()}")
    console.print()

    if not Confirm.ask("[yellow]Proceed with linking?[/yellow]", default=True):
        console.print("[dim]Cancelled[/dim]")
        raise click.Abort()

    console.print()
    with console.status("[cyan]Linking to Desearch...[/cyan]"):
        desearch_success = _link_to_desearch(
            desearch_api_key, coldkey_keypair.ss58_address, coldkey_signature_hex
        )

    if not desearch_success:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ Desearch linking failed[/red]\n\n"
                "[yellow]Check your API key and try again[/yellow]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print("[green]✓[/green] Desearch account linked")

    console.print()
    with console.status("[cyan]Storing credentials in backend...[/cyan]"):
        backend_success = _store_backend_credentials(
            env,
            hotkey_keypair,
            coldkey_keypair.ss58_address,
            version_hash,
            version_signature_hex,
        )

    if not backend_success:
        console.print()
        console.print(
            Panel.fit(
                "[yellow]⚠️  Desearch linked but backend storage failed[/yellow]\n\n"
                "[dim]Your Desearch account is linked, but validators may not see it.[/dim]\n"
                "[yellow]Please contact support or try again.[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        raise click.Abort()

    console.print()
    console.print(
        Panel.fit(
            "[bold green]✓ Successfully linked Desearch account![/bold green]\n\n"
            f"[dim]Coldkey:[/dim] {coldkey_keypair.ss58_address[:16]}...\n"
            f"[dim]Code hash:[/dim] {version_hash[:16]}...\n\n"
            "[yellow]⚠️  Remember to re-link after uploading new agent code![/yellow]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def _fetch_latest_agent(env: str, keypair) -> typing.Optional[dict]:
    api_url = ENV_URLS[env]
    timestamp = int(time.time())
    payload = f"{keypair.ss58_address}:{timestamp}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{api_url}/api/v3/miner/agents",
                params={"limit": 1, "offset": 0},
                headers={
                    "Authorization": f"Bearer {signature_base64}",
                    "Miner-Public-Key": public_key_hex,
                    "Miner": keypair.ss58_address,
                    "X-Payload": payload,
                },
            )

        if response.status_code == 200:
            result = response.json()
            agents = result.get("items", [])
            if agents:
                return agents[0]
        return None
    except Exception:
        return None


def _link_to_desearch(api_key: str, coldkey_ss58: str, signature_hex: str) -> bool:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                DESEARCH_API_URL,
                json={"coldkey_ss58": coldkey_ss58, "signature_hex": signature_hex},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        return response.status_code == 200
    except Exception:
        return False


def _store_backend_credentials(
    env: str, keypair, coldkey: str, code_hash: str, code_signature_hex: str
) -> bool:
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
                    "service_name": "desearch",
                    "auth_type": "signature",
                    "credential_data": {
                        "coldkey": coldkey,
                        "code_hash": code_hash,
                        "code_signature_hex": code_signature_hex,
                    },
                },
                headers={
                    "Authorization": f"Bearer {signature_base64}",
                    "Miner-Public-Key": public_key_hex,
                    "Miner": keypair.ss58_address,
                    "X-Payload": payload,
                },
            )
        return response.status_code == 200
    except Exception:
        return False
