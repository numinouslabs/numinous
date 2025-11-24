import base64
import time
import typing
from datetime import datetime, timezone
from pathlib import Path

import click
import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection

console = Console()


@click.command()
@click.option("--wallet", "-w", type=str, help="Wallet name (interactive prompt if not provided)")
@click.option("--hotkey", "-k", type=str, help="Hotkey name (interactive prompt if not provided)")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment (interactive prompt if not provided)",
)
@click.option(
    "--wallet-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Custom wallet directory path (default: ~/.bittensor/wallets)",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Limit number of agents to retrieve (default: 10)",
)
@click.option(
    "--offset",
    "-o",
    type=int,
    default=0,
    help="Offset for pagination (default: 0)",
)
def list_agents(
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
    limit: int = 10,
    offset: int = 0,
) -> None:
    """List your uploaded miner agents

    \b
    Examples:
      # Interactive mode
      numi list-agents

      # List agents for specific wallet/env
      numi list-agents -w miner1 -k default -e prod

      # Pagination
      numi list-agents --limit 50 --offset 0
    """

    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]✨ Numinous - List Agents[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Prompt for environment if not provided
    if not env:
        console.print()
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    console.print(f"[dim]Network:[/dim] [yellow]{env.upper()}[/yellow]")
    console.print()

    # Prompt for wallet if not provided
    console.print()
    if not wallet or not hotkey:
        console.print(
            Panel.fit(
                "[yellow]Select wallet to list agents for[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        wallet, hotkey = prompt_wallet_selection(wallet_path)

    # Load wallet
    console.print()
    with console.status(f"[cyan]Loading wallet {wallet}/{hotkey}...[/cyan]"):
        keypair = load_keypair(wallet, hotkey, wallet_path)

    if not keypair:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ Failed to load wallet:[/red] {wallet}/{hotkey}\n"
                "[yellow]💡 Tip:[/yellow] Make sure the wallet exists and is properly configured",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(f"[green]✓[/green] Loaded wallet: [yellow]{keypair.ss58_address}[/yellow]")

    # Track the globally active agent ID (determined on first page)
    global_active_agent_id = None
    now = datetime.now(timezone.utc)

    while True:
        # Create signature with hotkey:timestamp
        timestamp = int(time.time())
        payload = f"{keypair.ss58_address}:{timestamp}"
        signature = keypair.sign(payload.encode())
        signature_base64 = base64.b64encode(signature).decode()
        public_key_hex = keypair.public_key.hex()

        # Fetch agents
        api_url = ENV_URLS[env]
        console.print()

        try:
            with console.status(
                f"[cyan]Fetching agents from {env.upper()}...[/cyan]", spinner="dots"
            ):
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(
                        f"{api_url}/api/v3/miner/agents",
                        params={"limit": limit, "offset": offset},
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
                total_count = result.get("total_count", 0)

                console.print()
                if not agents:
                    console.print(
                        Panel.fit(
                            "[yellow]No agents found for this miner.[/yellow]",
                            border_style="yellow",
                        )
                    )
                    console.print()
                    return

                console.print(
                    Panel.fit(
                        f"[bold green]✓ Found {len(agents)} agents (Total: {total_count})[/bold green]\n"
                        f"[dim]Showing {offset+1}-{min(offset+len(agents), total_count)} of {total_count}[/dim]",
                        border_style="green",
                    )
                )
                console.print()

                agents.sort(key=lambda x: x.get("version_number", 0), reverse=True)

                if global_active_agent_id is None and offset == 0:
                    for agent in agents:
                        activated_at_str = agent.get("activated_at")
                        if activated_at_str:
                            try:
                                activated_at = datetime.fromisoformat(
                                    activated_at_str.replace("Z", "+00:00")
                                )
                                if activated_at <= now:
                                    global_active_agent_id = agent.get("version_id")
                                    break
                            except (ValueError, TypeError):
                                continue

                table = Table(
                    show_header=True,
                    header_style="bold cyan",
                    box=box.ROUNDED,
                    title=f"🤖 Miner Agents ({env.upper()})",
                    title_style="bold magenta",
                )

                table.add_column("Version", justify="right", style="cyan")
                table.add_column("Name", style="white")
                table.add_column("Status", justify="center")
                table.add_column("Created At", style="dim")
                table.add_column("Activated At", style="dim")
                table.add_column("ID", style="dim", no_wrap=True)

                for agent in agents:
                    version = str(agent.get("version_number", "N/A"))
                    name = agent.get("agent_name", "N/A")
                    version_id = agent.get("version_id", "N/A")

                    created_at = agent.get("created_at", "")
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            created_at = dt.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            pass

                    activated_at_raw = agent.get("activated_at")
                    activated_at_display = "Pending"
                    is_active_time = False

                    if activated_at_raw:
                        try:
                            dt = datetime.fromisoformat(activated_at_raw.replace("Z", "+00:00"))
                            activated_at_display = dt.strftime("%Y-%m-%d %H:%M")
                            if dt <= now:
                                is_active_time = True
                        except Exception:
                            pass

                    status = ""
                    if version_id == global_active_agent_id:
                        status = "[bold green]ACTIVE[/bold green]"
                    elif is_active_time:
                        status = "[yellow]Old[/yellow]"
                    else:
                        status = "[blue]Pending[/blue]"

                    table.add_row(
                        version, name, status, created_at, activated_at_display, version_id
                    )

                console.print(table)
                console.print()

                if offset + limit < total_count:
                    if Confirm.ask("[bold cyan]Show more agents?[/bold cyan]", default=True):
                        offset += limit
                        continue
                    else:
                        console.print(
                            "[dim]💡 Tip: Use [cyan]numi inspect-agent[/cyan] to view or download "
                            "any activated agent code![/dim]"
                        )
                        console.print()
                        break
                else:
                    console.print(
                        "[dim]💡 Tip: Use [cyan]numi inspect-agent[/cyan] to view or download "
                        "any activated agent code![/dim]"
                    )
                    console.print()
                    break

            elif response.status_code == 401:
                console.print()
                console.print(
                    Panel.fit(
                        "[red]✗ Unauthorized[/red]\n\n" "Check your wallet credentials.",
                        border_style="red",
                    )
                )
                console.print()
                raise click.Abort()

            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except Exception:
                    pass

                console.print()
                console.print(
                    Panel.fit(
                        f"[red]✗ Request failed ({response.status_code})[/red]\n\n" f"{error_msg}",
                        border_style="red",
                    )
                )
                console.print()
                raise click.Abort()

        except httpx.RequestError as e:
            console.print()
            console.print(
                Panel.fit(
                    f"[red]✗ Connection failed[/red]\n\n" f"{str(e)}",
                    border_style="red",
                )
            )
            console.print()
            raise click.Abort()


if __name__ == "__main__":
    list_agents(standalone_mode=True)
