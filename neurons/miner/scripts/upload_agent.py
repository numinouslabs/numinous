import base64
import hashlib
import typing
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


def list_available_agents() -> list[Path]:
    agent_files = []
    agents_dir = Path("neurons/miner/agents")

    if agents_dir.exists() and agents_dir.is_dir():
        for file in agents_dir.glob("*.py"):
            if file.is_file():
                agent_files.append(file)

    return sorted(agent_files)


def prompt_agent_selection() -> Path:
    available_agents = list_available_agents()

    if not available_agents:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ No agent files found in neurons/miner/agents/[/red]\n"
                "[yellow]💡 Tip:[/yellow] Place your agent files (.py) there",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="📁 Available Agents (neurons/miner/agents/)",
        title_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="green")

    for idx, agent_path in enumerate(available_agents, 1):
        table.add_row(str(idx), agent_path.name)

    console.print()
    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask("[bold cyan]Select agent file[/bold cyan]", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_agents):
                return available_agents[idx]
            else:
                console.print(
                    f"[red]✗[/red] Invalid choice. Please enter 1-{len(available_agents)}"
                )
        except ValueError:
            console.print("[red]✗[/red] Please enter a number")


def find_agent_file(agent_file: str) -> typing.Optional[Path]:
    agent_path = Path(agent_file)

    if agent_path.is_absolute() or agent_path.exists():
        return agent_path

    miner_path = Path("miner/agents") / agent_file
    if miner_path.exists():
        return miner_path

    neurons_path = Path("neurons/miner/agents") / agent_file
    if neurons_path.exists():
        return neurons_path

    return None


@click.command()
@click.option(
    "--agent-file", "-f", type=str, help="Agent file path (interactive prompt if not provided)"
)
@click.option("--wallet", "-w", type=str, help="Wallet name (interactive prompt if not provided)")
@click.option("--hotkey", "-k", type=str, help="Hotkey name (interactive prompt if not provided)")
@click.option("--name", "-n", type=str, help="Agent name (interactive prompt if not provided)")
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
def upload(
    agent_file: typing.Optional[str] = None,
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    name: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
) -> None:
    """Upload your forecasting agent to Numinous

    \b
    Examples:
      # Interactive mode (prompts for everything)
      numi upload-agent

      # Upload specific file
      numi upload-agent --agent-file my_agent.py

      # Upload to production with specific wallet
      numi upload-agent -f my_agent.py --env prod --wallet miner1 --hotkey default

      # Quick upload with all options
      numi upload-agent -f my_agent.py -e prod -w miner1 -k default -n "My Forecaster"

      # Use custom wallet directory
      numi upload-agent -f my_agent.py --wallet-path /path/to/custom/wallets
    """

    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]✨ Numinous - Agent Upload[/bold cyan]",
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

    # Prompt for agent file if not provided
    if not agent_file:
        agent_path = prompt_agent_selection()
    else:
        agent_path = find_agent_file(agent_file)
        if not agent_path or not agent_path.exists():
            console.print()
            console.print(
                Panel.fit(
                    f"[red]✗ Agent file not found:[/red] {agent_file}\n"
                    "[yellow]💡 Tip:[/yellow] Place it in neurons/miner/agents/ or provide full path",
                    border_style="red",
                )
            )
            console.print()
            raise click.Abort()

    console.print(f"[green]✓[/green] Found agent: [cyan]{agent_path.name}[/cyan]")

    # Prompt for agent name if not provided
    if not name:
        console.print()
        name = Prompt.ask("[bold cyan]Agent name[/bold cyan]")
        if not name:
            console.print("[red]✗[/red] Agent name is required")
            raise click.Abort()

    # Prompt for wallet if not provided
    if not wallet or not hotkey:
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

    # Read agent file
    with open(agent_path, "rb") as f:
        file_content = f.read()

    # Create signature
    file_hash = hashlib.sha256(file_content).hexdigest()
    payload = f"{keypair.ss58_address}:{file_hash}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    # Show upload summary
    console.print()
    summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="white")
    summary.add_row("📝 Agent", name)
    summary.add_row("📄 File", agent_path.name)
    summary.add_row("📦 Size", f"{len(file_content):,} bytes")
    summary.add_row("🌐 Network", env.upper())
    summary.add_row("👤 Wallet", f"{wallet}/{hotkey}")
    summary.add_row("🔑 Address", keypair.ss58_address[:16] + "..." + keypair.ss58_address[-8:])

    console.print(Panel.fit(summary, title="📋 Upload Summary", border_style="cyan"))
    console.print()

    # Confirm upload (only for prod)
    if env == "prod":
        if not Confirm.ask("[yellow]⚠️  Upload to PRODUCTION?[/yellow]", default=True):
            console.print("[dim]Upload cancelled[/dim]")
            raise click.Abort()
        console.print()

    # Upload
    api_url = ENV_URLS[env]

    with console.status(f"[cyan]Uploading to {env.upper()}...[/cyan]", spinner="dots"):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{api_url}/api/v3/miner/upload_agent",
                    files={"agent_file": (agent_path.name, file_content)},
                    data={"name": name},
                    headers={
                        "Authorization": f"Bearer {signature_base64}",
                        "Miner-Public-Key": public_key_hex,
                        "Miner": keypair.ss58_address,
                        "X-Payload": payload,
                    },
                )

            if response.status_code == 200:
                result = response.json()
                console.print()
                console.print(
                    Panel.fit(
                        f"[bold green]✓ Upload successful![/bold green]\n\n"
                        f"[dim]Agent ID:[/dim] [cyan]{result.get('version_id', 'N/A')}[/cyan]\n"
                        f"[dim]Network:[/dim] [yellow]{env.upper()}[/yellow]\n\n"
                        f"[yellow]⚠️  Remember to link services for this new code![/yellow]\n"
                        f"[cyan]Run: numi services link[/cyan]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                console.print()
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
                        f"[red]✗ Upload failed ({response.status_code})[/red]\n\n" f"{error_msg}",
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
    upload(standalone_mode=True)
