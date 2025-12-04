import base64
import time
import typing
from pathlib import Path
from uuid import UUID

import click
import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from neurons.miner.scripts.numinous_config import ENV_URLS
from neurons.miner.scripts.wallet_utils import load_keypair, prompt_wallet_selection

console = Console()


def validate_run_id(run_id: str) -> str:
    try:
        UUID(run_id)
        return run_id
    except ValueError:
        raise ValueError(f"Invalid run_id format: {run_id}. Must be a valid UUID.")


@click.command()
@click.option(
    "--run-id",
    "-r",
    type=str,
    help="Run ID (UUID) to fetch logs for (interactive prompt if not provided)",
)
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
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Save logs to file instead of displaying",
)
def fetch_logs(
    run_id: typing.Optional[str] = None,
    wallet: typing.Optional[str] = None,
    hotkey: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    wallet_path: typing.Optional[Path] = None,
    output: typing.Optional[Path] = None,
) -> None:
    """Fetch agent run logs from Numinous

    \b
    Examples:
      # Fully interactive mode (prompts for run-id, environment, and wallet)
      numi fetch-logs

      # Fetch logs with interactive prompts
      numi fetch-logs --run-id "123e4567-e89b-12d3-a456-426614174000"

      # Fully specified (no prompts)
      numi fetch-logs -r "123e4567-e89b-12d3-a456-426614174000" -e prod -w miner1 -k default

      # Fetch from test environment
      numi fetch-logs -r "123e4567-e89b-12d3-a456-426614174000" --env test

      # Save logs to file
      numi fetch-logs -r "123e4567-e89b-12d3-a456-426614174000" -e prod -o logs.txt

      # Use custom wallet directory
      numi fetch-logs -r "123e4567-e89b-12d3-a456-426614174000" --wallet-path /path/to/wallets
    """

    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]✨ Numinous - Fetch Agent Logs[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Prompt for run_id if not provided
    if not run_id:
        console.print()
        run_id = Prompt.ask("[bold cyan]Enter Run ID (UUID)[/bold cyan]")
        if not run_id:
            console.print("[red]✗[/red] Run ID is required")
            raise click.Abort()

    # Validate run_id
    try:
        run_id = validate_run_id(run_id)
    except ValueError as e:
        console.print()
        console.print(
            Panel.fit(
                f"[red]✗ {str(e)}[/red]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    console.print(f"[dim]Run ID:[/dim] [cyan]{run_id}[/cyan]")

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
    console.print(
        Panel.fit(
            "[yellow]Select wallet from the miner whose run ID you are fetching logs for[/yellow]",
            border_style="yellow",
        )
    )
    console.print()
    if not wallet or not hotkey:
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

    # Create signature with run_id + timestamp
    timestamp = int(time.time())
    payload = f"{keypair.ss58_address}:{timestamp}"
    signature = keypair.sign(payload.encode())
    signature_base64 = base64.b64encode(signature).decode()
    public_key_hex = keypair.public_key.hex()

    # Fetch logs
    api_url = ENV_URLS[env]
    console.print()
    with console.status(f"[cyan]Fetching logs from {env.upper()}...[/cyan]", spinner="dots"):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{api_url}/api/v3/miner/logs/{run_id}",
                    headers={
                        "Authorization": f"Bearer {signature_base64}",
                        "Miner-Public-Key": public_key_hex,
                        "Miner": keypair.ss58_address,
                        "X-Payload": payload,
                    },
                )

            if response.status_code == 200:
                result = response.json()
                log_content = result.get("log_content", "")
                metadata = result.get("metadata")
                returned_run_id = result.get("run_id")

                console.print()
                console.print(
                    Panel.fit(
                        "[bold green]✓ Logs fetched successfully![/bold green]",
                        border_style="green",
                    )
                )
                console.print()

                # Display metadata if available
                if metadata:
                    meta_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
                    meta_table.add_column(style="dim", width=25)
                    meta_table.add_column(style="white")

                    # Add run_id first
                    if returned_run_id:
                        meta_table.add_row("Run ID", str(returned_run_id))

                    # Add all metadata fields
                    meta_table.add_row("Event ID", str(metadata.get("event_id", "N/A")))
                    meta_table.add_row("Validator UID", str(metadata.get("vali_uid", "N/A")))

                    validator_hotkey = str(metadata.get("vali_hotkey", "N/A"))
                    if len(validator_hotkey) > 50:
                        validator_hotkey = validator_hotkey[:16] + "..." + validator_hotkey[-8:]
                    meta_table.add_row("Validator Hotkey", validator_hotkey)

                    meta_table.add_row("Version ID", str(metadata.get("version_id", "N/A")))
                    meta_table.add_row("Status", str(metadata.get("status", "N/A")))
                    meta_table.add_row("Is Final", str(metadata.get("is_final", "N/A")))

                    # Add prediction fields if available
                    prediction = metadata.get("prediction")
                    if prediction:
                        meta_table.add_row(
                            "Interval Prediction",
                            str(prediction.get("interval_agg_prediction", "N/A")),
                        )
                        meta_table.add_row(
                            "Interval DateTime", str(prediction.get("interval_datetime", "N/A"))
                        )
                        meta_table.add_row(
                            "Submitted At", str(prediction.get("submitted_at", "N/A"))
                        )
                    else:
                        meta_table.add_row("Interval Prediction", "N/A")
                        meta_table.add_row("Interval DateTime", "N/A")
                        meta_table.add_row("Submitted At", "N/A")

                    console.print(
                        Panel.fit(meta_table, title="📊 Run Metadata", border_style="cyan")
                    )
                    console.print()

                # Save or display logs
                if output:
                    output.write_text(log_content)
                    console.print(f"[green]✓[/green] Logs saved to: [cyan]{output}[/cyan]")
                    console.print()
                else:
                    # Display logs with syntax highlighting
                    console.print(
                        Panel(
                            Syntax(log_content, "text", theme="monokai", word_wrap=True),
                            title="📜 Log Content",
                            border_style="blue",
                            padding=(1, 2),
                        )
                    )
                    console.print()

            elif response.status_code == 401:
                console.print()
                console.print(
                    Panel.fit(
                        "[red]✗ Unauthorized[/red]\n\n"
                        "This run_id does not belong to your miner or you don't have permission to access it.",
                        border_style="red",
                    )
                )
                console.print()
                raise click.Abort()

            elif response.status_code == 404:
                console.print()
                console.print(
                    Panel.fit(
                        "[red]✗ Log file not found[/red]\n\n" "Run ID does not exist",
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
    fetch_logs(standalone_mode=True)
