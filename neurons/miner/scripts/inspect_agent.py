import typing
from pathlib import Path
from uuid import UUID

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from neurons.miner.scripts.numinous_config import ENV_URLS

console = Console()


def validate_version_id(version_id: str) -> str:
    try:
        UUID(version_id)
        return version_id
    except ValueError:
        raise ValueError(f"Invalid version_id format: {version_id}. Must be a valid UUID.")


@click.command()
@click.option(
    "--version-id",
    "-v",
    type=str,
    help="Agent version ID (UUID) to inspect (interactive prompt if not provided)",
)
@click.option(
    "--env",
    "-e",
    type=click.Choice(["test", "prod"], case_sensitive=False),
    help="Network environment (interactive prompt if not provided)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Save code to file (interactive prompt if not provided)",
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Skip preview and download directly",
)
def inspect_agent(
    version_id: typing.Optional[str] = None,
    env: typing.Optional[str] = None,
    output: typing.Optional[Path] = None,
    no_preview: bool = False,
) -> None:
    """Inspect and download any activated agent code

    This command allows you to view or download the code of any activated
    agent in the network. Perfect for competitive analysis and learning!

    \b
    Examples:
      # Interactive mode
      numi inspect-agent

      # Inspect specific agent
      numi inspect-agent --version-id "123e4567-e89b-12d3-a456-426614174000"

      # Download directly without preview
      numi inspect-agent -v "123e4567-e89b-12d3-a456-426614174000" --no-preview -o agent.py

      # Inspect from production
      numi inspect-agent -v "123e4567-e89b-12d3-a456-426614174000" -e prod
    """

    # Print header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]✨ Numinous - Inspect Agent[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Prompt for version_id if not provided
    if not version_id:
        console.print()
        version_id = Prompt.ask("[bold cyan]Enter Agent Version ID (UUID)[/bold cyan]")
        if not version_id:
            console.print("[red]✗[/red] Version ID is required")
            raise click.Abort()

    # Validate version_id
    try:
        version_id = validate_version_id(version_id)
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

    console.print(f"[dim]Version ID:[/dim] [cyan]{version_id}[/cyan]")

    # Prompt for environment if not provided
    if not env:
        console.print()
        env_choice = Prompt.ask(
            "[bold cyan]Select environment[/bold cyan]", choices=["test", "prod"], default="test"
        )
        env = env_choice.lower()

    console.print(f"[dim]Network:[/dim] [yellow]{env.upper()}[/yellow]")
    console.print()

    # Fetch agent code
    api_url = ENV_URLS[env]
    console.print()

    try:
        with console.status(
            f"[cyan]Fetching agent code from {env.upper()}...[/cyan]", spinner="dots"
        ):
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{api_url}/api/v3/miner/agents/{version_id}/code",
                )

        if response.status_code == 200:
            code_content = response.text
            file_size = len(code_content.encode("utf-8"))

            console.print()
            console.print(
                Panel.fit(
                    f"[bold green]✓ Agent code retrieved successfully![/bold green]\n\n"
                    f"[dim]Size:[/dim] {file_size:,} bytes\n"
                    f"[dim]Lines:[/dim] {len(code_content.splitlines())}",
                    border_style="green",
                )
            )
            console.print()

            if not no_preview:
                show_preview = Confirm.ask(
                    "[bold cyan]Preview code in terminal?[/bold cyan]", default=True
                )

                if show_preview:
                    console.print()
                    total_lines = len(code_content.splitlines())
                    console.print(
                        Panel(
                            Syntax(code_content, "python", theme="one-dark", line_numbers=True),
                            title=f"📜 Agent Code ({total_lines} lines) - Scroll to view all",
                            border_style="blue",
                            padding=(1, 2),
                        )
                    )
                    console.print()

            if output:
                save_path = output
            else:
                should_download = Confirm.ask(
                    "[bold cyan]Download code to file?[/bold cyan]", default=True
                )

                if not should_download:
                    console.print("[dim]Done.[/dim]")
                    return

                console.print()
                default_path = f"./neurons/miner/agents/{version_id}.py"
                save_path_str = Prompt.ask("[bold cyan]Save to[/bold cyan]", default=default_path)
                save_path = Path(save_path_str)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(code_content)

            console.print()
            console.print(
                Panel.fit(
                    f"[bold green]✓ Code saved successfully![/bold green]\n\n"
                    f"[dim]Location:[/dim] [cyan]{save_path.absolute()}[/cyan]",
                    border_style="green",
                )
            )
            console.print()

        elif response.status_code == 404:
            console.print()
            console.print(
                Panel.fit(
                    "[red]✗ Agent not found or not yet activated[/red]\n\n"
                    "This agent either doesn't exist or hasn't been activated yet.\n"
                    "Only activated agents are publicly accessible.",
                    border_style="red",
                )
            )
            console.print()
            raise click.Abort()

        elif response.status_code == 503:
            console.print()
            console.print(
                Panel.fit(
                    "[red]✗ Service unavailable[/red]\n\n"
                    "Failed to retrieve agent code from storage.\n"
                    "Please try again later.",
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
    inspect_agent(standalone_mode=True)
