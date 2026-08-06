import click
from rich.console import Console
from rich.prompt import Prompt

console = Console()

MINIMUM_API_KEY_LENGTH = 8
MAXIMUM_API_KEY_LENGTH = 512


def find_api_key_problem(api_key: str) -> str | None:
    if not api_key:
        return "API key cannot be empty"

    if any(character.isspace() for character in api_key):
        return "API key contains spaces or line breaks — did you paste something else?"

    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as error:
        offending_character = api_key[error.start]
        return (
            f"API key contains a non-ASCII character ({offending_character!r}) "
            f"at position {error.start} — did you paste something else?"
        )

    if len(api_key) < MINIMUM_API_KEY_LENGTH:
        return f"API key is only {len(api_key)} characters long — that looks too short"

    if len(api_key) > MAXIMUM_API_KEY_LENGTH:
        return (
            f"API key is {len(api_key)} characters long, longer than the "
            f"{MAXIMUM_API_KEY_LENGTH} character maximum — did you paste something else?"
        )

    return None


def prompt_for_api_key(prompt: str) -> str:
    while True:
        api_key = Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]", default="", show_default=False)
        api_key = api_key.strip()

        if not api_key:
            console.print("[red]✗[/red] API key cannot be empty")
            raise click.Abort()

        problem = find_api_key_problem(api_key)
        if not problem:
            return api_key

        console.print(f"[red]✗[/red] {problem}")
        console.print()
