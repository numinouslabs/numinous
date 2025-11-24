from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

console = Console()

GATEWAY_ENV_PATH = Path("neurons/miner/gateway/.env")


def check_env_vars() -> dict[str, bool]:
    if not GATEWAY_ENV_PATH.exists():
        return {"CHUTES_API_KEY": False, "DESEARCH_API_KEY": False}

    env_content = GATEWAY_ENV_PATH.read_text()
    return {
        "CHUTES_API_KEY": "CHUTES_API_KEY=" in env_content
        and not env_content.split("CHUTES_API_KEY=")[1].split("\n")[0].strip() == "",
        "DESEARCH_API_KEY": "DESEARCH_API_KEY=" in env_content
        and not env_content.split("DESEARCH_API_KEY=")[1].split("\n")[0].strip() == "",
    }


def setup_api_keys(force_all: bool = False) -> bool:
    console.print()
    console.print("[cyan]🔑 API Key Setup[/cyan]")
    console.print()
    console.print("[dim]You can get your API keys from:[/dim]")
    console.print("[dim]  • Chutes: [link=https://chutes.ai]https://chutes.ai[/link][/dim]")
    console.print("[dim]  • Desearch: [link=https://desearch.ai]https://desearch.ai[/link][/dim]")
    console.print()

    env_status = check_env_vars()

    chutes_key = None
    desearch_key = None

    if not env_status["CHUTES_API_KEY"] or force_all:
        chutes_key = Prompt.ask("[cyan]Chutes API Key[/cyan]")
        chutes_key = chutes_key.strip() if chutes_key else None

    if not env_status["DESEARCH_API_KEY"] or force_all:
        desearch_key = Prompt.ask("[cyan]Desearch API Key[/cyan]")
        desearch_key = desearch_key.strip() if desearch_key else None

    existing_content = ""
    if GATEWAY_ENV_PATH.exists():
        existing_content = GATEWAY_ENV_PATH.read_text()

    lines = existing_content.split("\n") if existing_content else []

    def update_or_add_key(lines: list[str], key: str, value: str) -> list[str]:
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}")
        return lines

    if chutes_key:
        lines = update_or_add_key(lines, "CHUTES_API_KEY", chutes_key)

    if desearch_key:
        lines = update_or_add_key(lines, "DESEARCH_API_KEY", desearch_key)

    try:
        GATEWAY_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

        new_content = "\n".join(lines)
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        GATEWAY_ENV_PATH.write_text(new_content)

        console.print()
        console.print(f"[green]✓[/green] API keys saved to [cyan]{GATEWAY_ENV_PATH}[/cyan]")
        console.print()

        return True

    except Exception as e:
        console.print()
        console.print(f"[red]✗[/red] Failed to save API keys: {e}")
        console.print()
        return False
