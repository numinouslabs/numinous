from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from rich.console import Console
from rich.prompt import Prompt

from neurons.validator.models.track import TrackEnum
from neurons.validator.sandbox.signing_proxy.track_config import TRACK_ALLOWED_PREFIXES

console = Console()

GATEWAY_ENV_PATH = Path("neurons/miner/gateway/.env")


class GatewayApiKey(NamedTuple):
    env_var: str
    display_name: str
    key_url: str
    gateway_route: str


API_KEYS = [
    GatewayApiKey(
        "OPENAI_API_KEY",
        "OpenAI",
        "https://platform.openai.com/api-keys",
        "/api/gateway/openai",
    ),
    GatewayApiKey(
        "OPENROUTER_API_KEY",
        "OpenRouter",
        "https://openrouter.ai",
        "/api/gateway/openrouter",
    ),
    GatewayApiKey(
        "NUMINOUS_SIGNALS_API_KEY",
        "Numinous Signals",
        "https://eversight.numinouslabs.io/api-keys",
        "/api/gateway/numinous-signals",
    ),
    GatewayApiKey(
        "LIGHTNING_ROD_API_KEY",
        "Lightning Rod",
        "https://lightningrod.ai",
        "/api/gateway/lightning-rod",
    ),
    GatewayApiKey("CHUTES_API_KEY", "Chutes", "https://chutes.ai", "/api/gateway/chutes"),
    GatewayApiKey("DESEARCH_API_KEY", "Desearch", "https://desearch.ai", "/api/gateway/desearch"),
    GatewayApiKey(
        "PERPLEXITY_API_KEY",
        "Perplexity",
        "https://www.perplexity.ai/settings/api",
        "/api/gateway/perplexity",
    ),
    GatewayApiKey("VERICORE_API_KEY", "Vericore", "https://vericore.ai", "/api/gateway/vericore"),
    GatewayApiKey(
        "UNUSUAL_WHALES_API_KEY",
        "Unusual Whales",
        "https://unusualwhales.com/pricing?product=api",
        "/api/gateway/unusual-whales",
    ),
]


def usable_on_signal(api_key: GatewayApiKey) -> bool:
    allowed_prefixes = TRACK_ALLOWED_PREFIXES[TrackEnum.SIGNAL.value]
    return any(
        prefix == api_key.gateway_route or prefix.startswith(f"{api_key.gateway_route}/")
        for prefix in allowed_prefixes
    )


class ApiKeyStatus(StrEnum):
    SET = "SET"
    SKIPPED = "SKIPPED"
    UNDECIDED = "UNDECIDED"


def _read_env_values() -> dict[str, str]:
    if not GATEWAY_ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for line in GATEWAY_ENV_PATH.read_text().splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue
        name, _, value = stripped_line.partition("=")
        values[name.strip()] = value.strip()

    return values


def read_api_key_statuses() -> dict[str, ApiKeyStatus]:
    values = _read_env_values()

    statuses: dict[str, ApiKeyStatus] = {}
    for api_key in API_KEYS:
        if api_key.env_var not in values:
            statuses[api_key.env_var] = ApiKeyStatus.UNDECIDED
        elif values[api_key.env_var]:
            statuses[api_key.env_var] = ApiKeyStatus.SET
        else:
            statuses[api_key.env_var] = ApiKeyStatus.SKIPPED

    return statuses


def undecided_api_keys() -> list[str]:
    return [
        env_var
        for env_var, status in read_api_key_statuses().items()
        if status is ApiKeyStatus.UNDECIDED
    ]


def _prompt_for_key(api_key: GatewayApiKey, current_status: ApiKeyStatus) -> str | None:
    hint = "Enter to keep" if current_status is ApiKeyStatus.SET else "Enter to skip"
    unused_note = "" if usable_on_signal(api_key) else " · not used on SIGNAL"
    value = Prompt.ask(
        f"[cyan]{api_key.display_name} API Key[/cyan] [dim]({hint}{unused_note})[/dim]",
        default="",
        show_default=False,
    ).strip()

    if value:
        return value

    if current_status is ApiKeyStatus.SET:
        return None

    return ""


def _write_keys(keys_to_set: dict[str, str]) -> bool:
    existing_content = GATEWAY_ENV_PATH.read_text() if GATEWAY_ENV_PATH.exists() else ""
    lines = existing_content.split("\n") if existing_content else []

    for env_var, value in keys_to_set.items():
        for index, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[index] = f"{env_var}={value}"
                break
        else:
            lines.append(f"{env_var}={value}")

    try:
        GATEWAY_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

        new_content = "\n".join(lines)
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"

        GATEWAY_ENV_PATH.write_text(new_content)
        return True

    except Exception as error:
        console.print()
        console.print(f"[red]Failed to save API keys: {error}[/red]")
        console.print()
        return False


def setup_api_keys(force_all: bool = False) -> bool:
    statuses = read_api_key_statuses()
    keys_to_prompt = [
        api_key
        for api_key in API_KEYS
        if force_all or statuses[api_key.env_var] is ApiKeyStatus.UNDECIDED
    ]

    if not keys_to_prompt:
        console.print()
        console.print("[dim]Every API key is already set or skipped — nothing to do.[/dim]")
        console.print()
        return True

    console.print()
    console.print("[cyan]API Key Setup[/cyan]")
    console.print()
    console.print("[dim]Skipped keys are remembered — you will not be asked again.[/dim]")
    console.print(
        "[dim]Only the services on the SIGNAL allowlist are reachable by a scored agent.[/dim]"
    )
    console.print()
    console.print("[dim]You can get your API keys from:[/dim]")
    for api_key in keys_to_prompt:
        note = "" if usable_on_signal(api_key) else " [dim](not used on SIGNAL)[/dim]"
        console.print(
            f"[dim]  - {api_key.display_name}: "
            f"[link={api_key.key_url}]{api_key.key_url}[/link][/dim]{note}"
        )
    console.print()

    keys_to_set: dict[str, str] = {}
    for api_key in keys_to_prompt:
        value = _prompt_for_key(api_key, statuses[api_key.env_var])
        if value is not None:
            keys_to_set[api_key.env_var] = value

    if not _write_keys(keys_to_set):
        return False

    console.print()
    console.print(f"[green]OK[/green] API keys saved to [cyan]{GATEWAY_ENV_PATH}[/cyan]")
    console.print()

    return True
