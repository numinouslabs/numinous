from typing import NoReturn

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from neurons.validator.models.track import TrackEnum
from neurons.validator.sandbox.signing_proxy.track_config import TRACK_ALLOWED_PREFIXES

console = Console()

TRACK_CHOICES = [track.value for track in TrackEnum]
SCORED_AGENT_TRACK = TrackEnum.SIGNAL.value
DEFAULT_CREDENTIAL_TRACK = TrackEnum.SIGNAL.value


def _show_track_allowlist(track: str) -> None:
    allowed_prefixes = TRACK_ALLOWED_PREFIXES.get(track, [])
    allowed_display = "\n".join(f"  • [cyan]{prefix}[/cyan]" for prefix in allowed_prefixes)
    console.print()
    console.print(
        Panel.fit(
            f"[yellow]Track {track} — Allowed gateway endpoints:[/yellow]\n\n"
            f"{allowed_display}\n\n"
            "[dim]Your agent will get 403 for any other endpoint.[/dim]",
            border_style="yellow",
        )
    )


def _reject_dead_track() -> NoReturn:
    console.print()
    console.print(
        Panel.fit(
            f"[red]✗ {TrackEnum.MAIN.value} is not a valid upload target[/red]\n\n"
            f"[yellow]No events are broadcast on {TrackEnum.MAIN.value}, so the agent "
            "would never run.[/yellow]\n\n"
            f"[dim]Upload to {SCORED_AGENT_TRACK} instead — it is the default, so you "
            "can simply drop the --track flag.[/dim]",
            border_style="red",
        )
    )
    console.print()
    raise click.Abort()


def _confirm_dead_track() -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[yellow]No events are broadcast on {TrackEnum.MAIN.value} — an agent "
            "uploaded there never runs and earns nothing.[/yellow]\n\n"
            f"[dim]{SCORED_AGENT_TRACK} is the only live track. "
            "See docs/miner-setup.md.[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    if not Confirm.ask(
        f"[yellow]Continue with {TrackEnum.MAIN.value} anyway?[/yellow]", default=False
    ):
        console.print("[dim]Cancelled[/dim]")
        raise click.Abort()


def resolve_agent_track(track: str | None, allow_dead_track: bool) -> str:
    if not track:
        _show_track_allowlist(SCORED_AGENT_TRACK)
        return SCORED_AGENT_TRACK

    track = track.upper()

    if track not in TRACK_CHOICES:
        console.print(f"[red]✗ Unknown track:[/red] {track}")
        console.print(f"[dim]Valid tracks:[/dim] {', '.join(TRACK_CHOICES)}")
        raise click.Abort()

    if track == TrackEnum.MAIN.value:
        if not allow_dead_track:
            _reject_dead_track()

        _confirm_dead_track()
        return track

    _show_track_allowlist(track)
    return track


def prompt_credential_track(show_fallback_note: bool) -> str:
    if show_fallback_note:
        console.print()
        console.print(
            f"[dim]{DEFAULT_CREDENTIAL_TRACK} is the only live track — link here unless you "
            f"know you need otherwise.\nA key linked for {TrackEnum.MAIN.value} instead acts "
            "as a fallback for every track.[/dim]"
        )

    console.print()
    return Prompt.ask(
        "[bold cyan]Select track[/bold cyan]",
        choices=TRACK_CHOICES,
        default=DEFAULT_CREDENTIAL_TRACK,
    )
