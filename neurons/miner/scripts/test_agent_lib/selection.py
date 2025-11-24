import uuid
from datetime import datetime
from pathlib import Path

import click
import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

API_BASE_URL = "https://ifgames.win"


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
    table.add_column("Size", style="yellow", justify="right")

    for idx, agent_path in enumerate(available_agents, 1):
        size_kb = agent_path.stat().st_size / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        table.add_row(str(idx), agent_path.name, size_str)

    console.print()
    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask("[bold cyan]Select agent[/bold cyan]", default="1")
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


def find_agent_file(agent_file: str) -> Path | None:
    agent_path = Path(agent_file)

    if agent_path.is_absolute() and agent_path.exists():
        return agent_path

    if agent_path.exists():
        return agent_path

    miner_path = Path("miner/agents") / agent_file
    if miner_path.exists():
        return miner_path

    neurons_path = Path("neurons/miner/agents") / agent_file
    if neurons_path.exists():
        return neurons_path

    return None


def select_agent(agent_file: str | None) -> Path:
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

    console.print(f"[green]✓[/green] Selected agent: [cyan]{agent_path.name}[/cyan]")
    console.print()

    return agent_path


def fetch_live_events(limit: int = 10, offset: int = 0) -> list[dict]:
    try:
        with console.status("[cyan]Fetching live events from Numinous...[/cyan]"):
            response = httpx.get(
                f"{API_BASE_URL}/api/v2/events",
                params={"from_date": 0, "offset": offset, "limit": limit},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        events = data.get("items", [])
        if not events:
            return []

        parsed_events = []
        for event in events:
            cutoff_value = event.get("cutoff")
            if isinstance(cutoff_value, (int, float)):
                cutoff_str = datetime.fromtimestamp(cutoff_value).isoformat() + "Z"
            else:
                cutoff_str = str(cutoff_value) if cutoff_value else ""

            parsed_events.append(
                {
                    "event_id": event.get("event_id"),
                    "title": event.get("title", ""),
                    "description": event.get("description", ""),
                    "market_type": event.get("market_type", "binary"),
                    "cutoff": cutoff_str,
                    "metadata": event.get("event_metadata", {}),
                }
            )

        return parsed_events

    except httpx.HTTPError as e:
        console.print(f"[red]✗ Failed to fetch events: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]✗ Unexpected error: {e}[/red]")
        return []


def format_cutoff(cutoff_value: int | float | str) -> str:
    try:
        if isinstance(cutoff_value, (int, float)):
            dt = datetime.fromtimestamp(cutoff_value)
            return dt.strftime("%b %d, %Y %H:%M UTC")

        if isinstance(cutoff_value, str):
            try:
                timestamp = float(cutoff_value)
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime("%b %d, %Y %H:%M UTC")
            except (ValueError, OSError):
                dt = datetime.fromisoformat(cutoff_value.replace("Z", "+00:00"))
                return dt.strftime("%b %d, %Y %H:%M UTC")

        return str(cutoff_value) if cutoff_value else "N/A"
    except Exception:
        return str(cutoff_value) if cutoff_value else "N/A"


def prompt_event_selection_mode() -> str:
    console.print()
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column(style="dim", width=4)
    table.add_column(style="white")

    table.add_row("1", "Single event test (quick)")
    table.add_row("2", "Multiple events test (batch)")

    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask("[bold cyan]Select test mode[/bold cyan]", default="1")
        if choice in ["1", "2"]:
            return choice
        console.print("[red]✗[/red] Please enter 1 or 2")


def prompt_event_source() -> str:
    console.print()
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column(style="dim", width=4)
    table.add_column(style="white")

    table.add_row("1", "Live events (from Numinous)")
    table.add_row("2", "Manual input (custom event)")

    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask("[bold cyan]Select event source[/bold cyan]", default="1")
        if choice in ["1", "2"]:
            return choice
        console.print("[red]✗[/red] Please enter 1 or 2")


def display_events_table(
    events: list[dict], page_num: int = 1, page_size: int = 20, has_more: bool = False
) -> None:
    """Display events in a table with pagination info"""
    page_info = f"📋 Live Events (Page {page_num})"
    if not has_more and page_num == 1:
        page_info = "📋 Live Events"

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title=page_info,
        title_style="bold magenta",
    )
    table.add_column("#", style="dim", width=6, no_wrap=True)  # Wider for larger numbers
    table.add_column("Title", style="green", max_width=80, no_wrap=False)  # Wrap at 80 chars
    table.add_column("Market Type", style="yellow", width=12, no_wrap=True)
    table.add_column("Cutoff", style="cyan", width=20, no_wrap=True)

    # Calculate starting index based on page
    start_idx = (page_num - 1) * page_size + 1

    for idx, event in enumerate(events, start=start_idx):
        title = event["title"]  # No truncation!
        market_type = str(event.get("market_type", "LLM"))
        cutoff = format_cutoff(event.get("cutoff", ""))

        table.add_row(str(idx), title, market_type, cutoff)

    console.print()
    console.print(table)
    console.print()


def parse_selection_ranges(selection: str) -> list[int]:
    indices = set()
    parts = [p.strip() for p in selection.split(",")]

    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-", 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                if start_num <= end_num:
                    indices.update(range(start_num, end_num + 1))
                else:
                    pass
            except ValueError:
                pass
        else:
            try:
                indices.add(int(part))
            except ValueError:
                pass

    return sorted(indices)


def paginated_event_selector(allow_multiple: bool = False, page_size: int = 20) -> list[dict]:
    current_page = 1
    offset = 0
    all_loaded_events = {}  # {global_idx: event_dict}

    # Fetch first page
    current_events = fetch_live_events(limit=page_size, offset=offset)

    if not current_events:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ No live events available[/red]\n"
                "[yellow]💡 Tip:[/yellow] Try manual input or check back later",
                border_style="red",
            )
        )
        console.print()
        return []

    start_idx = (current_page - 1) * page_size + 1
    for idx, event in enumerate(current_events):
        all_loaded_events[start_idx + idx] = event

    while True:
        # Check if there might be more pages
        has_more = len(current_events) == page_size

        # Calculate global index range for current page
        start_idx = (current_page - 1) * page_size + 1
        end_idx = start_idx + len(current_events) - 1

        # Calculate range of ALL loaded events
        loaded_min = min(all_loaded_events.keys()) if all_loaded_events else 1
        loaded_max = max(all_loaded_events.keys()) if all_loaded_events else end_idx

        # Display current page
        display_events_table(
            current_events, page_num=current_page, page_size=page_size, has_more=has_more
        )

        # Show options
        console.print("[dim]Options:[/dim]")
        if allow_multiple:
            console.print(
                f"  • Enter event numbers from [cyan]ANY loaded page ({loaded_min}-{loaded_max})[/cyan]"
            )
            console.print(f"  • [dim]Current page: {start_idx}-{end_idx}[/dim]")
            console.print("  • Examples: '1,3,5' or '1-30' or '5-15,20,25-30'")
            console.print(
                "  • Type [cyan]'all'[/cyan] to select ALL loaded events (across all pages)"
            )
        else:
            console.print(
                f"  • Enter event number [cyan]({loaded_min}-{loaded_max})[/cyan] from any loaded page"
            )
            console.print(f"  • [dim]Current page: {start_idx}-{end_idx}[/dim]")

        if has_more:
            console.print("  • Type [cyan]'next'[/cyan] to load and view next page")
        if current_page > 1:
            console.print("  • Type [cyan]'prev'[/cyan] to go to previous page")
        console.print()

        if allow_multiple:
            choice = Prompt.ask("[bold cyan]Select event(s) or navigate[/bold cyan]", default="all")
        else:
            choice = Prompt.ask("[bold cyan]Select event or navigate[/bold cyan]", default="1")

        choice_lower = choice.lower().strip()

        # Navigation: Next page
        if choice_lower == "next":
            if not has_more:
                console.print("[yellow]⚠ No more events available[/yellow]")
                console.print()
                continue

            offset += page_size
            current_page += 1
            new_events = fetch_live_events(limit=page_size, offset=offset)

            if not new_events:
                console.print("[yellow]⚠ No more events available[/yellow]")
                offset -= page_size
                current_page -= 1
                console.print()
                continue

            current_events = new_events

            # Add newly loaded events to cache
            start_idx = (current_page - 1) * page_size + 1
            for idx, event in enumerate(current_events):
                all_loaded_events[start_idx + idx] = event

            continue

        # Navigation: Previous page
        if choice_lower == "prev":
            if current_page <= 1:
                console.print("[yellow]⚠ Already on first page[/yellow]")
                console.print()
                continue

            offset -= page_size
            current_page -= 1
            current_events = fetch_live_events(limit=page_size, offset=offset)

            start_idx = (current_page - 1) * page_size + 1
            for idx, event in enumerate(current_events):
                all_loaded_events[start_idx + idx] = event

            continue

        # Selection: All (batch mode only) - now selects ALL loaded events
        if choice_lower == "all" and allow_multiple:
            # Return all loaded events in order
            return [all_loaded_events[idx] for idx in sorted(all_loaded_events.keys())]

        # Selection: Specific event(s)
        if allow_multiple:
            try:
                global_indices = parse_selection_ranges(choice)

                if not global_indices:
                    console.print("[red]✗[/red] No valid numbers found")
                    console.print("[dim]Example: 1,3,5 or 1-30 or 5-15,20-25[/dim]")
                    console.print()
                    continue

                selected = []
                invalid = []

                for global_idx in global_indices:
                    if global_idx in all_loaded_events:
                        selected.append(all_loaded_events[global_idx])
                    else:
                        invalid.append(global_idx)

                if invalid:
                    console.print(
                        f"[red]✗[/red] Events not loaded yet: {', '.join(map(str, invalid))}"
                    )
                    console.print(
                        f"[dim]Loaded range: {loaded_min}-{loaded_max} (navigate to load more)[/dim]"
                    )
                    console.print()
                    continue

                if not selected:
                    console.print("[red]✗[/red] No events selected")
                    console.print()
                    continue

                return selected

            except Exception as e:
                console.print(f"[red]✗[/red] Invalid format: {e}")
                console.print("[dim]Example: 1,3,5 or 1-30[/dim]")
                console.print()
        else:
            try:
                global_idx = int(choice)
                if global_idx in all_loaded_events:
                    return [all_loaded_events[global_idx]]
                console.print(f"[red]✗[/red] Event #{global_idx} not loaded yet")
                console.print(f"[dim]Loaded range: {loaded_min}-{loaded_max}[/dim]")
                console.print()
            except ValueError:
                console.print("[red]✗[/red] Please enter a number or 'next'/'prev'")
                console.print()


def select_single_event_from_live() -> dict | None:
    result = paginated_event_selector(allow_multiple=False, page_size=15)
    return result[0] if result else None


def select_multiple_events_from_live() -> list[dict]:
    return paginated_event_selector(allow_multiple=True, page_size=15)


def prompt_manual_event() -> dict:
    console.print()
    console.print("[cyan]Manual Event Input[/cyan]")
    console.print()

    title = Prompt.ask("[bold cyan]Event Title[/bold cyan]")
    description = Prompt.ask("[bold cyan]Event Description[/bold cyan]", default=title)

    console.print()
    console.print(
        "[dim]Format: YYYY-MM-DD or YYYY-MM-DD HH:MM (e.g., 2025-12-31 or 2025-12-31 23:59)[/dim]"
    )
    cutoff_str = Prompt.ask("[bold cyan]Cutoff Date & Time[/bold cyan]")

    try:
        try:
            cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d %H:%M")
        except ValueError:
            cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
            cutoff_dt = cutoff_dt.replace(hour=23, minute=59, second=59)

        cutoff_iso = cutoff_dt.isoformat() + "Z"
    except ValueError:
        console.print("[yellow]⚠ Invalid date format, using a far future date[/yellow]")
        cutoff_iso = "2030-12-31T23:59:59Z"

    return {
        "event_id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "market_type": "llm",
        "cutoff": cutoff_iso,
        "metadata": {},
    }


def select_events() -> list[dict]:
    mode = prompt_event_selection_mode()
    if mode == "1":
        source = prompt_event_source()

        if source == "1":
            event = select_single_event_from_live()
            if event:
                return [event]
            return []
        else:
            event = prompt_manual_event()
            return [event]

    else:
        source = prompt_event_source()

        if source == "1":
            return select_multiple_events_from_live()
        else:
            console.print()
            num = Prompt.ask("[bold cyan]How many events to input?[/bold cyan]", default="1")
            try:
                num = int(num)
                num = max(1, min(10, num))
            except ValueError:
                num = 1

            events = []
            for i in range(num):
                console.print(f"\n[cyan]Event {i + 1} of {num}:[/cyan]")
                event = prompt_manual_event()
                events.append(event)

            return events
