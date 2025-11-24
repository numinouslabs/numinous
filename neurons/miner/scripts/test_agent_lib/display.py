import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def show_results(results: dict) -> None:
    if not results or results.get("status") == "error":
        console.print(f"[red]✗ Error: {results.get('error', 'Unknown error')}[/red]")
        return

    total = results.get("total", 0)
    success = results.get("success", 0)
    failed = results.get("failed", 0)
    duration = results.get("duration", 0)
    agent = results.get("agent", "Unknown")

    success_rate = (success / total * 100) if total > 0 else 0

    summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="white")

    summary.add_row("🤖 Agent", agent)
    summary.add_row("📊 Total Tests", str(total))
    summary.add_row("✅ Passed", f"[green]{success}[/green] ({success_rate:.1f}%)")
    summary.add_row("❌ Failed", f"[red]{failed}[/red]")
    summary.add_row("⏱️  Duration", f"{duration:.2f}s")

    if total > 0:
        avg_time = duration / total
        summary.add_row("📈 Avg Time", f"{avg_time:.2f}s per test")

    console.print()
    console.print(Panel.fit(summary, title="📋 Test Summary", border_style="cyan"))
    console.print()

    tests = results.get("tests", [])
    if not tests:
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="📊 Detailed Results",
        title_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4, no_wrap=True)
    table.add_column("Event", style="white", max_width=40, no_wrap=False)
    table.add_column("Status", width=10, no_wrap=True)
    table.add_column("Prediction", width=12, justify="right", no_wrap=True)
    table.add_column("Reasoning", style="dim", max_width=60, no_wrap=False)
    table.add_column("Time", width=8, justify="right", no_wrap=True)

    for idx, test in enumerate(tests, 1):
        status = test.get("status", "unknown")
        event_title = test.get("event_title", "Unknown")

        if status == "success":
            status_str = "[green]✓ PASS[/green]"
        else:
            status_str = "[red]✗ FAIL[/red]"

        prediction = test.get("prediction")
        if prediction is not None:
            prediction_str = f"[yellow]{prediction:.4f}[/yellow]"
        else:
            prediction_str = "[dim]N/A[/dim]"

        reasoning = test.get("reasoning", "")
        reasoning_display = reasoning if reasoning else "[dim]N/A[/dim]"

        duration_val = test.get("duration", 0)
        duration_str = f"{duration_val:.2f}s"

        table.add_row(
            str(idx), event_title, status_str, prediction_str, reasoning_display, duration_str
        )

    console.print(table)
    console.print()

    errors = [t for t in tests if t.get("status") != "success"]
    if errors:
        console.print("[yellow]⚠️  Failed Tests Details:[/yellow]")
        console.print()

        for idx, test in enumerate(errors, 1):
            error_msg = test.get("error", "Unknown error")
            event_title = test.get("event_title", "Unknown")

            console.print(f"[red]Test {idx}:[/red] {event_title}")
            console.print(f"  [dim]Error:[/dim] {error_msg}")

            traceback = test.get("traceback")
            if traceback:
                console.print("  [dim]Traceback:[/dim]")
                traceback_lines = traceback.split("\n")
                for line in traceback_lines[-10:]:
                    if line.strip():
                        console.print(f"    [dim]{line}[/dim]")

            console.print()

    if success == total:
        console.print("[bold green]🎉 All tests passed![/bold green]")
    elif success > 0:
        console.print(f"[yellow]⚠️  {failed} test(s) failed. Review errors above.[/yellow]")
    else:
        console.print("[red]❌ All tests failed. Check your agent code and errors above.[/red]")

    console.print()

    show_logs_interactive(tests)
    save_results_to_file(results)


def show_logs_interactive(tests: list) -> None:
    if not tests:
        return

    view_logs = Confirm.ask("[cyan]📋 View agent logs?[/cyan]", default=False)
    if not view_logs:
        console.print()
        return

    console.print()

    # If only one test, just show it - no need to ask which one
    if len(tests) == 1:
        indices_to_show = [1]
    else:
        console.print("[dim]Enter test numbers (e.g., '1,3,5' or 'all'):[/dim]")
        selection = Prompt.ask("[cyan]Tests to view[/cyan]", default="all")

        if selection.lower().strip() == "all":
            indices_to_show = list(range(1, len(tests) + 1))
        else:
            try:
                indices_to_show = [int(x.strip()) for x in selection.split(",")]
                indices_to_show = [i for i in indices_to_show if 1 <= i <= len(tests)]
            except ValueError:
                console.print("[red]Invalid input. Showing all logs.[/red]")
                indices_to_show = list(range(1, len(tests) + 1))

        if not indices_to_show:
            console.print("[yellow]No valid test numbers specified.[/yellow]")
            console.print()
            return

    console.print()
    console.print("[cyan]📋 Agent Logs:[/cyan]")
    console.print()

    for idx in indices_to_show:
        test = tests[idx - 1]
        event_title = test.get("event_title", "Unknown")[:60]
        logs = test.get("logs", "").strip()

        if logs:
            console.print(f"[bold cyan]Test {idx}:[/bold cyan] {event_title}")
            console.print(Panel(logs, border_style="dim", padding=(0, 1)))
            console.print()
        else:
            console.print(f"[bold cyan]Test {idx}:[/bold cyan] {event_title}")
            console.print("  [dim]No logs captured[/dim]")
            console.print()

    console.print()


def save_results_to_file(results: dict) -> None:
    save = Confirm.ask("[cyan]💾 Save results to file?[/cyan]", default=True)

    if not save:
        console.print()
        return

    console.print()

    miner_dir = Path(__file__).parent.parent.parent
    base_results_dir = miner_dir / "test-results"
    base_results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = results.get("agent", "agent").replace(".py", "")
    run_dir_name = f"{timestamp}_{agent_name}"
    run_dir = base_results_dir / run_dir_name
    run_dir.mkdir(exist_ok=True)

    json_path = run_dir / "results.json"
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]✓[/green] JSON saved: [cyan]{json_path}[/cyan]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save JSON: {e}[/red]")

    txt_path = run_dir / "report.txt"
    try:
        with open(txt_path, "w") as f:
            write_text_report(f, results)
        console.print(f"[green]✓[/green] Report saved: [cyan]{txt_path}[/cyan]")
    except Exception as e:
        console.print(f"[red]✗ Failed to save report: {e}[/red]")

    console.print()


def write_text_report(file: TextIO, results: dict) -> None:
    agent = results.get("agent", "Unknown")
    total = results.get("total", 0)
    success = results.get("success", 0)
    failed = results.get("failed", 0)
    duration = results.get("duration", 0)
    tests = results.get("tests", [])

    file.write("=" * 80 + "\n")
    file.write("AGENT TEST RESULTS\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    file.write("SUMMARY\n")
    file.write("-" * 80 + "\n")
    file.write(f"Agent:        {agent}\n")
    file.write(f"Total Tests:  {total}\n")
    file.write(f"Passed:       {success} ({success/total*100:.1f}%)\n")
    file.write(f"Failed:       {failed}\n")
    file.write(f"Duration:     {duration:.2f}s\n")
    file.write(f"Avg Time:     {duration/total:.2f}s per test\n")
    file.write("\n")

    file.write("DETAILED RESULTS\n")
    file.write("-" * 80 + "\n\n")

    for idx, test in enumerate(tests, 1):
        event_title = test.get("event_title", "Unknown")
        status = test.get("status", "unknown")
        test_duration = test.get("duration", 0)

        file.write(f"Test {idx}: {event_title}\n")
        file.write(f"  Status:     {'PASS' if status == 'success' else 'FAIL'}\n")
        file.write(f"  Duration:   {test_duration:.2f}s\n")

        if status == "success":
            prediction = test.get("prediction")
            reasoning = test.get("reasoning", "")
            file.write(f"  Prediction: {prediction:.4f}\n")
            if reasoning:
                file.write("  Reasoning:\n")
                wrapped = textwrap.fill(
                    reasoning, width=74, initial_indent="    ", subsequent_indent="    "
                )
                file.write(wrapped + "\n")
        else:
            error = test.get("error", "Unknown error")
            file.write(f"  Error:      {error}\n")

        file.write("\n")

    file.write("AGENT LOGS\n")
    file.write("-" * 80 + "\n\n")

    for idx, test in enumerate(tests, 1):
        event_title = test.get("event_title", "Unknown")[:60]
        logs = test.get("logs", "").strip()

        file.write(f"Test {idx}: {event_title}\n")
        file.write("-" * 40 + "\n")
        if logs:
            file.write(logs + "\n")
        else:
            file.write("No logs captured\n")
        file.write("\n")

    file.write("=" * 80 + "\n")
