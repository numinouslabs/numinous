import subprocess
import time
from pathlib import Path

import httpx
from rich.console import Console

console = Console()

GATEWAY_URL = "http://localhost:8000"
GATEWAY_LOG_FILE = Path("gateway.log")


def check_gateway_health() -> bool:
    try:
        response = httpx.get(f"{GATEWAY_URL}/api/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def get_gateway_pid() -> int | None:
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if (
                    cmdline
                    and "uvicorn" in " ".join(cmdline)
                    and "neurons.miner.gateway.app" in " ".join(cmdline)
                ):
                    return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    except ImportError:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "neurons.miner.gateway.app"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split()[0])
        except Exception:
            pass
        return None


def stop_gateway() -> bool:
    pid = get_gateway_pid()
    if not pid:
        return False

    try:
        import psutil

        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True
    except ImportError:
        try:
            subprocess.run(["kill", str(pid)], check=True)
            return True
        except Exception:
            return False
    except Exception:
        return False


def start_gateway() -> tuple[bool, int | None, Path | None]:
    try:
        log_handle = open(GATEWAY_LOG_FILE, "a")

        process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "neurons.miner.gateway.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        console.print("  [cyan]Starting gateway...[/cyan]", end="")
        for i in range(10):
            time.sleep(0.5)
            if check_gateway_health():
                console.print(" [green]✓[/green]")
                return True, process.pid, GATEWAY_LOG_FILE
            console.print(".", end="")

        console.print(" [red]✗[/red]")
        log_handle.close()
        return False, None, None

    except Exception as e:
        console.print(f" [red]✗[/red] Error: {e}")
        return False, None, None


def show_gateway_status() -> None:
    console.print()
    console.print("[cyan]🌐 Gateway Status[/cyan]")
    console.print()

    is_healthy = check_gateway_health()
    pid = get_gateway_pid()

    if is_healthy and pid:
        console.print("  [green]✓[/green] Running")
        console.print(f"  [dim]URL:[/dim] {GATEWAY_URL}")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print(f"  [dim]Logs:[/dim] {GATEWAY_LOG_FILE.absolute()}")
        console.print()
        console.print("  [yellow]📋 View logs:[/yellow] [cyan]numi gateway logs[/cyan]")
        console.print("  [yellow]🛑 Stop:[/yellow] [cyan]numi gateway stop[/cyan]")
    elif pid:
        console.print("  [yellow]⚠[/yellow] Process running but not responding")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print()
        console.print("  [yellow]🛑 Stop:[/yellow] [cyan]numi gateway stop[/cyan]")
    else:
        console.print("  [red]✗[/red] Not running")
        console.print()
        console.print("  [yellow]🚀 Start:[/yellow] [cyan]numi gateway start[/cyan]")

    console.print()


def tail_logs(follow: bool = True) -> None:
    if not GATEWAY_LOG_FILE.exists():
        console.print()
        console.print(f"[yellow]⚠ Log file not found: {GATEWAY_LOG_FILE}[/yellow]")
        console.print()
        return

    try:
        if follow:
            subprocess.run(["tail", "-f", str(GATEWAY_LOG_FILE)])
        else:
            subprocess.run(["tail", "-n", "50", str(GATEWAY_LOG_FILE)])
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Log viewing stopped[/dim]")
        console.print()
    except Exception as e:
        console.print()
        console.print(f"[red]✗ Error viewing logs: {e}[/red]")
        console.print()
