import asyncio
import time
import uuid
from pathlib import Path

from bittensor_wallet import Wallet
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Prompt

from neurons.validator.sandbox import SandboxManager
from neurons.validator.utils.logger.logger import NuminousLogger

console = Console()


def get_gateway_url() -> str:
    return "http://host.docker.internal:8000"


GATEWAY_URL = get_gateway_url()


def prompt_wallet_selection() -> tuple[str, str]:
    from neurons.miner.scripts.wallet_utils import list_available_wallets
    from neurons.miner.scripts.wallet_utils import prompt_wallet_selection as wallet_prompt

    available_wallets = list_available_wallets()
    if not available_wallets:
        console.print()
        console.print("[yellow]⚠ No wallets found[/yellow]")
        console.print("[dim]Using default wallet: validator/default[/dim]")
        console.print()
        return "validator", "default"

    console.print()
    console.print("[cyan]Select wallet for sandbox initialization[/cyan]")
    console.print("[dim](This wallet is only used for the signing proxy, not for uploading)[/dim]")

    return wallet_prompt()


def run_tests(agent_path: Path, events: list[dict]) -> dict:
    return asyncio.run(run_tests_async(agent_path, events))


async def run_tests_async(agent_path: Path, events: list[dict]) -> dict:
    console.print()
    console.print("[cyan]🚀 Initializing sandbox environment...[/cyan]")
    console.print()

    wallet_name, hotkey_name = prompt_wallet_selection()

    try:
        wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
        console.print(
            f"[green]✓[/green] Loaded wallet: [yellow]{wallet_name}/{hotkey_name}[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to load wallet: {e}[/red]")
        console.print("[dim]Using default wallet configuration[/dim]")
        wallet = Wallet(name="validator", hotkey="default")
    console.print()

    logger = NuminousLogger(name="test-agent", level="WARNING")

    console.print("[cyan]Building Docker images (if needed)...[/cyan]")
    try:
        sandbox_manager = SandboxManager(
            bt_wallet=wallet,
            gateway_url=GATEWAY_URL,
            logger=logger,
            log_docker_to_stdout=False,
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize sandbox: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return {"status": "error", "error": str(e)}

    console.print("[green]✓[/green] Sandbox environment ready")
    console.print()

    try:
        agent_code = agent_path.read_text()
    except Exception as e:
        console.print(f"[red]✗ Failed to load agent code: {e}[/red]")
        return {"status": "error", "error": str(e)}

    console.print("[cyan]Test Configuration:[/cyan]")
    console.print(f"  • Agent: [yellow]{agent_path.name}[/yellow]")
    console.print(f"  • Events: [yellow]{len(events)}[/yellow]")
    console.print()

    max_concurrent = 1
    if len(events) > 1:
        default_concurrent_value = min(len(events), 10)
        max_concurrent_str = Prompt.ask(
            "[bold cyan]Max concurrent sandboxes[/bold cyan]",
            default=str(default_concurrent_value),
        )
        try:
            max_concurrent = int(max_concurrent_str)
            max_concurrent = max(1, min(20, max_concurrent))
        except ValueError:
            max_concurrent = default_concurrent_value

    console.print(f"[dim]Running with max {max_concurrent} concurrent sandbox(es)[/dim]")
    console.print()

    results = await run_all_tests(
        sandbox_manager=sandbox_manager,
        agent_code=agent_code,
        agent_name=agent_path.name,
        events=events,
        max_concurrent=max_concurrent,
    )

    console.print()
    console.print("[cyan]Cleaning up sandbox environment...[/cyan]")
    sandbox_manager.close()
    console.print("[green]✓[/green] Cleanup complete")
    console.print()

    return results


async def run_all_tests(
    sandbox_manager: SandboxManager,
    agent_code: str,
    agent_name: str,
    events: list[dict],
    max_concurrent: int,
) -> dict:
    results = {
        "agent": agent_name,
        "total": len(events),
        "success": 0,
        "failed": 0,
        "tests": [],
        "start_time": time.time(),
    }

    semaphore = asyncio.Semaphore(max_concurrent)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running tests[/cyan]", total=len(events))
        tasks = []
        for event in events:
            test_task = run_single_test(
                semaphore=semaphore,
                sandbox_manager=sandbox_manager,
                agent_code=agent_code,
                event=event,
                progress=progress,
                progress_task=task,
            )
            tasks.append(test_task)

        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        for test_result in test_results:
            if isinstance(test_result, Exception):
                results["failed"] += 1
                results["tests"].append(
                    {
                        "status": "error",
                        "error": str(test_result),
                    }
                )
            elif test_result["status"] == "success":
                results["success"] += 1
                results["tests"].append(test_result)
            else:
                results["failed"] += 1
                results["tests"].append(test_result)

    results["end_time"] = time.time()
    results["duration"] = results["end_time"] - results["start_time"]

    return results


async def run_single_test(
    semaphore: asyncio.Semaphore,
    sandbox_manager: SandboxManager,
    agent_code: str,
    event: dict,
    progress: Progress,
    progress_task,
) -> dict:
    async with semaphore:
        run_id = str(uuid.uuid4())
        start_time = time.time()

        event_data = {
            "event_id": event["event_id"],
            "market_type": event["market_type"],
            "event_type": event["market_type"],
            "title": event["title"],
            "description": event["description"],
            "cutoff": event["cutoff"],
            "metadata": event.get("metadata", {}),
        }

        loop = asyncio.get_event_loop()
        result_future = asyncio.Future()

        def on_finish(result):
            if not result_future.done():
                loop.call_soon_threadsafe(result_future.set_result, result)

        try:
            sandbox_manager.create_sandbox(
                agent_code=agent_code,
                event_data=event_data,
                run_id=run_id,
                on_finish=on_finish,
                timeout=300,
            )
        except Exception as e:
            result = {
                "status": "error",
                "error": f"Sandbox creation failed: {str(e)}",
                "traceback": "",
            }
            if not result_future.done():
                result_future.set_result(result)

        try:
            result = await asyncio.wait_for(result_future, timeout=180)
        except asyncio.TimeoutError:
            result = {
                "status": "error",
                "error": "Sandbox execution timeout (125s)",
                "traceback": "",
            }
        except Exception as e:
            result = {
                "status": "error",
                "error": f"Execution error: {str(e)}",
                "traceback": "",
            }

        end_time = time.time()
        duration = end_time - start_time
        progress.update(progress_task, advance=1)

        test_result = {
            "event_id": event["event_id"],
            "event_title": event["title"],
            "status": result.get("status", "error"),
            "duration": duration,
            "run_id": run_id,
            "logs": result.get("logs", ""),
        }

        if result.get("status") == "success":
            output = result.get("output", {})
            test_result["prediction"] = output.get("prediction")
            test_result["reasoning"] = output.get("reasoning", "")
        else:
            test_result["error"] = result.get("error", "Unknown error")
            test_result["traceback"] = result.get("traceback", "")

        return test_result
