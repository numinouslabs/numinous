import json
import typing
from getpass import getpass
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def list_available_wallets(
    wallet_path: typing.Optional[Path] = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    if wallet_path is None:
        wallets_dir = Path.home() / ".bittensor" / "wallets"
    else:
        wallets_dir = wallet_path

    if not wallets_dir.exists():
        return []

    wallets = []
    for wallet_path_item in wallets_dir.iterdir():
        if wallet_path_item.is_dir():
            hotkeys_dir = wallet_path_item / "hotkeys"
            if hotkeys_dir.exists():
                hotkeys = []
                for h in hotkeys_dir.iterdir():
                    if h.is_file() and not h.name.endswith("pub.txt"):
                        hotkey_name = h.stem if h.suffix == ".json" else h.name
                        # Load keypair to get the address
                        keypair = load_keypair(wallet_path_item.name, hotkey_name, wallet_path)
                        address = keypair.ss58_address if keypair else "Unknown"
                        hotkeys.append((hotkey_name, address))
                if hotkeys:
                    # Sort by hotkey name
                    hotkeys.sort(key=lambda x: x[0])
                    wallets.append((wallet_path_item.name, hotkeys))

    return wallets


def load_keypair(wallet_name: str, hotkey_name: str, wallet_path: typing.Optional[Path] = None):
    from bittensor import Keypair

    if wallet_path is None:
        wallet_dir = Path.home() / ".bittensor" / "wallets" / wallet_name
    else:
        wallet_dir = wallet_path / wallet_name

    hotkeys_dir = wallet_dir / "hotkeys"
    hotkey_paths = [hotkeys_dir / f"{hotkey_name}.json", hotkeys_dir / hotkey_name]
    for hotkey_path in hotkey_paths:
        if hotkey_path.exists():
            try:
                with open(hotkey_path) as f:
                    data = json.load(f)
                mnemonic = data.get("mnemonic") or data.get("secretPhrase")
                if not mnemonic:
                    raise ValueError("No mnemonic or secretPhrase found in hotkey file")
                keypair = Keypair.create_from_mnemonic(mnemonic)
                return keypair
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load wallet from {hotkey_path}: {e}")
                continue

    return None


def load_coldkey(wallet_name: str, wallet_path: typing.Optional[Path] = None):
    from bittensor_wallet import Wallet

    if wallet_path is None:
        wallet_dir = Path.home() / ".bittensor" / "wallets"
    else:
        wallet_dir = wallet_path

    wallet = Wallet(name=wallet_name, path=str(wallet_dir))
    console.print()
    password = getpass("Enter coldkey password: ")
    console.print()

    try:
        keypair = wallet.get_coldkey(password)
        return keypair
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load coldkey: {e}")
        return None


def prompt_wallet_selection(wallet_path: typing.Optional[Path] = None) -> tuple[str, str]:
    available_wallets = list_available_wallets(wallet_path)
    if not available_wallets:
        console.print()
        console.print(
            Panel.fit(
                "[red]✗ No wallets found in ~/.bittensor/wallets/[/red]\n"
                "[yellow]💡 Tip:[/yellow] Create a wallet first: [cyan]btcli wallet create[/cyan]",
                border_style="red",
            )
        )
        console.print()
        raise click.Abort()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="📋 Available Wallets",
        title_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Wallet", style="green")
    table.add_column("Hotkey", style="yellow")
    table.add_column("Address", style="blue")

    wallet_options = []
    for wallet, hotkeys in available_wallets:
        for hotkey_name, hotkey_address in hotkeys:
            wallet_options.append((wallet, hotkey_name))
            display_address = hotkey_address
            if len(hotkey_address) > 50:
                display_address = hotkey_address[:8] + "..." + hotkey_address[-8:]
            table.add_row(str(len(wallet_options)), wallet, hotkey_name, display_address)

    console.print()
    console.print(table)
    console.print()

    while True:
        choice = Prompt.ask("[bold cyan]Select wallet[/bold cyan]", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(wallet_options):
                return wallet_options[idx]
            else:
                console.print(f"[red]✗[/red] Invalid choice. Please enter 1-{len(wallet_options)}")
        except ValueError:
            console.print("[red]✗[/red] Please enter a number")
