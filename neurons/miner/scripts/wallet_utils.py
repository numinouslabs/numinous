import typing
from getpass import getpass
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

if typing.TYPE_CHECKING:
    from bittensor_wallet import Keypair

console = Console()

ENCRYPTED_ADDRESS_PLACEHOLDER = "Encrypted"
UNREADABLE_ADDRESS_PLACEHOLDER = "Unknown"


def _resolve_hotkey_path(
    wallet_name: str, hotkey_name: str, wallet_path: typing.Optional[Path]
) -> typing.Optional[Path]:
    if wallet_path is None:
        wallet_dir = Path.home() / ".bittensor" / "wallets" / wallet_name
    else:
        wallet_dir = wallet_path / wallet_name

    hotkeys_dir = wallet_dir / "hotkeys"

    for candidate in (hotkeys_dir / f"{hotkey_name}.json", hotkeys_dir / hotkey_name):
        if candidate.exists():
            return candidate

    return None


def _read_hotkey_address(hotkey_path: Path) -> str:
    from bittensor_wallet.keyfile import (
        deserialize_keypair_from_keyfile_data,
        keyfile_data_is_encrypted,
    )

    try:
        keyfile_data = hotkey_path.read_bytes()

        if keyfile_data_is_encrypted(keyfile_data):
            return ENCRYPTED_ADDRESS_PLACEHOLDER

        return deserialize_keypair_from_keyfile_data(keyfile_data).ss58_address
    except Exception:
        return UNREADABLE_ADDRESS_PLACEHOLDER


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
                for hotkey_file in hotkeys_dir.iterdir():
                    if hotkey_file.is_file() and not hotkey_file.name.endswith("pub.txt"):
                        hotkey_name = (
                            hotkey_file.stem if hotkey_file.suffix == ".json" else hotkey_file.name
                        )
                        hotkeys.append((hotkey_name, _read_hotkey_address(hotkey_file)))
                if hotkeys:
                    # Sort by hotkey name
                    hotkeys.sort(key=lambda x: x[0])
                    wallets.append((wallet_path_item.name, hotkeys))

    return wallets


def load_keypair(
    wallet_name: str, hotkey_name: str, wallet_path: typing.Optional[Path] = None
) -> typing.Optional["Keypair"]:
    from bittensor_wallet.keyfile import Keyfile

    hotkey_path = _resolve_hotkey_path(wallet_name, hotkey_name, wallet_path)

    if hotkey_path is None:
        return None

    try:
        return Keyfile(path=str(hotkey_path)).keypair
    except Exception as error:
        console.print(f"[red]✗[/red] Failed to load wallet from {hotkey_path}: {error}")

        return None


def load_coldkey(wallet_name: str, wallet_path: typing.Optional[Path] = None):
    from bittensor_wallet import Wallet

    if wallet_path is None:
        wallet_dir = Path.home() / ".bittensor" / "wallets"
    else:
        wallet_dir = wallet_path

    wallet = Wallet(name=wallet_name, path=str(wallet_dir))
    console.print()
    password = getpass("Enter wallet password: ")
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
