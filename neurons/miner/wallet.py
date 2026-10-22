from pathlib import Path

from bittensor_wallet import Keypair
from bittensor_wallet.keyfile import Keyfile

DEFAULT_WALLET_PATH = Path.home() / ".bittensor" / "wallets"


class HotkeyNotFound(Exception):
    pass


def load_hotkey(wallet_name: str, hotkey_name: str, wallet_path: Path) -> Keypair:
    hotkeys = wallet_path.expanduser() / wallet_name / "hotkeys"
    candidates = [hotkeys / f"{hotkey_name}.json", hotkeys / hotkey_name]
    hotkey_file = next((candidate for candidate in candidates if candidate.is_file()), None)
    if hotkey_file is None:
        raise HotkeyNotFound(f"no hotkey file at {hotkeys / hotkey_name}(.json)")

    return Keyfile(path=str(hotkey_file)).keypair
