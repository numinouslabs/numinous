from pathlib import Path

import pytest
from bittensor import Wallet

from neurons.miner.wallet import HotkeyNotFound, load_hotkey

SEED = "0x" + "02" * 32


def make_hotkey_file(wallet_path: Path, hotkey_name: str) -> str:
    wallet = Wallet(name="miner", hotkey=hotkey_name, path=str(wallet_path))
    wallet.regenerate_hotkey(seed=SEED, use_password=False, overwrite=True, suppress=True)
    return wallet.hotkey.ss58_address


class TestLoadHotkey:
    def test_loads_a_hotkey_file(self, tmp_path: Path):
        address = make_hotkey_file(tmp_path, "default")

        assert load_hotkey("miner", "default", tmp_path).ss58_address == address

    def test_loads_a_hotkey_file_saved_with_a_json_suffix(self, tmp_path: Path):
        address = make_hotkey_file(tmp_path, "default")
        hotkeys = tmp_path / "miner" / "hotkeys"
        (hotkeys / "default").rename(hotkeys / "default.json")

        assert load_hotkey("miner", "default", tmp_path).ss58_address == address

    def test_a_missing_hotkey_says_where_it_looked(self, tmp_path: Path):
        make_hotkey_file(tmp_path, "default")

        with pytest.raises(HotkeyNotFound, match="miner/hotkeys/other"):
            load_hotkey("miner", "other", tmp_path)

    def test_a_missing_wallet(self, tmp_path: Path):
        with pytest.raises(HotkeyNotFound):
            load_hotkey("no_such_wallet", "default", tmp_path)
