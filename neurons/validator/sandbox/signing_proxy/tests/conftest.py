from unittest.mock import MagicMock

import pytest
from bittensor_wallet import Wallet


@pytest.fixture
def mock_wallet() -> MagicMock:
    mock_wallet = MagicMock(spec=Wallet)
    mock_hotkey = MagicMock()
    mock_hotkey.ss58_address = "5CCgXySACBvSJ9mz76FwhksstiGSaNuNr5fMqCYJ8efioFaE"
    mock_hotkey.public_key.hex.return_value = "0x1234567890abcdef"
    mock_hotkey.sign.return_value = b"mock_signature"
    mock_wallet.hotkey = mock_hotkey
    return mock_wallet
