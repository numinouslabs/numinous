from datetime import datetime

from bittensor import timelock
from bittensor_wallet import Keypair

SEALED_FORECAST_MAX_BYTES = 40_960


def seal(forecast: str, reveal_round: int) -> bytes:
    return bytes(timelock.encrypt(forecast, reveal_round=reveal_round))


def sign_forecast(hotkey: Keypair, question_id: str, sealed: bytes) -> bytes:
    return hotkey.sign(question_id.encode() + sealed)


def receipt_message(
    miner_hotkey: str, question_id: str, sealed_sha256: str, received_at: datetime
) -> bytes:
    return f"{miner_hotkey}:{question_id}:{sealed_sha256}:{received_at.isoformat()}".encode()
