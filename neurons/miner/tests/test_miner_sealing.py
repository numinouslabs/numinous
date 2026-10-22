from datetime import datetime, timedelta, timezone

import pytest
from bittensor import timelock
from bittensor_wallet import Keypair

from neurons.miner.sealing import receipt_message, seal, sign_forecast

MINER = Keypair.create_from_uri("//Alice")
QUESTION_ID = "1d3ffd76-b934-4741-931c-582e53a47751"


class TestSeal:
    def test_the_seal_carries_its_round_and_cannot_be_opened_early(self):
        reveal_round = timelock.current_round() + 1000

        sealed = seal('{"version": 2}', reveal_round)

        assert timelock.Timelocked.parse(sealed).reveal_round == reveal_round
        with pytest.raises(timelock.TimelockNotReady):
            timelock.decrypt(sealed)

    def test_a_past_round_is_refused(self):
        with pytest.raises(timelock.TimelockError):
            seal('{"version": 2}', timelock.current_round() - 1)


class TestSignForecast:
    def test_verifies_over_question_id_and_sealed_bytes(self):
        sealed = seal('{"version": 2}', timelock.current_round() + 1000)

        signature = sign_forecast(MINER, QUESTION_ID, sealed)

        assert Keypair(ss58_address=MINER.ss58_address).verify(
            QUESTION_ID.encode() + sealed, signature
        )
        assert not Keypair(ss58_address=MINER.ss58_address).verify(
            "another-question".encode() + sealed, signature
        )


class TestReceiptMessage:
    def test_same_bytes_as_the_backend_signs(self):
        received_at = datetime(2026, 10, 13, 17, 34, 22, 516434, tzinfo=timezone.utc)

        assert receipt_message(MINER.ss58_address, QUESTION_ID, "abc", received_at) == (
            f"{MINER.ss58_address}:{QUESTION_ID}:abc:2026-10-13T17:34:22.516434+00:00".encode()
        )

    def test_a_moment_in_another_timezone_keeps_its_offset(self):
        received_at = datetime(2026, 10, 13, 14, 34, 22, tzinfo=timezone(timedelta(hours=-3)))

        assert receipt_message("miner", "question", "abc", received_at) == (
            b"miner:question:abc:2026-10-13T14:34:22-03:00"
        )
