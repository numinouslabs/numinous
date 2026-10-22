import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest
from aiohttp import ClientResponseError
from aioresponses import aioresponses
from bittensor import Wallet, http_auth

from neurons.validator.models.dkg_client import (
    ClosedRoundsPage,
    MinerScore,
    ResultsPage,
    RoundHashesPage,
    ScoringRun,
    StoredScores,
)
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __version__

STAGING_URL = "https://stg.numinous.earth"
STAGING_HOTKEY = "5EgzbPFTqhpKwZQSVDe4PfDhKArvWMNEWXPJoiVTqKe1FW45"
PROD_URL = "https://numinous.earth"
PROD_HOTKEY = "5CyqMG9eDSHbNgsRW49e4SRSAGSaEWoiVVWv6fD7eWRPCuPB"
ROUND_ID = "31554aee-295f-4524-a1fa-86de56327742"
QUESTION_ID = "1d3ffd76-b934-4741-931c-582e53a47751"
MINER = "5EcEdnhQ7cJBQSfcb4T8YkGVEBic3Giks9fmiUhmYa6YZdCo"
SINCE = datetime(2026, 10, 10, 9, 30, 15, 123456, tzinfo=timezone(timedelta(hours=-3)))
CLOSED_ROUNDS_PATH = "/api/v3/dkg/rounds?closed_since=2026-10-10T12:30:15Z&offset=0"
HASHES_PATH = f"/api/v3/dkg/rounds/{ROUND_ID}/hashes?offset=0"
RESULTS_PATH = "/api/v3/dkg/results?since=2026-10-10T12:30:15Z&offset=0"
SCORES_PATH = "/api/v3/dkg/scores"
OTHER_MINER = "5D4gFKHHM9eBhre5oCs29KEiW5o6c9witqVUSyMzpw1gri1N"

CLOSED_ROUNDS = {
    "rounds": [
        {
            "id": ROUND_ID,
            "deadline_at": "2026-10-13T18:00:00Z",
            "horizon_end_at": "2026-11-12T18:00:00Z",
            "reveal_round": 32713961,
            "questions": [
                {
                    "id": QUESTION_ID,
                    "entity_id": "02f6f374-0544-54a9-82c5-810bb9ff9352",
                    "entity": "Apple",
                    "counterparty_id": "65ea72f0-9e81-56e9-b0f6-b66266a14ca2",
                    "counterparty": "TSMC",
                    "relationship": {"head": "TSMC", "relation": "SUPPLIES", "tail": "Apple"},
                }
            ],
        }
    ],
    "next_offset": None,
}

ROUND_HASHES = {
    "round_id": ROUND_ID,
    "deadline_at": "2026-10-13T18:00:00Z",
    "horizon_end_at": "2026-11-12T18:00:00Z",
    "forecasts": [
        {
            "miner_uid": 196,
            "miner_hotkey": MINER,
            "question_id": QUESTION_ID,
            "sealed_sha256": "c9d56c84b41dd8d250773e07b6a0dd1259b68f07b7f0328235c01b614a04c099",
            "received_at": "2026-10-13T17:34:22.516434Z",
        }
    ],
    "next_offset": 1000,
}

RESULTS = {
    "forecasts": [
        {
            "miner_uid": 196,
            "miner_hotkey": MINER,
            "question_id": QUESTION_ID,
            "round_id": ROUND_ID,
            "deadline_at": "2026-10-13T18:00:00Z",
            "horizon_end_at": "2026-11-12T18:00:00Z",
            "received_at": "2026-10-13T17:34:22.516434Z",
            "sealed": "abcd",
            "signature": "ef01",
            "opened": {
                "version": 2,
                "statements": [
                    {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.1}
                ],
            },
            "invalid_reason": None,
            "results": [
                {
                    "statement_position": 0,
                    "outcome": True,
                    "reason": "a reported development is the act the forecast states",
                    "event_id": "8f0c2a4e-2b1d-4c3e-9a7f-1e2d3c4b5a69",
                    "reported_at": "2026-10-20T09:00:00Z",
                    "judged_at": "2026-10-21T07:30:00Z",
                }
            ],
        },
        {
            "miner_uid": 196,
            "miner_hotkey": MINER,
            "question_id": "051ef6d8-f5ca-4048-b2c9-73fae35c65e0",
            "round_id": ROUND_ID,
            "deadline_at": "2026-10-13T18:00:00Z",
            "horizon_end_at": "2026-11-12T18:00:00Z",
            "received_at": "2026-10-13T17:34:23.650435Z",
            "sealed": "1234",
            "signature": "5678",
            "opened": None,
            "invalid_reason": "not a valid forecast: relations: Extra inputs are not permitted",
            "results": [],
        },
    ],
    "next_offset": None,
}

SCORING_RUN = ScoringRun(
    scored_at=datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc),
    spec_version=3013,
    scores=[
        MinerScore(miner_uid=196, miner_hotkey=MINER, score=3.7071067811865475),
        MinerScore(miner_uid=207, miner_hotkey=OTHER_MINER, score=0.0),
    ],
)


@pytest.fixture
def wallet(tmp_path: Path) -> Wallet:
    wallet = Wallet(name="test", hotkey="default", path=str(tmp_path))
    wallet.regenerate_hotkey(
        seed="0x" + "01" * 32, use_password=False, overwrite=True, suppress=True
    )
    return wallet


@pytest.fixture
def logger() -> MagicMock:
    return MagicMock(spec=NuminousLogger)


@pytest.fixture
def client(wallet: Wallet, logger: MagicMock) -> DkgClient:
    return DkgClient(env="test", logger=logger, bt_wallet=wallet)


def assert_signed_for(
    wallet: Wallet, receiver_hotkey: str, mocked: aioresponses, path: str
) -> None:
    [calls] = mocked.requests.values()
    [call] = calls
    caller = http_auth.verify(
        call.kwargs["headers"], b"", method="GET", path=path, self_hotkey_ss58=receiver_hotkey
    )
    assert caller.hotkey_ss58 == wallet.hotkey.ss58_address


class TestDkgClient:
    def test_init_rejects_an_unknown_env(self, wallet: Wallet, logger: MagicMock):
        with pytest.raises(ValueError):
            DkgClient(env="staging", logger=logger, bt_wallet=wallet)

    def test_init_rejects_anything_but_a_logger(self, wallet: Wallet):
        with pytest.raises(TypeError):
            DkgClient(env="test", logger="not a logger", bt_wallet=wallet)

    def test_init_rejects_anything_but_a_wallet(self, logger: MagicMock):
        with pytest.raises(TypeError):
            DkgClient(env="test", logger=logger, bt_wallet="not a wallet")

    async def test_get_closed_rounds(self, client: DkgClient, wallet: Wallet):
        with aioresponses() as mocked:
            mocked.get(
                CLOSED_ROUNDS_PATH,
                payload=CLOSED_ROUNDS,
            )

            page = await client.get_closed_rounds(SINCE, 0)

            assert client._DkgClient__backend == (STAGING_URL, STAGING_HOTKEY)
            assert page == ClosedRoundsPage.model_validate(CLOSED_ROUNDS)
            assert page.rounds[0].questions[0].relationship.relation == "SUPPLIES"
            assert_signed_for(wallet, STAGING_HOTKEY, mocked, CLOSED_ROUNDS_PATH)

    async def test_get_round_hashes(self, client: DkgClient, wallet: Wallet):
        with aioresponses() as mocked:
            mocked.get(
                f"/api/v3/dkg/rounds/{ROUND_ID}/hashes?offset=1000",
                payload=ROUND_HASHES,
            )

            page = await client.get_round_hashes(ROUND_ID, 1000)

            assert page == RoundHashesPage.model_validate(ROUND_HASHES)
            assert (page.forecasts[0].miner_uid, page.forecasts[0].miner_hotkey) == (196, MINER)
            assert page.next_offset == 1000
            assert_signed_for(
                wallet, STAGING_HOTKEY, mocked, f"/api/v3/dkg/rounds/{ROUND_ID}/hashes?offset=1000"
            )

    async def test_get_results(self, client: DkgClient, wallet: Wallet):
        with aioresponses() as mocked:
            mocked.get(
                "/api/v3/dkg/results?since=2026-10-10T12:30:15Z&offset=50",
                payload=RESULTS,
            )

            page = await client.get_results(SINCE, 50)

            assert page == ResultsPage.model_validate(RESULTS)
            opened, invalid = page.forecasts
            assert (opened.miner_uid, invalid.miner_uid) == (196, 196)
            assert opened.opened["statements"][0]["probability"] == 0.1
            assert opened.results[0].reported_at == datetime(
                2026, 10, 20, 9, 0, tzinfo=timezone.utc
            )
            assert invalid.opened is None
            assert invalid.results == []
            assert_signed_for(
                wallet,
                STAGING_HOTKEY,
                mocked,
                "/api/v3/dkg/results?since=2026-10-10T12:30:15Z&offset=50",
            )

    async def test_prod_calls_the_prod_backend_signed_for_its_hotkey(
        self, wallet: Wallet, logger: MagicMock
    ):
        client = DkgClient(env="prod", logger=logger, bt_wallet=wallet)

        with aioresponses() as mocked:
            mocked.get(HASHES_PATH, payload=ROUND_HASHES)

            await client.get_round_hashes(ROUND_ID, 0)

            assert client._DkgClient__backend == (PROD_URL, PROD_HOTKEY)
            assert_signed_for(wallet, PROD_HOTKEY, mocked, HASHES_PATH)

    async def test_every_request_signs_a_fresh_nonce(self, client: DkgClient):
        with aioresponses() as mocked:
            mocked.get(HASHES_PATH, payload=ROUND_HASHES, repeat=True)

            await client.get_round_hashes(ROUND_ID, 0)
            await client.get_round_hashes(ROUND_ID, 0)

            [calls] = mocked.requests.values()
            first, second = calls
            assert (
                first.kwargs["headers"]["X-Bittensor-Nonce"]
                != second.kwargs["headers"]["X-Bittensor-Nonce"]
            )

    async def test_a_success_is_logged_at_debug(self, client: DkgClient, logger: MagicMock):
        with aioresponses() as mocked:
            mocked.get(HASHES_PATH, payload=ROUND_HASHES)

            await client.get_round_hashes(ROUND_ID, 0)

        logger.debug.assert_called_once_with(
            "Http request finished",
            extra={
                "path": f"/api/v3/dkg/rounds/{ROUND_ID}/hashes?offset=0",
                "response_status": 200,
                "elapsed_time_ms": ANY,
            },
        )
        logger.error.assert_not_called()

    async def test_an_error_is_logged_and_raised(self, client: DkgClient, logger: MagicMock):
        with aioresponses() as mocked:
            mocked.get(
                RESULTS_PATH,
                status=403,
                body='{"detail":"this hotkey is not a validator of the subnet"}',
            )

            with pytest.raises(ClientResponseError) as error:
                await client.get_results(SINCE, 0)

        assert error.value.status == 403
        logger.error.assert_called_once_with(
            "Http request failed",
            extra={
                "path": "/api/v3/dkg/results?since=2026-10-10T12:30:15Z&offset=0",
                "response_status": 403,
                "elapsed_time_ms": ANY,
                "response_message": '{"detail":"this hotkey is not a validator of the subnet"}',
            },
        )

    async def test_post_scores_sends_the_run_signed_with_its_body(
        self, client: DkgClient, wallet: Wallet
    ):
        with aioresponses() as mocked:
            mocked.post(SCORES_PATH, payload={"stored": 2})

            stored = await client.post_scores(SCORING_RUN)

            assert stored == StoredScores(stored=2)
            [calls] = mocked.requests.values()
            [call] = calls
            assert json.loads(call.kwargs["data"]) == {
                "scored_at": "2026-10-19T15:00:00Z",
                "spec_version": 3013,
                "scores": [
                    {"miner_uid": 196, "miner_hotkey": MINER, "score": 3.7071067811865475},
                    {"miner_uid": 207, "miner_hotkey": OTHER_MINER, "score": 0.0},
                ],
            }
            assert call.kwargs["headers"]["Content-Type"] == "application/json"
            caller = http_auth.verify(
                call.kwargs["headers"],
                call.kwargs["data"],
                method="POST",
                path=SCORES_PATH,
                self_hotkey_ss58=STAGING_HOTKEY,
            )
            assert caller.hotkey_ss58 == wallet.hotkey.ss58_address

    async def test_the_post_signature_covers_the_body(self, client: DkgClient):
        with aioresponses() as mocked:
            mocked.post(SCORES_PATH, payload={"stored": 2})

            await client.post_scores(SCORING_RUN)

            [calls] = mocked.requests.values()
            [call] = calls
            changed_body = call.kwargs["data"].replace(b"3.7071067811865475", b"9.0")
            with pytest.raises(http_auth.BadSignature):
                http_auth.verify(
                    call.kwargs["headers"],
                    changed_body,
                    method="POST",
                    path=SCORES_PATH,
                    self_hotkey_ss58=STAGING_HOTKEY,
                )

    async def test_a_repeated_run_answers_stored_zero(self, client: DkgClient):
        with aioresponses() as mocked:
            mocked.post(SCORES_PATH, payload={"stored": 0})

            assert await client.post_scores(SCORING_RUN) == StoredScores(stored=0)

    async def test_a_rejected_post_is_logged_and_raised(self, client: DkgClient, logger: MagicMock):
        with aioresponses() as mocked:
            mocked.post(SCORES_PATH, status=422, body='{"detail":"a uid appears twice"}')

            with pytest.raises(ClientResponseError) as error:
                await client.post_scores(SCORING_RUN)

        assert error.value.status == 422
        logger.error.assert_called_once_with(
            "Http request failed",
            extra={
                "path": SCORES_PATH,
                "response_status": 422,
                "elapsed_time_ms": ANY,
                "response_message": '{"detail":"a uid appears twice"}',
            },
        )

    async def test_every_request_carries_the_validator_version(self, client: DkgClient):
        async with client._DkgClient__session() as session:
            assert session.headers["Validator-Version"] == __version__
