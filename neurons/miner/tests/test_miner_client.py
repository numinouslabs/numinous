import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest
from bittensor import http_auth, timelock
from bittensor_wallet import Keypair

from neurons.miner.client import (
    BACKENDS,
    AlreadySubmitted,
    BackendRejected,
    DeadlinePassed,
    ForecastNotValid,
    MinerClient,
    NoRoundAccepting,
    NotAQuestionOfTheRound,
    ReceiptInvalid,
    forecast_errors,
)
from neurons.miner.models import ForecastError
from neurons.miner.sealing import seal

HOTKEY = Keypair.create_from_uri("//Alice")

ROUND = {
    "id": "31554aee-295f-4524-a1fa-86de56327742",
    "deadline_at": "2026-10-15T01:00:00Z",
    "horizon_end_at": "2026-10-15T01:00:00Z",
    "reveal_round": 32713961,
    "questions": [
        {
            "id": "1d3ffd76-b934-4741-931c-582e53a47751",
            "entity_id": "02f6f374-0544-54a9-82c5-810bb9ff9352",
            "entity": "Apple",
            "counterparty_id": None,
            "counterparty": None,
            "relationship": None,
            "a_field_added_later": "kept",
        }
    ],
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "statements"],
    "properties": {
        "version": {"const": 2},
        "statements": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["head", "relation", "tail", "probability"],
                "properties": {
                    "head": {"type": "string"},
                    "relation": {"type": "string"},
                    "tail": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


MY_FORECAST = {
    "round_id": ROUND["id"],
    "deadline_at": "2026-10-13T18:00:00Z",
    "horizon_end_at": "2026-11-12T18:00:00Z",
    "question": ROUND["questions"][0],
    "received_at": "2026-10-13T17:34:22.516434Z",
    "sealed_sha256": "c9d56c84b41dd8d250773e07b6a0dd1259b68f07b7f0328235c01b614a04c099",
    "state": "opened",
    "opened": {
        "version": 2,
        "statements": [
            {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.1},
            {"head": "Apple", "relation": "ACQUIRES", "tail": "TSMC", "probability": 0.01},
            {"head": "Apple", "relation": "PARTNERS_WITH", "tail": "TSMC", "probability": 0.4},
        ],
    },
    "invalid_reason": None,
    "statements": [
        {
            "statement_position": 0,
            "status": "right",
            "reason": "a reported development is the act the forecast states",
            "event": {
                "id": "8f0c2a4e-2b1d-4c3e-9a7f-1e2d3c4b5a69",
                "description": "Apple invests in TSMC",
                "reported_at": "2026-10-20T09:00:00Z",
            },
            "judged_at": "2026-10-21T07:30:00Z",
        },
        {
            "statement_position": 1,
            "status": "wrong",
            "reason": "the graph already held this relationship at the deadline",
            "event": None,
            "judged_at": "2026-10-13T18:02:02.381899Z",
        },
        {
            "statement_position": 2,
            "status": "waiting",
            "reason": None,
            "event": None,
            "judged_at": None,
        },
    ],
}
MY_FORECASTS_PATH = "/api/v3/dkg/miner/forecasts"


def response(status: int, url: str, **content) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", url), **content)


class TestForecastErrors:
    def test_a_valid_forecast_has_none(self):
        forecast = {
            "version": 2,
            "statements": [
                {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.15}
            ],
        }

        assert forecast_errors(forecast, SCHEMA) == []

    def test_each_error_points_at_its_field(self):
        forecast = {
            "version": 2,
            "statements": [
                {
                    "head": "Apple",
                    "relation": "ACQUIRES",
                    "tail": "TSMC",
                    "probability": 1.5,
                    "confidence": "high",
                }
            ],
        }

        assert forecast_errors(forecast, SCHEMA) == [
            ForecastError(
                path="statements/0",
                message="Additional properties are not allowed ('confidence' was unexpected)",
            ),
            ForecastError(
                path="statements/0/probability", message="1.5 is greater than the maximum of 1"
            ),
        ]

    def test_an_error_on_the_whole_forecast(self):
        assert forecast_errors({"version": 2}, SCHEMA) == [
            ForecastError(path="(forecast)", message="'statements' is a required property")
        ]


class TestMinerClient:
    @pytest.mark.parametrize("env", ["test", "prod"])
    def test_get_round_is_signed_for_the_backend_of_its_env(self, env: str):
        backend = BACKENDS[env]
        url = f"{backend.url}/api/v3/dkg/questions"

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(200, url, json=ROUND)
        ) as get:
            current = MinerClient(env, HOTKEY).get_round()

        called_url = get.call_args.args[0]
        headers = get.call_args.kwargs["headers"]
        caller = http_auth.verify(
            headers,
            b"",
            method="GET",
            path="/api/v3/dkg/questions",
            self_hotkey_ss58=backend.hotkey,
        )
        assert called_url == url
        assert caller.hotkey_ss58 == HOTKEY.ss58_address
        assert current.reveal_round == 32713961
        assert current.questions[0].entity == "Apple"

    def test_get_round_keeps_fields_it_does_not_know(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/questions"

        with patch("neurons.miner.client.httpx.get", return_value=response(200, url, json=ROUND)):
            current = MinerClient("test", HOTKEY).get_round()

        assert current.model_dump(mode="json")["questions"][0]["a_field_added_later"] == "kept"

    def test_no_round_accepting(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/questions"
        not_found = response(404, url, json={"detail": "no round is accepting forecasts"})

        with patch("neurons.miner.client.httpx.get", return_value=not_found):
            with pytest.raises(NoRoundAccepting, match="no round is accepting forecasts"):
                MinerClient("test", HOTKEY).get_round()

    def test_a_rejection_carries_the_status_and_the_backend_detail(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/forecasts/schema"
        rejected = response(400, url, json={"detail": "Neuron not registered on subnet"})

        with patch("neurons.miner.client.httpx.get", return_value=rejected):
            with pytest.raises(BackendRejected) as error:
                MinerClient("test", HOTKEY).get_schema()

        assert (error.value.status, error.value.detail) == (400, "Neuron not registered on subnet")

    def test_a_rejection_without_json_keeps_the_text(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/forecasts/schema"
        rejected = response(502, url, text="Bad gateway")

        with patch("neurons.miner.client.httpx.get", return_value=rejected):
            with pytest.raises(BackendRejected) as error:
                MinerClient("test", HOTKEY).get_schema()

        assert (error.value.status, error.value.detail) == (502, "Bad gateway")

    def test_get_schema(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/forecasts/schema"

        with patch("neurons.miner.client.httpx.get", return_value=response(200, url, json=SCHEMA)):
            assert MinerClient("test", HOTKEY).get_schema() == SCHEMA


class TestGetMyForecasts:
    def page(self, forecasts: list[dict], next_offset: int | None) -> httpx.Response:
        url = f"{BACKENDS['test'].url}{MY_FORECASTS_PATH}"
        return response(200, url, json={"forecasts": forecasts, "next_offset": next_offset})

    def signed_path(self, call) -> str:
        url = call.args[0]
        path = url.removeprefix(BACKENDS["test"].url)
        caller = http_auth.verify(
            call.kwargs["headers"],
            b"",
            method="GET",
            path=path,
            self_hotkey_ss58=BACKENDS["test"].hotkey,
        )
        assert caller.hotkey_ss58 == HOTKEY.ss58_address
        return path

    def test_without_filters(self):
        with patch(
            "neurons.miner.client.httpx.get", return_value=self.page([MY_FORECAST], None)
        ) as get:
            forecasts = MinerClient("test", HOTKEY).get_my_forecasts(None, None, None)

        assert self.signed_path(get.call_args) == f"{MY_FORECASTS_PATH}?offset=0"
        assert [forecast.state for forecast in forecasts] == ["opened"]
        assert [item.status for item in forecasts[0].statements] == ["right", "wrong", "waiting"]
        assert forecasts[0].statements[0].event.reported_at == datetime(
            2026, 10, 20, 9, 0, tzinfo=timezone.utc
        )

    def test_filters_are_part_of_the_signed_path(self):
        round_id = uuid.UUID(ROUND["id"])
        entity_id = uuid.UUID(ROUND["questions"][0]["entity_id"])
        buenos_aires = timezone(timedelta(hours=-3))
        changed_since = datetime(2026, 10, 13, 15, 2, 0, 500000, tzinfo=buenos_aires)

        with patch("neurons.miner.client.httpx.get", return_value=self.page([], None)) as get:
            MinerClient("test", HOTKEY).get_my_forecasts(round_id, entity_id, changed_since)

        assert self.signed_path(get.call_args) == (
            f"{MY_FORECASTS_PATH}?round_id={round_id}&entity_id={entity_id}"
            "&changed_since=2026-10-13T18:02:00Z&offset=0"
        )

    def test_walks_every_page(self):
        second = MY_FORECAST | {"sealed_sha256": "another"}

        with patch(
            "neurons.miner.client.httpx.get",
            side_effect=[self.page([MY_FORECAST], 50), self.page([second], None)],
        ) as get:
            forecasts = MinerClient("test", HOTKEY).get_my_forecasts(None, None, None)

        assert [self.signed_path(call) for call in get.call_args_list] == [
            f"{MY_FORECASTS_PATH}?offset=0",
            f"{MY_FORECASTS_PATH}?offset=50",
        ]
        assert [forecast.sealed_sha256 for forecast in forecasts] == [
            MY_FORECAST["sealed_sha256"],
            "another",
        ]

    def test_keeps_fields_it_does_not_know(self):
        with patch("neurons.miner.client.httpx.get", return_value=self.page([MY_FORECAST], None)):
            [forecast] = MinerClient("test", HOTKEY).get_my_forecasts(None, None, None)

        assert forecast.model_dump(mode="json")["question"]["a_field_added_later"] == "kept"


QUESTION_ID = ROUND["questions"][0]["id"]
FORECAST = {
    "version": 2,
    "statements": [
        {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.15}
    ],
}
RECEIPT_KEY = Keypair.create_from_seed("0x" + "07" * 32)


class StandInBackend:
    def __init__(self):
        self.reveal_round = timelock.current_round() + 1000
        self.answer = "receipt"
        self.posted: list[tuple[str, bytes, dict]] = []

    def get(self, url: str, headers: dict, timeout: int) -> httpx.Response:
        path = url.removeprefix(BACKENDS["test"].url)
        if path == "/api/v3/dkg/forecasts/schema":
            return response(200, url, json=SCHEMA)
        if self.answer == "no round":
            return response(404, url, json={"detail": "no round is accepting forecasts"})
        return response(200, url, json=ROUND | {"reveal_round": self.reveal_round})

    def post(self, url: str, content: bytes, headers: dict, timeout: int) -> httpx.Response:
        self.posted.append((url, content, headers))
        statuses = {"conflict": 409, "gone": 410, "refused": 400}
        if self.answer in statuses:
            return response(statuses[self.answer], url, json={"detail": self.answer})
        return response(200, url, json=self.receipt(json.loads(content)))

    def receipt(self, body: dict) -> dict:
        received_at = datetime.now(timezone.utc)
        sealed_sha256 = hashlib.sha256(bytes.fromhex(body["sealed"])).hexdigest()
        message = (
            f"{HOTKEY.ss58_address}:{body['question_id']}:{sealed_sha256}:{received_at.isoformat()}"
        )
        receipt = {
            "miner_hotkey": HOTKEY.ss58_address,
            "question_id": body["question_id"],
            "sealed_sha256": sealed_sha256,
            "received_at": received_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "signer_hotkey": RECEIPT_KEY.ss58_address,
            "signature": RECEIPT_KEY.sign(message.encode()).hex(),
        }
        if self.answer == "wrong hash":
            receipt["sealed_sha256"] = "0" * 64
        if self.answer == "forged signature":
            receipt["signature"] = RECEIPT_KEY.sign(b"something else").hex()
        return receipt


class TestSubmit:
    @pytest.fixture
    def backend(self):
        stand_in = StandInBackend()
        with (
            patch("neurons.miner.client.httpx.get", side_effect=stand_in.get),
            patch("neurons.miner.client.httpx.post", side_effect=stand_in.post),
            patch.dict(
                BACKENDS,
                {"test": BACKENDS["test"]._replace(receipt_signer=RECEIPT_KEY.ss58_address)},
            ),
        ):
            yield stand_in

    def test_sends_a_sealed_signed_forecast_and_checks_the_receipt(self, backend: StandInBackend):
        submission = MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

        [(url, content, headers)] = backend.posted
        caller = http_auth.verify(
            headers,
            content,
            method="POST",
            path="/api/v3/dkg/forecasts",
            self_hotkey_ss58=BACKENDS["test"].hotkey,
        )
        body = json.loads(content)
        sealed = bytes.fromhex(body["sealed"])
        assert url == f"{BACKENDS['test'].url}/api/v3/dkg/forecasts"
        assert caller.hotkey_ss58 == HOTKEY.ss58_address
        assert body["question_id"] == QUESTION_ID
        assert timelock.Timelocked.parse(sealed).reveal_round == backend.reveal_round
        assert Keypair(ss58_address=HOTKEY.ss58_address).verify(
            QUESTION_ID.encode() + sealed, bytes.fromhex(body["signature"])
        )
        assert (submission.round_id, submission.reveal_round) == (ROUND["id"], backend.reveal_round)
        assert (submission.sealed, submission.signature) == (body["sealed"], body["signature"])
        assert submission.receipt.sealed_sha256 == hashlib.sha256(sealed).hexdigest()

    def test_seals_compact_utf8(self, backend: StandInBackend):
        forecast = {
            "version": 2,
            "statements": [{"head": "苹果", "relation": "投资", "tail": "台积电", "probability": 0.2}],
        }

        with patch("neurons.miner.client.seal", wraps=seal) as sealing:
            MinerClient("test", HOTKEY).submit(QUESTION_ID, forecast)

        sealed_text = sealing.call_args.args[0]
        assert sealed_text == json.dumps(forecast, ensure_ascii=False, separators=(",", ":"))
        assert "苹果" in sealed_text and " " not in sealed_text

    def test_an_invalid_forecast_is_not_sent(self, backend: StandInBackend):
        with pytest.raises(ForecastNotValid) as error:
            MinerClient("test", HOTKEY).submit(QUESTION_ID, {"version": 2, "statements": []})

        assert error.value.errors[0].path == "statements"
        assert backend.posted == []

    def test_a_forecast_too_large_once_sealed_is_not_sent(self, backend: StandInBackend):
        with patch("neurons.miner.client.SEALED_FORECAST_MAX_BYTES", 100):
            with pytest.raises(ForecastNotValid, match="1 errors") as error:
                MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

        assert error.value.errors[0].path == "(forecast)"
        assert error.value.errors[0].message.endswith(
            "bytes, the limit is 100: shorten the rationales or the sources"
        )
        assert backend.posted == []

    def test_a_question_of_another_round_is_not_sent(self, backend: StandInBackend):
        with pytest.raises(NotAQuestionOfTheRound, match=ROUND["id"]):
            MinerClient("test", HOTKEY).submit(str(uuid.uuid4()), FORECAST)

        assert backend.posted == []

    def test_no_round_accepting(self, backend: StandInBackend):
        backend.answer = "no round"

        with pytest.raises(NoRoundAccepting):
            MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

        assert backend.posted == []

    @pytest.mark.parametrize(
        "answer, error",
        [("conflict", AlreadySubmitted), ("gone", DeadlinePassed), ("refused", BackendRejected)],
    )
    def test_a_refusal(self, backend: StandInBackend, answer: str, error: type[Exception]):
        backend.answer = answer

        with pytest.raises(error, match=answer):
            MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

    @pytest.mark.parametrize(
        "answer, message",
        [
            ("wrong hash", "sealed_sha256 is 000"),
            ("forged signature", "the signature does not verify against the receipt key"),
        ],
    )
    def test_a_receipt_that_does_not_match(
        self, backend: StandInBackend, answer: str, message: str
    ):
        backend.answer = answer

        with pytest.raises(ReceiptInvalid, match="the backend accepted the forecast") as error:
            MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

        assert message in str(error.value)

    def test_a_receipt_from_another_key(self, backend: StandInBackend):
        with patch.dict(
            BACKENDS, {"test": BACKENDS["test"]._replace(receipt_signer=HOTKEY.ss58_address)}
        ):
            with pytest.raises(
                ReceiptInvalid, match=f"signer_hotkey is {RECEIPT_KEY.ss58_address}"
            ):
                MinerClient("test", HOTKEY).submit(QUESTION_ID, FORECAST)

    @pytest.mark.parametrize(
        "env, receipt_signer",
        [
            ("test", "5FnLD5YMV2UedmZDedtR9DJ2qPSD2K4LauaJh4yM6fBm7h6V"),
            ("prod", "5EHw3u757wYSuLYK6RT5aBbeCUdAPsTuyqa6wi6MBLCdH3hD"),
        ],
    )
    def test_the_published_receipt_keys(self, env: str, receipt_signer: str):
        assert BACKENDS[env].receipt_signer == receipt_signer


def verified_path(call) -> str:
    path = call.args[0].removeprefix(BACKENDS["test"].url)
    caller = http_auth.verify(
        call.kwargs["headers"],
        b"",
        method="GET",
        path=path,
        self_hotkey_ss58=BACKENDS["test"].hotkey,
    )
    assert caller.hotkey_ss58 == HOTKEY.ss58_address
    return path


GRAPH_PAGE = {
    "edges": [
        {
            "head": "NVIDIA",
            "head_id": "c4399b0f-ab16-5a64-aab3-89f18ce8e2d3",
            "relation": "SUPPLIES",
            "tail": "Zeta",
            "tail_id": "e5",
            "standing": "stated",
            "support": "news",
            "first_seen": "2026-10-12T15:00:00Z",
            "last_seen": "2026-10-12T15:00:00Z",
            "observations": None,
        }
    ],
    "events": [],
    "next_offset": 500,
}
ENTITY_PAGE = {
    "entities": [
        {"id": "e1", "canonical": "AT&T", "entity_type": "company", "aliases": ["AT&T Inc"]}
    ],
    "next_offset": None,
}


class TestGetGraph:
    def test_without_filters(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/graph"

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(200, url, json=GRAPH_PAGE)
        ) as get:
            page = MinerClient("test", HOTKEY).get_graph(None, [], None, False, 100, 0)

        assert (
            verified_path(get.call_args)
            == "/api/v3/dkg/graph?with_observations=false&limit=100&offset=0"
        )
        assert page == GRAPH_PAGE

    def test_every_filter_is_part_of_the_signed_path(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/graph"
        entity_id = uuid.UUID("c4399b0f-ab16-5a64-aab3-89f18ce8e2d3")
        published_by = datetime(2026, 10, 1, 0, 0, tzinfo=timezone(timedelta(hours=-3)))

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(200, url, json=GRAPH_PAGE)
        ) as get:
            MinerClient("test", HOTKEY).get_graph(
                entity_id, ["SUPPLIES", "INVESTS_IN"], published_by, True, 50, 100
            )

        assert verified_path(get.call_args) == (
            f"/api/v3/dkg/graph?entity_id={entity_id}&relation=SUPPLIES&relation=INVESTS_IN"
            "&published_by=2026-10-01T03:00:00Z&with_observations=true&limit=50&offset=100"
        )


class TestGetEntities:
    @pytest.mark.parametrize(
        "name, encoded",
        [("nvidia", "nvidia"), ("AT&T", "AT%26T"), ("hon hai", "hon%20hai")],
    )
    def test_the_name_is_encoded_before_signing(self, name: str, encoded: str):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/entities"

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(200, url, json=ENTITY_PAGE)
        ) as get:
            page = MinerClient("test", HOTKEY).get_entities(name, [], None, 0)

        assert verified_path(get.call_args) == f"/api/v3/dkg/entities?name={encoded}&offset=0"
        assert page == ENTITY_PAGE

    def test_types_limit_and_offset(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/entities"

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(200, url, json=ENTITY_PAGE)
        ) as get:
            MinerClient("test", HOTKEY).get_entities(None, ["company", "product"], 2, 4)

        assert verified_path(get.call_args) == (
            "/api/v3/dkg/entities?entity_type=company&entity_type=product&limit=2&offset=4"
        )


class TestRefusalMessages:
    def test_field_problems_read_as_one_line_each(self):
        url = f"{BACKENDS['test'].url}/api/v3/dkg/entities"
        problems = {
            "detail": [
                {
                    "type": "string_too_short",
                    "loc": ["query", "name"],
                    "msg": "String should have at least 2 characters",
                },
                {
                    "type": "enum",
                    "loc": ["query", "entity_type", 0],
                    "msg": "Input should be 'company' or 'product'",
                },
            ]
        }

        with patch(
            "neurons.miner.client.httpx.get", return_value=response(422, url, json=problems)
        ):
            with pytest.raises(BackendRejected) as error:
                MinerClient("test", HOTKEY).get_entities("x", ["nonsense"], None, 0)

        assert error.value.detail == (
            "name: String should have at least 2 characters; "
            "entity_type.0: Input should be 'company' or 'product'"
        )
