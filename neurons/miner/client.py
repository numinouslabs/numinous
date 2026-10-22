import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Literal, NamedTuple
from urllib.parse import quote

import httpx
from bittensor import http_auth
from bittensor_wallet import Keypair
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from neurons.miner.models import (
    ForecastError,
    ForecastReceipt,
    MinerForecastResponse,
    MinerForecastsPage,
    RoundResponse,
    Submission,
)
from neurons.miner.sealing import SEALED_FORECAST_MAX_BYTES, receipt_message, seal, sign_forecast

MinerEnv = Literal["test", "prod"]


class _Backend(NamedTuple):
    url: str
    hotkey: str
    receipt_signer: str


BACKENDS: dict[MinerEnv, _Backend] = {
    "prod": _Backend(
        "https://numinous.earth",
        "5CyqMG9eDSHbNgsRW49e4SRSAGSaEWoiVVWv6fD7eWRPCuPB",
        "5EHw3u757wYSuLYK6RT5aBbeCUdAPsTuyqa6wi6MBLCdH3hD",
    ),
    "test": _Backend(
        "https://stg.numinous.earth",
        "5EgzbPFTqhpKwZQSVDe4PfDhKArvWMNEWXPJoiVTqKe1FW45",
        "5FnLD5YMV2UedmZDedtR9DJ2qPSD2K4LauaJh4yM6fBm7h6V",
    ),
}


class BackendRejected(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class NoRoundAccepting(Exception):
    pass


class NotAQuestionOfTheRound(Exception):
    pass


class ForecastNotValid(Exception):
    def __init__(self, errors: list[ForecastError]):
        super().__init__(f"{len(errors)} errors")
        self.errors = errors


class AlreadySubmitted(Exception):
    pass


class DeadlinePassed(Exception):
    pass


class ReceiptInvalid(Exception):
    pass


def _field_problem(problem: dict[str, JsonValue]) -> str:
    field = ".".join(str(part) for part in problem["loc"] if part not in ("query", "body"))
    return f"{field}: {problem['msg']}"


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return response.text

    if isinstance(detail, list):
        return "; ".join(_field_problem(problem) for problem in detail)
    return str(detail)


def _query_moment(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _with_query(path: str, parameters: list[tuple[str, str | None]]) -> str:
    present = "&".join(
        f"{name}={quote(value, safe=':')}" for name, value in parameters if value is not None
    )
    return f"{path}?{present}"


def forecast_errors(forecast: JsonValue, schema: dict[str, JsonValue]) -> list[ForecastError]:
    errors = Draft202012Validator(schema).iter_errors(forecast)
    return [
        ForecastError(
            path="/".join(str(part) for part in error.absolute_path) or "(forecast)",
            message=error.message,
        )
        for error in sorted(errors, key=lambda error: [str(part) for part in error.absolute_path])
    ]


class MinerClient:
    def __init__(self, env: MinerEnv, hotkey: Keypair):
        self._backend = BACKENDS[env]
        self._hotkey = hotkey

    def get_round(self) -> RoundResponse:
        try:
            return RoundResponse.model_validate(self._signed_get("/api/v3/dkg/questions"))
        except BackendRejected as rejected:
            if rejected.status == 404:
                raise NoRoundAccepting(rejected.detail) from None
            raise

    def get_schema(self) -> dict[str, JsonValue]:
        return self._signed_get("/api/v3/dkg/forecasts/schema")

    def get_my_forecasts(
        self,
        round_id: uuid.UUID | None,
        entity_id: uuid.UUID | None,
        changed_since: datetime | None,
    ) -> list[MinerForecastResponse]:
        forecasts: list[MinerForecastResponse] = []
        offset = 0

        while True:
            path = _with_query(
                "/api/v3/dkg/miner/forecasts",
                [
                    ("round_id", str(round_id) if round_id else None),
                    ("entity_id", str(entity_id) if entity_id else None),
                    ("changed_since", _query_moment(changed_since) if changed_since else None),
                    ("offset", str(offset)),
                ],
            )
            page = MinerForecastsPage.model_validate(self._signed_get(path))
            forecasts.extend(page.forecasts)

            if page.next_offset is None:
                return forecasts

            offset = page.next_offset

    def get_graph(
        self,
        entity_id: uuid.UUID | None,
        relations: list[str],
        published_by: datetime | None,
        with_observations: bool,
        limit: int,
        offset: int,
    ) -> dict[str, JsonValue]:
        path = _with_query(
            "/api/v3/dkg/graph",
            [
                ("entity_id", str(entity_id) if entity_id else None),
                *[("relation", relation) for relation in relations],
                ("published_by", _query_moment(published_by) if published_by else None),
                ("with_observations", str(with_observations).lower()),
                ("limit", str(limit)),
                ("offset", str(offset)),
            ],
        )
        return self._signed_get(path)

    def get_entities(
        self, name: str | None, entity_types: list[str], limit: int | None, offset: int
    ) -> dict[str, JsonValue]:
        path = _with_query(
            "/api/v3/dkg/entities",
            [
                ("name", name),
                *[("entity_type", entity_type) for entity_type in entity_types],
                ("limit", str(limit) if limit else None),
                ("offset", str(offset)),
            ],
        )
        return self._signed_get(path)

    def submit(self, question_id: str, forecast: JsonValue) -> Submission:
        errors = forecast_errors(forecast, self.get_schema())
        if errors:
            raise ForecastNotValid(errors)

        current = self.get_round()
        if question_id not in {question.id for question in current.questions}:
            raise NotAQuestionOfTheRound(
                f"{question_id} is not a question of the round accepting forecasts ({current.id})"
            )

        sealed = self._seal_within_limit(forecast, current.reveal_round)
        signature = sign_forecast(self._hotkey, question_id, sealed)
        receipt = self._post_forecast(question_id, sealed, signature)
        self._check_receipt(receipt, question_id, sealed)

        return Submission(
            round_id=current.id,
            reveal_round=current.reveal_round,
            sealed=sealed.hex(),
            signature=signature.hex(),
            receipt=receipt,
        )

    def _seal_within_limit(self, forecast: JsonValue, reveal_round: int) -> bytes:
        sealed = seal(json.dumps(forecast, ensure_ascii=False, separators=(",", ":")), reveal_round)
        if len(sealed) > SEALED_FORECAST_MAX_BYTES:
            raise ForecastNotValid(
                [
                    ForecastError(
                        path="(forecast)",
                        message=(
                            f"the sealed forecast is {len(sealed):,} bytes, the limit is "
                            f"{SEALED_FORECAST_MAX_BYTES:,}: shorten the rationales or the sources"
                        ),
                    )
                ]
            )
        return sealed

    def _post_forecast(self, question_id: str, sealed: bytes, signature: bytes) -> ForecastReceipt:
        body = {"question_id": question_id, "sealed": sealed.hex(), "signature": signature.hex()}
        try:
            return ForecastReceipt.model_validate(self._signed_post("/api/v3/dkg/forecasts", body))
        except BackendRejected as rejected:
            if rejected.status == 409:
                raise AlreadySubmitted(rejected.detail) from None
            if rejected.status == 410:
                raise DeadlinePassed(rejected.detail) from None
            raise

    def _check_receipt(self, receipt: ForecastReceipt, question_id: str, sealed: bytes) -> None:
        mismatches = [
            f"{name} is {received}, expected {expected}"
            for name, received, expected in (
                ("signer_hotkey", receipt.signer_hotkey, self._backend.receipt_signer),
                ("miner_hotkey", receipt.miner_hotkey, self._hotkey.ss58_address),
                ("question_id", receipt.question_id, question_id),
                ("sealed_sha256", receipt.sealed_sha256, hashlib.sha256(sealed).hexdigest()),
            )
            if received != expected
        ]
        message = receipt_message(
            receipt.miner_hotkey, receipt.question_id, receipt.sealed_sha256, receipt.received_at
        )
        if not mismatches and not Keypair(ss58_address=receipt.signer_hotkey).verify(
            message, bytes.fromhex(receipt.signature)
        ):
            mismatches.append("the signature does not verify against the receipt key")

        if mismatches:
            raise ReceiptInvalid(
                "the backend accepted the forecast, but its receipt does not match what was sent: "
                + "; ".join(mismatches)
            )

    def _signed_get(self, path: str) -> JsonValue:
        headers = http_auth.sign(
            self._hotkey, method="GET", path=path, receiver_ss58=self._backend.hotkey
        )
        response = httpx.get(self._backend.url + path, headers=headers, timeout=60)
        if response.is_error:
            raise BackendRejected(response.status_code, _detail(response))

        return response.json()

    def _signed_post(self, path: str, body: dict[str, str]) -> JsonValue:
        content = json.dumps(body).encode()
        headers = http_auth.sign(
            self._hotkey,
            method="POST",
            path=path,
            body=content,
            receiver_ss58=self._backend.hotkey,
        ) | {"Content-Type": "application/json"}
        response = httpx.post(
            self._backend.url + path, content=content, headers=headers, timeout=60
        )
        if response.is_error:
            raise BackendRejected(response.status_code, _detail(response))

        return response.json()
