import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from bittensor import Wallet
from click.testing import CliRunner

from cli import numi
from neurons.miner.client import (
    AlreadySubmitted,
    BackendRejected,
    DeadlinePassed,
    ForecastNotValid,
    NoRoundAccepting,
    NotAQuestionOfTheRound,
    ReceiptInvalid,
)
from neurons.miner.commands.common import ExitCode
from neurons.miner.models import (
    ForecastError,
    ForecastReceipt,
    MinerForecastResponse,
    RoundResponse,
    Submission,
)
from neurons.miner.tests.test_miner_client import (
    ENTITY_PAGE,
    GRAPH_PAGE,
    MY_FORECAST,
    ROUND,
    SCHEMA,
)

VALID = {
    "version": 2,
    "statements": [
        {"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 0.15}
    ],
}
INVALID = {
    "version": 2,
    "statements": [{"head": "Apple", "relation": "INVESTS_IN", "tail": "TSMC", "probability": 1.5}],
}


@pytest.fixture
def wallet_path(tmp_path: Path) -> Path:
    wallet = Wallet(name="miner", hotkey="default", path=str(tmp_path))
    wallet.regenerate_hotkey(
        seed="0x" + "03" * 32, use_password=False, overwrite=True, suppress=True
    )
    return tmp_path


@pytest.fixture
def environment(wallet_path: Path) -> dict[str, str]:
    return {
        "NUMINOUS_ENV": "test",
        "NUMINOUS_WALLET": "miner",
        "NUMINOUS_WALLET_PATH": str(wallet_path),
    }


def run(arguments: list[str], environment: dict[str, str], stdin: str | None = None):
    return CliRunner().invoke(numi, arguments, env=environment, input=stdin)


class TestRound:
    def test_prints_the_round(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_round",
            return_value=RoundResponse.model_validate(ROUND),
        ):
            result = run(["round"], environment)

        assert result.exit_code == ExitCode.OK
        assert json.loads(result.output)["reveal_round"] == 32713961

    def test_no_round_accepting(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_round",
            side_effect=NoRoundAccepting("no round is accepting forecasts"),
        ):
            result = run(["round"], environment)

        assert result.exit_code == ExitCode.NO_ROUND_ACCEPTING
        assert json.loads(result.output) == {
            "error": "no_round_accepting",
            "message": "no round is accepting forecasts",
        }

    def test_a_backend_rejection(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_round",
            side_effect=BackendRejected(400, "Neuron not registered on subnet"),
        ):
            result = run(["round"], environment)

        assert result.exit_code == ExitCode.BACKEND_REJECTED
        assert json.loads(result.output)["message"] == "400: Neuron not registered on subnet"

    def test_the_backend_unreachable(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_round",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = run(["round"], environment)

        assert result.exit_code == ExitCode.BACKEND_UNREACHABLE
        assert json.loads(result.output)["message"] == "ConnectError: connection refused"

    def test_a_missing_hotkey(self, environment: dict[str, str]):
        result = run(["round", "--hotkey", "other"], environment)

        assert result.exit_code == ExitCode.HOTKEY_NOT_FOUND
        assert json.loads(result.output)["error"] == "hotkey_not_found"

    def test_flags_work_without_env_vars(self, wallet_path: Path):
        with patch(
            "neurons.miner.client.MinerClient.get_round",
            return_value=RoundResponse.model_validate(ROUND),
        ):
            result = run(
                ["round", "--env", "test", "--wallet", "miner", "--wallet-path", str(wallet_path)],
                {},
            )

        assert result.exit_code == ExitCode.OK

    def test_the_wallet_is_required(self, wallet_path: Path):
        result = run(["round", "--env", "test"], {})

        assert result.exit_code == 2
        assert "Missing option '--wallet'" in result.output


class TestSchema:
    def test_prints_the_schema(self, environment: dict[str, str]):
        with patch("neurons.miner.client.MinerClient.get_schema", return_value=SCHEMA):
            result = run(["schema"], environment)

        assert result.exit_code == ExitCode.OK
        assert json.loads(result.output) == SCHEMA


class TestValidate:
    @pytest.fixture(autouse=True)
    def served_schema(self):
        with patch("neurons.miner.client.MinerClient.get_schema", return_value=SCHEMA):
            yield

    def test_a_valid_forecast_from_a_file(self, environment: dict[str, str], tmp_path: Path):
        forecast_file = tmp_path / "forecast.json"
        forecast_file.write_text(json.dumps(VALID))

        result = run(["validate", str(forecast_file)], environment)

        assert result.exit_code == ExitCode.OK
        assert json.loads(result.output) == {"valid": True}

    def test_a_valid_forecast_from_stdin(self, environment: dict[str, str]):
        result = run(["validate"], environment, stdin=json.dumps(VALID))

        assert result.exit_code == ExitCode.OK
        assert json.loads(result.output) == {"valid": True}

    def test_an_invalid_forecast(self, environment: dict[str, str]):
        result = run(["validate"], environment, stdin=json.dumps(INVALID))

        assert result.exit_code == ExitCode.INVALID_FORECAST
        assert json.loads(result.output) == {
            "valid": False,
            "errors": [
                {
                    "path": "statements/0/probability",
                    "message": "1.5 is greater than the maximum of 1",
                }
            ],
        }

    def test_not_json(self, environment: dict[str, str]):
        result = run(["validate"], environment, stdin="not json at all")

        assert result.exit_code == ExitCode.INVALID_FORECAST
        assert json.loads(result.output)["error"] == "invalid_forecast"


class TestForecasts:
    def test_prints_every_forecast(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_my_forecasts",
            return_value=[MinerForecastResponse.model_validate(MY_FORECAST)],
        ) as get_my_forecasts:
            result = run(["forecasts"], environment)

        assert result.exit_code == ExitCode.OK
        get_my_forecasts.assert_called_once_with(None, None, None)
        [printed] = json.loads(result.output)["forecasts"]
        assert printed["sealed_sha256"] == MY_FORECAST["sealed_sha256"]
        assert [item["status"] for item in printed["statements"]] == ["right", "wrong", "waiting"]

    def test_none_yet(self, environment: dict[str, str]):
        with patch("neurons.miner.client.MinerClient.get_my_forecasts", return_value=[]):
            result = run(["forecasts"], environment)

        assert result.exit_code == ExitCode.OK
        assert json.loads(result.output) == {"forecasts": []}

    @pytest.mark.parametrize(
        "changed_since",
        ["2026-10-13T18:02:00Z", "2026-10-13T18:02:00", "2026-10-13T15:02:00-03:00"],
    )
    def test_filters_reach_the_client(self, environment: dict[str, str], changed_since: str):
        round_id = ROUND["id"]
        entity_id = ROUND["questions"][0]["entity_id"]

        with patch(
            "neurons.miner.client.MinerClient.get_my_forecasts", return_value=[]
        ) as get_my_forecasts:
            result = run(
                [
                    "forecasts",
                    "--round",
                    round_id,
                    "--entity",
                    entity_id,
                    "--changed-since",
                    changed_since,
                ],
                environment,
            )

        assert result.exit_code == ExitCode.OK
        get_my_forecasts.assert_called_once_with(
            uuid.UUID(round_id),
            uuid.UUID(entity_id),
            datetime(2026, 10, 13, 18, 2, tzinfo=timezone.utc),
        )

    @pytest.mark.parametrize(
        "arguments, message",
        [
            (["--changed-since", "yesterday"], "'yesterday' is not an ISO 8601 date-time"),
            (["--round", "not-a-uuid"], "'not-a-uuid' is not a valid UUID"),
            (["--entity", "nvidia"], "'nvidia' is not a valid UUID"),
        ],
    )
    def test_bad_filters(self, environment: dict[str, str], arguments: list[str], message: str):
        with patch("neurons.miner.client.MinerClient.get_my_forecasts") as get_my_forecasts:
            result = run(["forecasts", *arguments], environment)

        assert result.exit_code == 2
        assert message in result.output
        get_my_forecasts.assert_not_called()

    def test_a_backend_rejection(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_my_forecasts",
            side_effect=BackendRejected(401, "nonce was already used"),
        ):
            result = run(["forecasts"], environment)

        assert result.exit_code == ExitCode.BACKEND_REJECTED
        assert json.loads(result.output)["message"] == "401: nonce was already used"


SUBMISSION = Submission(
    round_id=ROUND["id"],
    reveal_round=33091222,
    sealed="abcd",
    signature="ef01",
    receipt=ForecastReceipt(
        miner_hotkey="5Miner",
        question_id=ROUND["questions"][0]["id"],
        sealed_sha256="sha",
        received_at=datetime(2026, 10, 15, 14, 14, 47, tzinfo=timezone.utc),
        signer_hotkey="5Receipts",
        signature="beef",
    ),
)


class TestSubmit:
    @pytest.fixture(autouse=True)
    def receipts_folder(self, tmp_path: Path):
        with patch("neurons.miner.commands.submit.RECEIPTS_FOLDER", tmp_path / "receipts"):
            yield tmp_path / "receipts"

    def test_sends_and_saves_the_receipt(self, environment: dict[str, str], receipts_folder: Path):
        question_id = ROUND["questions"][0]["id"]

        with patch("neurons.miner.client.MinerClient.submit", return_value=SUBMISSION) as submit:
            result = run(
                ["submit", "--question", question_id], environment, stdin=json.dumps(VALID)
            )

        assert result.exit_code == ExitCode.OK
        submit.assert_called_once_with(question_id, VALID)
        printed = json.loads(result.output)
        saved_file = receipts_folder / "5Miner" / ROUND["id"] / f"{question_id}.json"
        assert printed == {
            "sent": True,
            "round_id": ROUND["id"],
            "receipt": SUBMISSION.receipt.model_dump(mode="json"),
            "saved_to": str(saved_file),
        }
        assert json.loads(saved_file.read_text()) == SUBMISSION.model_dump(mode="json") | {
            "forecast": VALID
        }

    def test_reads_the_forecast_from_a_file(self, environment: dict[str, str], tmp_path: Path):
        forecast_file = tmp_path / "forecast.json"
        forecast_file.write_text(json.dumps(VALID))

        with patch("neurons.miner.client.MinerClient.submit", return_value=SUBMISSION) as submit:
            result = run(
                ["submit", "--question", ROUND["questions"][0]["id"], str(forecast_file)],
                environment,
            )

        assert result.exit_code == ExitCode.OK
        assert submit.call_args.args[1] == VALID

    def test_an_invalid_forecast(self, environment: dict[str, str], receipts_folder: Path):
        errors = [ForecastError(path="statements", message="[] should be non-empty")]

        with patch("neurons.miner.client.MinerClient.submit", side_effect=ForecastNotValid(errors)):
            result = run(
                ["submit", "--question", ROUND["questions"][0]["id"]],
                environment,
                stdin=json.dumps(VALID),
            )

        assert result.exit_code == ExitCode.INVALID_FORECAST
        assert json.loads(result.output) == {
            "valid": False,
            "errors": [{"path": "statements", "message": "[] should be non-empty"}],
        }
        assert not receipts_folder.exists()

    @pytest.mark.parametrize(
        "error, exit_code",
        [
            (NoRoundAccepting("no round is accepting forecasts"), ExitCode.NO_ROUND_ACCEPTING),
            (AlreadySubmitted("already submitted"), ExitCode.ALREADY_SUBMITTED),
            (DeadlinePassed("deadline passed"), ExitCode.DEADLINE_PASSED),
            (NotAQuestionOfTheRound("not in the round"), ExitCode.NOT_A_QUESTION_OF_THE_ROUND),
            (ReceiptInvalid("the backend accepted the forecast, but"), ExitCode.RECEIPT_INVALID),
            (BackendRejected(400, "seal does not match"), ExitCode.BACKEND_REJECTED),
        ],
    )
    def test_each_outcome_has_its_exit_code(
        self,
        environment: dict[str, str],
        receipts_folder: Path,
        error: Exception,
        exit_code: ExitCode,
    ):
        with patch("neurons.miner.client.MinerClient.submit", side_effect=error):
            result = run(
                ["submit", "--question", ROUND["questions"][0]["id"]],
                environment,
                stdin=json.dumps(VALID),
            )

        assert result.exit_code == exit_code
        assert json.loads(result.output) == {
            "error": exit_code.name.lower(),
            "message": str(error),
        }
        assert not receipts_folder.exists()

    def test_not_json_is_not_sent(self, environment: dict[str, str]):
        with patch("neurons.miner.client.MinerClient.submit") as submit:
            result = run(
                ["submit", "--question", ROUND["questions"][0]["id"]], environment, stdin="nope"
            )

        assert result.exit_code == ExitCode.INVALID_FORECAST
        submit.assert_not_called()

    @pytest.mark.parametrize(
        "arguments, message",
        [([], "Missing option '--question'"), (["--question", "apple"], "is not a valid UUID")],
    )
    def test_the_question_is_required(
        self, environment: dict[str, str], arguments: list[str], message: str
    ):
        with patch("neurons.miner.client.MinerClient.submit") as submit:
            result = run(["submit", *arguments], environment, stdin=json.dumps(VALID))

        assert result.exit_code == 2
        assert message in result.output
        submit.assert_not_called()


class TestGraph:
    def test_defaults(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_graph", return_value=GRAPH_PAGE
        ) as get_graph:
            result = run(["graph"], environment)

        assert result.exit_code == ExitCode.OK
        get_graph.assert_called_once_with(None, [], None, False, 100, 0)
        assert json.loads(result.output) == GRAPH_PAGE

    def test_every_flag_reaches_the_client(self, environment: dict[str, str]):
        entity_id = "c4399b0f-ab16-5a64-aab3-89f18ce8e2d3"

        with patch(
            "neurons.miner.client.MinerClient.get_graph", return_value=GRAPH_PAGE
        ) as get_graph:
            result = run(
                [
                    "graph",
                    "--entity",
                    entity_id,
                    "--relation",
                    "SUPPLIES",
                    "--relation",
                    "INVESTS_IN",
                    "--published-by",
                    "2026-09-30T21:00:00-03:00",
                    "--with-observations",
                    "--limit",
                    "50",
                    "--offset",
                    "100",
                ],
                environment,
            )

        assert result.exit_code == ExitCode.OK
        get_graph.assert_called_once_with(
            uuid.UUID(entity_id),
            ["SUPPLIES", "INVESTS_IN"],
            datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc),
            True,
            50,
            100,
        )

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--limit", "0"],
            ["--limit", "2001"],
            ["--offset", "-1"],
            ["--published-by", "yesterday"],
            ["--entity", "nvidia"],
        ],
    )
    def test_bad_flags(self, environment: dict[str, str], arguments: list[str]):
        with patch("neurons.miner.client.MinerClient.get_graph") as get_graph:
            result = run(["graph", *arguments], environment)

        assert result.exit_code == 2
        get_graph.assert_not_called()

    def test_a_backend_rejection(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_graph",
            side_effect=BackendRejected(404, "unknown entity"),
        ):
            result = run(["graph", "--entity", str(uuid.uuid4())], environment)

        assert result.exit_code == ExitCode.BACKEND_REJECTED
        assert json.loads(result.output)["message"] == "404: unknown entity"


class TestEntities:
    def test_defaults(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_entities", return_value=ENTITY_PAGE
        ) as get_entities:
            result = run(["entities"], environment)

        assert result.exit_code == ExitCode.OK
        get_entities.assert_called_once_with(None, [], None, 0)
        assert json.loads(result.output) == ENTITY_PAGE

    def test_every_flag_reaches_the_client(self, environment: dict[str, str]):
        with patch(
            "neurons.miner.client.MinerClient.get_entities", return_value=ENTITY_PAGE
        ) as get_entities:
            result = run(
                [
                    "entities",
                    "--name",
                    "AT&T",
                    "--type",
                    "company",
                    "--type",
                    "product",
                    "--limit",
                    "2",
                    "--offset",
                    "4",
                ],
                environment,
            )

        assert result.exit_code == ExitCode.OK
        get_entities.assert_called_once_with("AT&T", ["company", "product"], 2, 4)

    def test_bad_limit(self, environment: dict[str, str]):
        with patch("neurons.miner.client.MinerClient.get_entities") as get_entities:
            result = run(["entities", "--limit", "0"], environment)

        assert result.exit_code == 2
        get_entities.assert_not_called()
