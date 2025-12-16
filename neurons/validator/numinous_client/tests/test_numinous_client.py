import asyncio
import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from aiohttp import ClientResponseError
from aiohttp.web import Response
from aioresponses import aioresponses
from bittensor_wallet import Wallet
from yarl import URL

from neurons.validator.models.chutes import ChuteModel, ChutesCompletion
from neurons.validator.models.desearch import AISearchResponse, ModelEnum, ToolEnum
from neurons.validator.models.numinous_client import (
    BatchUpdateAgentRunsRequest,
    ChutesInferenceRequest,
    CreateAgentRunRequest,
    CreateAgentRunResponse,
    DesearchAISearchRequest,
    GetAgentsResponse,
    GetEventsDeletedResponse,
    GetEventsResolvedResponse,
    GetEventsResponse,
    GetWeightsResponse,
    PostAgentLogsRequestBody,
    PostAgentRunsRequestBody,
    PostPredictionsRequestBody,
    PostScoresRequestBody,
    UpdateAgentRunRequest,
)
from neurons.validator.numinous_client.client import NuminousClient, NuminousEnvType
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __version__


def make_client_test_env(env: NuminousEnvType = "test"):
    logger = MagicMock(spec=NuminousLogger)

    hotkey_mock = MagicMock(
        sign=MagicMock(side_effect=lambda x: x.encode("utf-8")),
        ss58_address="ss58_address",
        public_key=b"public_key",
    )

    bt_wallet = MagicMock(
        spec=Wallet, get_hotkey=MagicMock(return_value=hotkey_mock), hotkey=hotkey_mock
    )

    return NuminousClient(env=env, logger=logger, bt_wallet=bt_wallet)


class TestNuminousClient:
    @pytest.fixture
    def client_test_env(self):
        return make_client_test_env(env="test")

    @pytest.mark.parametrize(
        "client,expected_base_url",
        [
            (
                make_client_test_env(env="test"),
                "https://stg.numinous.earth",
            ),
            (
                make_client_test_env(env="prod"),
                "https://numinous.earth",
            ),
        ],
    )
    async def test_default_session_config(self, client: NuminousClient, expected_base_url: str):
        session = client.create_session()

        assert session._base_url == URL(expected_base_url)
        assert session._timeout.total == 90

        # Verify that the default headers were set correctly
        assert session.headers["Validator-Version"] == __version__
        assert session.headers["Validator-Hash"] == commit_short_hash
        assert session.headers["Validator-Public-Key"] == b"public_key".hex()

    async def test_logger_interceptors_success(self, client_test_env: NuminousClient):
        logger = client_test_env._NuminousClient__logger

        context = SimpleNamespace()
        method = "GET"
        response_status = 200
        url = "/test"

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_end(
            None,
            context,
            MagicMock(
                response=Response(status=response_status),
                method=method,
                url=URL(url),
            ),
        )

        # Verify the debug call
        logger.debug.assert_called_once_with(
            "Http request finished",
            extra={
                "response_status": response_status,
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_error(self, client_test_env: NuminousClient):
        logger = client_test_env._NuminousClient__logger

        context = SimpleNamespace()
        method = "GET"
        response_status = 500
        response_message = '{"message": "error"}'
        url = "/test"

        response = MagicMock(status=response_status, text=AsyncMock(return_value=response_message))
        params = MagicMock(response=response, method=method, url=URL(url))

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_end(None, context, params)

        # Verify the error call
        logger.error.assert_called_once_with(
            "Http request failed",
            extra={
                "response_status": response_status,
                "response_message": response_message,
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_exception(self, client_test_env: NuminousClient):
        logger = client_test_env._NuminousClient__logger

        context = SimpleNamespace()
        method = "GET"
        url = "/test"

        # Test success response
        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_exception(
            None,
            context,
            MagicMock(method=method, url=URL(url)),
        )

        # Verify the debug call
        logger.exception.assert_called_once_with(
            "Http request exception",
            extra={
                "method": method,
                "url": url,
                "elapsed_time_ms": pytest.approx(100, abs=100),
            },
        )

    async def test_logger_interceptors_cancelled_error_exception(
        self, client_test_env: NuminousClient
    ):
        logger = client_test_env._NuminousClient__logger

        context = SimpleNamespace()
        method = "GET"
        url = "/test"

        await client_test_env.on_request_start(None, context, None)
        await client_test_env.on_request_exception(
            None,
            context,
            MagicMock(method=method, url=URL(url), exception=asyncio.exceptions.CancelledError()),
        )

        # Verify no log call
        logger.exception.assert_not_called()

    @pytest.mark.parametrize(
        "from_date,offset,limit",
        [
            (None, 0, 10),  # Missing from_date
            (1234567890, None, 10),  # Missing offset
            (1234567890, 0, None),  # Missing limit
        ],
    )
    async def test_get_events_invalid_params(
        self, client_test_env: NuminousClient, from_date, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_events(from_date=from_date, offset=offset, limit=limit)

    async def test_get_events_response(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "event_id": "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
                    "cutoff": 1733616000,
                    "title": "Will Tesla stock price reach $2000 by 2025?",
                    "description": (
                        "This market will resolve to 'Yes' if the closing stock price of Tesla reaches or exceeds $2000 "
                        "on any trading day in 2025. Otherwise, this market will resolve to 'No'.\n\n"
                        "Resolution source: NASDAQ official data."
                    ),
                    "market_type": "BINARY",
                    "created_at": 1733200000,
                    "event_metadata": {},
                },
                {
                    "event_id": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdef",
                    "cutoff": 1733617000,
                    "title": "Will AI surpass human intelligence by 2030?",
                    "description": (
                        "This market will resolve to 'Yes' if credible sources, including major AI researchers, "
                        "announce that AI has surpassed general human intelligence before December 31, 2030."
                    ),
                    "market_type": "POLYMARKET",
                    "created_at": 1733210000,
                    "event_metadata": {},
                },
            ],
        }

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v1/validators/events?from_date=1234567890&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_events(from_date=1234567890, offset=0, limit=10)

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == GetEventsResponse.model_validate(mock_response_data)

    async def test_get_events_error_raised(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v1/validators/events?from_date=1234567890&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_events(from_date=1234567890, offset=0, limit=10)

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    @pytest.mark.parametrize(
        "resolved_since,offset,limit",
        [
            (None, 0, 10),  # Missing resolved_since
            (1, 0, 10),  # Not str resolved_since
            ("2025-01-23T16:10:15Z", None, 10),  # Missing offset
            ("2025-01-23T16:10:15Z", 0, None),  # Missing limit
        ],
    )
    async def test_get_resolved_events_invalid_params(
        self, client_test_env: NuminousClient, resolved_since, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_resolved_events(
                resolved_since=resolved_since, offset=offset, limit=limit
            )

    async def test_get_resolved_events_response(self, client_test_env: NuminousClient):
        # Define mock response data

        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "event_id": "21a1578e-705b-4935-9dd1-5138bf279ad0",
                    "market_type": "MARKET_TYPE",
                    "answer": 0,
                    "created_at": "2025-01-20T16:10:15Z",
                    "resolved_at": "2025-01-23T16:10:15Z",
                    "forecasts": {},
                },
                {
                    "event_id": "2837d80d-6c90-4b10-9dda-44ee0db617a3",
                    "market_type": "MARKET_TYPE",
                    "answer": 1,
                    "created_at": "2025-01-20T16:10:15Z",
                    "resolved_at": "2025-01-23T16:10:15Z",
                    "forecasts": {"2025-01-20T16:10:15Z": 0.0001},
                },
            ],
        }

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events/resolved?resolved_since=2000-12-30T14:30&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_resolved_events(
                resolved_since="2000-12-30T14:30", offset=0, limit=10
            )

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == GetEventsResolvedResponse.model_validate(mock_response_data)

            assert len(result.items[0].forecasts) == 0
            assert len(result.items[1].forecasts) == 1

    async def test_get_resolved_events_error_raised(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v2/events/resolved?resolved_since=2000-12-30T14:30&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_resolved_events(
                    resolved_since="2000-12-30T14:30", offset=0, limit=10
                )

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    @pytest.mark.parametrize(
        "deleted_since,offset,limit",
        [
            (None, 0, 10),  # Missing deleted_since
            (1, 0, 10),  # Not str deleted_since
            ("2025-01-23T16:10:15Z", None, 10),  # Missing offset
            ("2025-01-23T16:10:15Z", 0, None),  # Missing limit
        ],
    )
    async def test_get_events_deleted_invalid_params(
        self, client_test_env: NuminousClient, deleted_since, offset, limit
    ):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_events_deleted(
                deleted_since=deleted_since, offset=offset, limit=limit
            )

    async def test_get_events_deleted_response(self, client_test_env: NuminousClient):
        # Define mock response data

        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "event_id": "21a1578e-705b-4935-9dd1-5138bf279ad0",
                    "market_type": "MARKET_TYPE",
                    "created_at": "2025-01-22T16:10:15Z",
                    "deleted_at": "2025-01-23T16:10:15Z",
                },
                {
                    "event_id": "2837d80d-6c90-4b10-9dda-44ee0db617a3",
                    "market_type": "MARKET_TYPE",
                    "created_at": "2025-01-22T16:10:15Z",
                    "deleted_at": "2025-01-23T16:10:15Z",
                },
            ],
        }

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            mocked.get(
                "/api/v2/events/deleted?deleted_since=2000-12-30T14:30&offset=0&limit=10",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_events_deleted(
                deleted_since="2000-12-30T14:30", offset=0, limit=10
            )

            mocked.assert_called_once()

            # Verify the response matches the mock data
            assert result == GetEventsDeletedResponse.model_validate(mock_response_data)

    async def test_get_events_deleted_error_raised(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"message": "Internal error"}
        status_code = 500

        # Use aioresponses context manager to mock HTTP requests
        with aioresponses() as mocked:
            url_path = "/api/v2/events/deleted?deleted_since=2000-12-30T14:30&offset=0&limit=10"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_events_deleted(
                    deleted_since="2000-12-30T14:30", offset=0, limit=10
                )

            mocked.assert_called_with(url_path)

            # Assert the exception
            assert e.value.status == status_code

    def test_make_auth_headers(self, client_test_env: NuminousClient):
        body = {"fake": "body"}

        auth_headers = client_test_env.make_auth_headers(data=json.dumps(body))

        encoded = base64.b64encode(json.dumps(body).encode("utf-8")).decode("utf-8")

        assert auth_headers == {
            "Authorization": f"Bearer {encoded}",
            "Validator": "ss58_address",
        }

    def test_make_get_auth_headers(self, client_test_env: NuminousClient):
        auth_headers = client_test_env.make_get_auth_headers()

        assert "Authorization" in auth_headers
        assert "Validator" in auth_headers
        assert "X-Payload" in auth_headers

        assert auth_headers["Validator"] == "ss58_address"

        payload = auth_headers["X-Payload"]
        assert payload.startswith("ss58_address:")

        timestamp_part = payload.split(":")[-1]
        assert timestamp_part.isdigit()

        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
        assert auth_headers["Authorization"] == f"Bearer {encoded_payload}"

    async def test_post_predictions(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        request_body = PostPredictionsRequestBody.model_validate(
            {
                "submissions": [
                    {
                        "unique_event_id": "unique_event_id",
                        "provider_type": "event_type",
                        "prediction": 1,
                        "interval_start_minutes": 100,
                        "interval_agg_prediction": 1.0,
                        "interval_agg_count": 1,
                        "interval_datetime": datetime.now(timezone.utc),
                        "miner_hotkey": "miner_hotkey",
                        "miner_uid": 1,
                        "validator_hotkey": "validator_hotkey",
                        "validator_uid": 2,
                        "submitted_at": datetime.now(timezone.utc),
                        "run_id": "d23e4567-e89b-12d3-a456-42661417400c",
                        "version_id": "e23e4567-e89b-12d3-a456-42661417400d",
                        "title": None,
                        "outcome": None,
                    }
                ],
                "events": None,
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/data"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_predictions(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            # Verify the response matches
            assert result == mock_response_data

    async def test_post_predictions_error_raised(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        request_body = PostPredictionsRequestBody.model_validate(
            {
                "submissions": [
                    {
                        "unique_event_id": "unique_event_id",
                        "provider_type": "event_type",
                        "prediction": 1,
                        "interval_start_minutes": 100,
                        "interval_agg_prediction": 1.0,
                        "interval_agg_count": 1,
                        "interval_datetime": datetime.now(timezone.utc),
                        "miner_hotkey": "miner_hotkey",
                        "miner_uid": 1,
                        "validator_hotkey": "validator_hotkey",
                        "validator_uid": 2,
                        "submitted_at": datetime.now(timezone.utc),
                        "run_id": "f23e4567-e89b-12d3-a456-42661417400e",
                        "version_id": "023e4567-e89b-12d3-a456-42661417400f",
                        "title": None,
                        "outcome": None,
                    }
                ],
                "events": None,
            }
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/data"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_predictions(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            # Assert the exception
            assert e.value.status == status_code

    async def test_post_scores(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        request_body = PostScoresRequestBody.model_validate(
            {
                "results": [
                    {
                        "event_id": "event_id",
                        "prediction": 1,
                        "answer": 1,
                        "miner_hotkey": "miner_hotkey",
                        "miner_uid": 1,
                        "miner_score": 1,
                        "miner_effective_score": 1,
                        "validator_hotkey": "validator_hotkey",
                        "validator_uid": 2,
                        "spec_version": "1.3.3",
                        "registered_date": datetime.now(timezone.utc),
                        "scored_at": datetime.now(timezone.utc),
                    }
                ]
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/results"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_scores(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            # Verify the response matches
            assert result == mock_response_data

    async def test_post_scores_error_raised(self, client_test_env: NuminousClient):
        # Define mock response data
        mock_response_data = {"fake_response": "ok"}

        request_body = PostScoresRequestBody.model_validate(
            {
                "results": [
                    {
                        "event_id": "event_id",
                        "prediction": 1,
                        "answer": 1,
                        "miner_hotkey": "miner_hotkey",
                        "miner_uid": 1,
                        "miner_score": 1,
                        "miner_effective_score": 1,
                        "validator_hotkey": "validator_hotkey",
                        "validator_uid": 2,
                        "spec_version": "1.3.3",
                        "registered_date": datetime.now(timezone.utc),
                        "scored_at": datetime.now(timezone.utc),
                    }
                ]
            }
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/results"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_scores(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            # Assert the exception
            assert e.value.status == status_code

    async def test_post_agent_logs(self, client_test_env: NuminousClient):
        mock_response_data = {}

        request_body = PostAgentLogsRequestBody.model_validate(
            {
                "run_id": "123e4567-e89b-12d3-a456-426614174000",
                "log_content": "Agent execution log:\nStep 1: Initialize\nStep 2: Process\nStep 3: Complete",
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/logs"

            mocked.post(
                url_path,
                status=204,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_agent_logs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert result == mock_response_data

    async def test_post_agent_logs_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"error": "Failed to process logs"}

        request_body = PostAgentLogsRequestBody.model_validate(
            {
                "run_id": "223e4567-e89b-12d3-a456-426614174001",
                "log_content": "Test log content",
            }
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/logs"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_agent_logs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert e.value.status == status_code

    async def test_post_agent_runs(self, client_test_env: NuminousClient):
        mock_response_data = {}

        request_body = PostAgentRunsRequestBody.model_validate(
            {
                "runs": [
                    {
                        "run_id": "123e4567-e89b-12d3-a456-426614174000",
                        "miner_uid": 10,
                        "miner_hotkey": "miner_hotkey_1",
                        "vali_uid": 5,
                        "vali_hotkey": "validator_hotkey",
                        "status": "SUCCESS",
                        "event_id": "event_123",
                        "version_id": "223e4567-e89b-12d3-a456-426614174001",
                        "is_final": True,
                    },
                    {
                        "run_id": "323e4567-e89b-12d3-a456-426614174002",
                        "miner_uid": 20,
                        "miner_hotkey": "miner_hotkey_2",
                        "vali_uid": 5,
                        "vali_hotkey": "validator_hotkey",
                        "status": "SANDBOX_TIMEOUT",
                        "event_id": "event_456",
                        "version_id": "423e4567-e89b-12d3-a456-426614174003",
                        "is_final": False,
                    },
                ]
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs"

            mocked.post(
                url_path,
                status=204,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.post_agent_runs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert result == mock_response_data

    async def test_post_agent_runs_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"error": "Failed to process runs"}

        request_body = PostAgentRunsRequestBody.model_validate(
            {
                "runs": [
                    {
                        "run_id": "523e4567-e89b-12d3-a456-426614174004",
                        "miner_uid": 30,
                        "miner_hotkey": "miner_hotkey_3",
                        "vali_uid": 5,
                        "vali_hotkey": "validator_hotkey",
                        "status": "INTERNAL_AGENT_ERROR",
                        "event_id": "event_789",
                        "version_id": "623e4567-e89b-12d3-a456-426614174005",
                        "is_final": True,
                    }
                ]
            }
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.post_agent_runs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert e.value.status == status_code

    async def test_create_agent_run(self, client_test_env: NuminousClient):
        mock_response_data = {"run_id": "723e4567-e89b-12d3-a456-426614174006"}

        request_body = CreateAgentRunRequest(
            miner_uid=10,
            miner_hotkey="miner_hotkey_1",
            vali_uid=5,
            vali_hotkey="validator_hotkey",
            event_id="event_123",
            version_id="823e4567-e89b-12d3-a456-426614174007",
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs/create"

            mocked.post(
                url_path,
                status=201,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.create_agent_run(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert isinstance(result, CreateAgentRunResponse)
            assert str(result.run_id) == mock_response_data["run_id"]

    async def test_create_agent_run_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"error": "Failed to create agent run"}

        request_body = CreateAgentRunRequest(
            miner_uid=10,
            miner_hotkey="miner_hotkey_1",
            vali_uid=5,
            vali_hotkey="validator_hotkey",
            event_id="event_456",
            version_id="923e4567-e89b-12d3-a456-426614174008",
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs/create"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.create_agent_run(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert e.value.status == status_code

    async def test_put_agent_runs(self, client_test_env: NuminousClient):
        request_body = BatchUpdateAgentRunsRequest(
            runs=[
                UpdateAgentRunRequest(
                    run_id=UUID("623e4567-e89b-12d3-a456-426614174006"),
                    status="SUCCESS",
                    is_final=True,
                ),
                UpdateAgentRunRequest(
                    run_id=UUID("723e4567-e89b-12d3-a456-426614174007"),
                    status="INTERNAL_AGENT_ERROR",
                    is_final=True,
                ),
            ]
        )

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs"

            mocked.put(url_path, status=204)

            await client_test_env.put_agent_runs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="PUT", data=request_body.model_dump_json()
            )

    async def test_put_agent_runs_error_raised(self, client_test_env: NuminousClient):
        request_body = BatchUpdateAgentRunsRequest(
            runs=[
                UpdateAgentRunRequest(
                    run_id=UUID("823e4567-e89b-12d3-a456-426614174008"),
                    status="SUCCESS",
                    is_final=True,
                )
            ]
        )

        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents/runs"

            mocked.put(
                url_path,
                status=status_code,
                body=json.dumps({"error": "Internal error"}).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.put_agent_runs(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="PUT", data=request_body.model_dump_json()
            )

            assert e.value.status == status_code

    @pytest.mark.parametrize(
        "offset,limit",
        [
            (None, 50),
            (0, None),
        ],
    )
    async def test_get_agents_invalid_params(self, client_test_env: NuminousClient, offset, limit):
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.get_agents(offset=offset, limit=limit)

    async def test_get_agents_response(self, client_test_env: NuminousClient):
        mock_response_data = {
            "count": 2,
            "items": [
                {
                    "version_id": "123e4567-e89b-12d3-a456-426614174000",
                    "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                    "miner_uid": 42,
                    "agent_name": "TestAgent1",
                    "version_number": 1,
                    "created_at": "2024-01-15T10:30:00Z",
                    "code": "ZGVmIGFnZW50X21haW4oKToKICAgIHJldHVybiAwLjU=",
                },
                {
                    "version_id": "223e4567-e89b-12d3-a456-426614174001",
                    "miner_hotkey": "5DTestHotkeyAnotherMiner123456789012345678901234",
                    "miner_uid": 43,
                    "agent_name": "TestAgent2",
                    "version_number": 2,
                    "created_at": "2024-01-15T11:30:00Z",
                    "code": "aW1wb3J0IHJhbmRvbQpkZWYgYWdlbnRfbWFpbigpOgogICAgcmV0dXJuIHJhbmRvbS5yYW5kb20oKQ==",
                },
            ],
        }

        with aioresponses() as mocked:
            mocked.get(
                "/api/v1/validators/agents?offset=0&limit=50",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_agents(offset=0, limit=50)

            mocked.assert_called_once()

            assert result == GetAgentsResponse.model_validate(mock_response_data)

    async def test_get_agents_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"message": "Service unavailable"}
        status_code = 503

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/agents?offset=0&limit=50"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_agents(offset=0, limit=50)

            mocked.assert_called_with(url_path)

            assert e.value.status == status_code

    async def test_get_agents_pagination(self, client_test_env: NuminousClient):
        mock_response_data = {
            "count": 100,
            "items": [
                {
                    "version_id": "323e4567-e89b-12d3-a456-426614174002",
                    "miner_hotkey": "5HPagedAgentHotkey123456789012345678901234567",
                    "miner_uid": 44,
                    "agent_name": "PagedAgent",
                    "version_number": 1,
                    "created_at": "2024-01-15T12:30:00Z",
                    "code": "ZGVmIGFnZW50X21haW4oKToKICAgIHJldHVybiAwLjc=",
                },
            ],
        }

        with aioresponses() as mocked:
            mocked.get(
                "/api/v1/validators/agents?offset=10&limit=20",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_agents(offset=10, limit=20)

            mocked.assert_called_once()

            assert result.count == 100
            assert len(result.items) == 1

    async def test_chutes_inference_invalid_params(self, client_test_env: NuminousClient):
        invalid_body = {"invalid_field": "value"}
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.chutes_inference(body=invalid_body)

    async def test_chutes_inference(self, client_test_env: NuminousClient):
        mock_response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-ai/DeepSeek-R1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "This is a test response."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        request_body = ChutesInferenceRequest.model_validate(
            {
                "run_id": "123e4567-e89b-12d3-a456-426614174000",
                "model": ChuteModel.DEEPSEEK_R1,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/gateway/chutes/chat/completions"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.chutes_inference(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert isinstance(result, ChutesCompletion)
            assert result.id == "chatcmpl-123"
            assert result.model == "deepseek-ai/DeepSeek-R1"
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "This is a test response."

    async def test_chutes_inference_with_dict(self, client_test_env: NuminousClient):
        mock_response_data = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "Qwen/Qwen3-32B",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response from dict input."},
                    "finish_reason": "stop",
                }
            ],
        }

        request_body_dict = {
            "run_id": "223e4567-e89b-12d3-a456-426614174001",
            "model": ChuteModel.QWEN3_32B,
            "messages": [
                {"role": "user", "content": "Test message"},
            ],
        }

        with aioresponses() as mocked:
            url_path = "/api/gateway/chutes/chat/completions"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.chutes_inference(body=request_body_dict)

            assert isinstance(result, ChutesCompletion)
            assert result.id == "chatcmpl-456"

    async def test_chutes_inference_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"error": "Service unavailable"}

        request_body = ChutesInferenceRequest.model_validate(
            {
                "run_id": "323e4567-e89b-12d3-a456-426614174002",
                "model": ChuteModel.DEEPSEEK_V3_1,
                "messages": [{"role": "user", "content": "Test"}],
            }
        )
        status_code = 503

        with aioresponses() as mocked:
            url_path = "/api/gateway/chutes/chat/completions"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.chutes_inference(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )
            assert e.value.status == status_code

    async def test_desearch_ai_search_invalid_params(self, client_test_env: NuminousClient):
        invalid_body = {"invalid_field": "value"}
        with pytest.raises(ValueError, match="Invalid parameters"):
            await client_test_env.desearch_ai_search(body=invalid_body)

    async def test_desearch_ai_search(self, client_test_env: NuminousClient):
        mock_response_data = {"id": "search-123", "results": [], "summary": "Test summary"}

        request_body = DesearchAISearchRequest.model_validate(
            {
                "run_id": "423e4567-e89b-12d3-a456-426614174003",
                "prompt": "What is quantum computing?",
                "model": ModelEnum.NOVA,
                "tools": [ToolEnum.WEB],
            }
        )

        with aioresponses() as mocked:
            url_path = "/api/gateway/desearch/ai/search"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.desearch_ai_search(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert isinstance(result, AISearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response_data

    async def test_desearch_ai_search_with_dict(self, client_test_env: NuminousClient):
        mock_response_data = {"id": "search-456", "results": []}

        request_body_dict = {
            "run_id": "523e4567-e89b-12d3-a456-426614174004",
            "prompt": "Latest AI news",
            "model": ModelEnum.ORBIT,
            "tools": [ToolEnum.WEB],
        }

        with aioresponses() as mocked:
            url_path = "/api/gateway/desearch/ai/search"

            mocked.post(
                url_path,
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.desearch_ai_search(body=request_body_dict)

            assert isinstance(result, AISearchResponse)
            assert result.model_dump(exclude_none=True) == mock_response_data

    async def test_desearch_ai_search_error_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"error": "Rate limit exceeded"}

        request_body = DesearchAISearchRequest.model_validate(
            {
                "run_id": "623e4567-e89b-12d3-a456-426614174005",
                "prompt": "Test query",
                "model": ModelEnum.NOVA,
                "tools": [ToolEnum.WEB],
            }
        )

        status_code = 429

        with aioresponses() as mocked:
            url_path = "/api/gateway/desearch/ai/search"

            mocked.post(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.desearch_ai_search(body=request_body)

            mocked.assert_called_with(
                url=url_path, method="POST", data=request_body.model_dump_json()
            )

            assert e.value.status == status_code

    async def test_get_weights_response(self, client_test_env: NuminousClient):
        mock_response_data = {
            "aggregated_at": "2025-01-30T12:00:00Z",
            "weights": [
                {
                    "miner_uid": 0,
                    "miner_hotkey": "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                    "aggregated_weight": 0.5,
                },
                {
                    "miner_uid": 1,
                    "miner_hotkey": "5Dpqn31QEwkqXoMJQF2xvPn9Rh6dDo9CKk1aq3PJxJZgP5Wf",
                    "aggregated_weight": 0.3,
                },
                {
                    "miner_uid": 2,
                    "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                    "aggregated_weight": 0.2,
                },
            ],
            "count": 3,
        }

        with aioresponses() as mocked:
            mocked.get(
                "/api/v1/validators/weights",
                status=200,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            result = await client_test_env.get_weights()

            mocked.assert_called_once()

            assert result == GetWeightsResponse.model_validate(mock_response_data)
            assert len(result.weights) == 3
            assert result.count == 3
            assert result.weights[0].miner_uid == 0
            assert result.weights[0].aggregated_weight == 0.5

    async def test_get_weights_error_503_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"message": "No weights available yet"}
        status_code = 503

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/weights"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_weights()

            mocked.assert_called_with(url_path)

            assert e.value.status == status_code

    async def test_get_weights_error_500_raised(self, client_test_env: NuminousClient):
        mock_response_data = {"message": "Internal server error"}
        status_code = 500

        with aioresponses() as mocked:
            url_path = "/api/v1/validators/weights"
            mocked.get(
                url_path,
                status=status_code,
                body=json.dumps(mock_response_data).encode("utf-8"),
            )

            with pytest.raises(ClientResponseError) as e:
                await client_test_env.get_weights()

            mocked.assert_called_with(url_path)

            assert e.value.status == status_code
