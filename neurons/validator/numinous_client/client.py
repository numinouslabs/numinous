import asyncio
import base64
import time

import aiohttp
import aiohttp.typedefs
from bittensor_wallet import Wallet
from pydantic import ValidationError

from neurons.validator.models.chutes import ChutesCompletion
from neurons.validator.models.desearch import AISearchResponse
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
)
from neurons.validator.utils.config import NuminousEnvType
from neurons.validator.utils.git import commit_short_hash
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __version__


class NuminousClient:
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: aiohttp.typedefs.LooseHeaders
    __logger: NuminousLogger
    __bt_wallet: Wallet

    def __init__(self, env: NuminousEnvType, logger: NuminousLogger, bt_wallet: Wallet) -> None:
        # Validate env
        if not isinstance(env, str):
            raise TypeError("env must be an instance of str.")

        # Validate logger
        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        # Validate bt_wallet
        if not isinstance(bt_wallet, Wallet):
            raise TypeError("bt_wallet must be an instance of Wallet.")

        self.__logger = logger
        self.__base_url = (
            "https://numinous.earth" if env == "prod" else "https://stg.numinous.earth"
        )
        self.__timeout = aiohttp.ClientTimeout(total=90)  # In seconds

        self.__bt_wallet = bt_wallet

        self.__headers = {
            "Validator-Version": __version__,
            "Validator-Hash": commit_short_hash,
            "Validator-Public-Key": bt_wallet.hotkey.public_key.hex(),
        }

    def create_session(self, other_headers: dict = None) -> aiohttp.ClientSession:
        headers = self.__headers.copy()
        if other_headers:
            headers.update(other_headers)

        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)

        return aiohttp.ClientSession(
            base_url=self.__base_url,
            timeout=self.__timeout,
            headers=headers,
            trace_configs=[trace_config],
        )

    async def on_request_start(self, _, trace_config_ctx, __):
        trace_config_ctx.start_time = time.time()

    async def on_request_end(self, _, trace_config_ctx, params: aiohttp.TraceRequestEndParams):
        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        response_status = params.response.status

        extra = {
            "response_status": response_status,
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        if response_status >= 400:
            # Add message if error
            response_message = await params.response.text()
            extra["response_message"] = response_message

            self.__logger.error("Http request failed", extra=extra)
        else:
            self.__logger.debug("Http request finished", extra=extra)

    async def on_request_exception(
        self, _, trace_config_ctx, params: aiohttp.TraceRequestExceptionParams
    ):
        exception = params.exception

        # Ignore cancelled exceptions
        if isinstance(exception, asyncio.exceptions.CancelledError):
            return

        elapsed_time_ms = round((time.time() - trace_config_ctx.start_time) * 1000)

        extra = {
            "method": params.method,
            "url": str(params.url),
            "elapsed_time_ms": elapsed_time_ms,
        }

        self.__logger.exception("Http request exception", extra=extra)

    def make_auth_headers(self, data: str) -> dict[str, str]:
        hot_key = self.__bt_wallet.get_hotkey()
        signed = base64.b64encode(hot_key.sign(data)).decode("utf-8")

        return {
            "Authorization": f"Bearer {signed}",
            "Validator": hot_key.ss58_address,
        }

    def make_get_auth_headers(self) -> dict[str, str]:
        timestamp = str(int(time.time()))
        payload = f"{self.__bt_wallet.hotkey.ss58_address}:{timestamp}"
        return {
            **self.make_auth_headers(data=payload),
            "X-Payload": payload,
        }

    async def get_events(self, from_date: int, offset: int, limit: int):
        # Check that all parameters are provided
        if from_date is None or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        auth_headers = self.make_get_auth_headers()

        async with self.create_session(other_headers=auth_headers) as session:
            path = f"/api/v1/validators/events?from_date={from_date}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                data = await response.json()

                return GetEventsResponse.model_validate(data)

    async def get_events_deleted(self, deleted_since: str, offset: int, limit: int):
        # Check that all parameters are provided
        if not isinstance(deleted_since, str) or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        async with self.create_session() as session:
            path = f"/api/v2/events/deleted?deleted_since={deleted_since}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                data = await response.json()

                return GetEventsDeletedResponse.model_validate(data)

    async def get_resolved_events(self, resolved_since: str, offset: int, limit: int):
        # Check that all parameters are provided
        if not isinstance(resolved_since, str) or offset is None or limit is None:
            raise ValueError("Invalid parameters")

        async with self.create_session() as session:
            path = f"/api/v2/events/resolved?resolved_since={resolved_since}&offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                data = await response.json()

                return GetEventsResolvedResponse.model_validate(data)

    async def post_predictions(self, body: PostPredictionsRequestBody):
        if not isinstance(body, PostPredictionsRequestBody):
            raise ValueError("Invalid parameter")

        assert len(body.submissions) > 0

        data = body.model_dump_json()

        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/data"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                return await response.json()

    async def post_scores(self, body: PostScoresRequestBody):
        if not isinstance(body, PostScoresRequestBody):
            raise ValueError("Invalid parameter")

        assert len(body.results) > 0

        data = body.model_dump_json()

        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/results"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                return await response.json()

    async def post_agent_logs(self, body: PostAgentLogsRequestBody):
        if not isinstance(body, PostAgentLogsRequestBody):
            raise ValueError("Invalid parameter")

        data = body.model_dump_json()
        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/agents/logs"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                return await response.json()

    async def post_agent_runs(self, body: PostAgentRunsRequestBody):
        if not isinstance(body, PostAgentRunsRequestBody):
            raise ValueError("Invalid parameter")

        assert len(body.runs) > 0

        data = body.model_dump_json()

        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/agents/runs"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                return await response.json()

    async def create_agent_run(self, body: CreateAgentRunRequest) -> CreateAgentRunResponse:
        if not isinstance(body, CreateAgentRunRequest):
            raise ValueError("Invalid parameter")

        data = body.model_dump_json()
        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/agents/runs/create"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                response_data = await response.json()
                return CreateAgentRunResponse.model_validate(response_data)

    async def put_agent_runs(self, body: BatchUpdateAgentRunsRequest) -> None:
        if not isinstance(body, BatchUpdateAgentRunsRequest):
            raise ValueError("Invalid parameter")

        assert len(body.runs) > 0

        data = body.model_dump_json()
        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/v1/validators/agents/runs"

            async with session.put(path, data=data) as response:
                response.raise_for_status()

    async def get_agents(self, offset: int, limit: int):
        if offset is None or limit is None:
            raise ValueError("Invalid parameters")

        auth_headers = self.make_get_auth_headers()

        async with self.create_session(other_headers=auth_headers) as session:
            path = f"/api/v1/validators/agents?offset={offset}&limit={limit}"

            async with session.get(path) as response:
                response.raise_for_status()

                data = await response.json()

                return GetAgentsResponse.model_validate(data)

    async def chutes_inference(self, body: dict | ChutesInferenceRequest):
        if isinstance(body, dict):
            try:
                body = ChutesInferenceRequest.model_validate(body)
            except ValidationError as e:
                raise ValueError(f"Invalid parameters: {e}")

        data = body.model_dump_json()
        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/gateway/chutes/chat/completions"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                data = await response.json()
                return ChutesCompletion.model_validate(data)

    async def desearch_ai_search(self, body: dict | DesearchAISearchRequest):
        if isinstance(body, dict):
            try:
                body = DesearchAISearchRequest.model_validate(body)
            except ValidationError as e:
                raise ValueError(f"Invalid parameters: {e}")

        data = body.model_dump_json()
        auth_headers = self.make_auth_headers(data=data)

        async with self.create_session(
            other_headers={**auth_headers, "Content-Type": "application/json"}
        ) as session:
            path = "/api/gateway/desearch/ai/search"

            async with session.post(path, data=data) as response:
                response.raise_for_status()

                data = await response.json()
                return AISearchResponse.model_validate(data)

    async def get_weights(self):
        auth_headers = self.make_get_auth_headers()

        async with self.create_session(other_headers=auth_headers) as session:
            path = "/api/v1/validators/weights"

            async with session.get(path) as response:
                response.raise_for_status()

                data = await response.json()

                return GetWeightsResponse.model_validate(data)
