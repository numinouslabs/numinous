import json
import time
from datetime import datetime, timezone
from typing import NamedTuple

import aiohttp
from bittensor import Wallet, http_auth
from pydantic import JsonValue

from neurons.validator.models.dkg_client import (
    ClosedRoundsPage,
    ResultsPage,
    RoundHashesPage,
    ScoringRun,
    StoredScores,
)
from neurons.validator.utils.config import NuminousEnvType
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __version__


class _Backend(NamedTuple):
    url: str
    hotkey: str


_BACKENDS: dict[NuminousEnvType, _Backend] = {
    "prod": _Backend("https://numinous.earth", "5CyqMG9eDSHbNgsRW49e4SRSAGSaEWoiVVWv6fD7eWRPCuPB"),
    "test": _Backend(
        "https://stg.numinous.earth", "5EgzbPFTqhpKwZQSVDe4PfDhKArvWMNEWXPJoiVTqKe1FW45"
    ),
}


def _query_moment(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DkgClient:
    __backend: _Backend
    __timeout: aiohttp.ClientTimeout
    __logger: NuminousLogger
    __bt_wallet: Wallet

    def __init__(self, env: NuminousEnvType, logger: NuminousLogger, bt_wallet: Wallet) -> None:
        if env not in _BACKENDS:
            raise ValueError(f"env must be one of {list(_BACKENDS)}.")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        if not isinstance(bt_wallet, Wallet):
            raise TypeError("bt_wallet must be an instance of Wallet.")

        self.__backend = _BACKENDS[env]
        self.__timeout = aiohttp.ClientTimeout(total=90)
        self.__logger = logger
        self.__bt_wallet = bt_wallet

    def __session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            base_url=self.__backend.url,
            timeout=self.__timeout,
            headers={"Validator-Version": __version__},
        )

    async def __read(
        self, response: aiohttp.ClientResponse, path: str, start_time: float
    ) -> JsonValue:
        extra = {
            "path": path,
            "response_status": response.status,
            "elapsed_time_ms": round((time.time() - start_time) * 1000),
        }
        if response.status >= 400:
            extra["response_message"] = await response.text()
            self.__logger.error("Http request failed", extra=extra)
        else:
            self.__logger.debug("Http request finished", extra=extra)

        response.raise_for_status()

        return await response.json()

    async def __signed_get(self, path: str) -> JsonValue:
        headers = http_auth.sign(
            self.__bt_wallet,
            method="GET",
            path=path,
            receiver_ss58=self.__backend.hotkey,
        )
        start_time = time.time()

        async with self.__session() as session:
            async with session.get(path, headers=headers) as response:
                return await self.__read(response, path, start_time)

    async def __signed_post(self, path: str, body: JsonValue) -> JsonValue:
        content = json.dumps(body).encode()
        headers = http_auth.sign(
            self.__bt_wallet,
            method="POST",
            path=path,
            body=content,
            receiver_ss58=self.__backend.hotkey,
        ) | {"Content-Type": "application/json"}
        start_time = time.time()

        async with self.__session() as session:
            async with session.post(path, data=content, headers=headers) as response:
                return await self.__read(response, path, start_time)

    async def get_closed_rounds(self, closed_since: datetime, offset: int) -> ClosedRoundsPage:
        page = await self.__signed_get(
            f"/api/v3/dkg/rounds?closed_since={_query_moment(closed_since)}&offset={offset}"
        )
        return ClosedRoundsPage.model_validate(page)

    async def get_round_hashes(self, round_id: str, offset: int) -> RoundHashesPage:
        page = await self.__signed_get(f"/api/v3/dkg/rounds/{round_id}/hashes?offset={offset}")
        return RoundHashesPage.model_validate(page)

    async def get_results(self, since: datetime, offset: int) -> ResultsPage:
        page = await self.__signed_get(
            f"/api/v3/dkg/results?since={_query_moment(since)}&offset={offset}"
        )
        return ResultsPage.model_validate(page)

    async def post_scores(self, scoring_run: ScoringRun) -> StoredScores:
        stored = await self.__signed_post("/api/v3/dkg/scores", scoring_run.model_dump(mode="json"))
        return StoredScores.model_validate(stored)
