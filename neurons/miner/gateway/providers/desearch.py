import typing

import aiohttp

from neurons.validator.models.desearch import (
    AISearchResponse,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
)


class DesearchClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Desearch API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://api.desearch.ai"
        self.__timeout = aiohttp.ClientTimeout(total=300)
        self.__headers = {
            "Authorization": self.__api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def ai_search(
        self,
        prompt: str,
        model: typing.Optional[str] = None,
        tools: typing.Optional[list[str]] = None,
        date_filter: typing.Optional[str] = None,
        result_type: typing.Optional[str] = None,
        system_message: typing.Optional[str] = None,
        count: int = 10,
    ) -> AISearchResponse:
        body = {
            "prompt": prompt,
            "tools": tools or ["web"],
            "model": model or "NOVA",
            "streaming": False,
            "count": count,
        }

        if date_filter is not None:
            body["date_filter"] = date_filter
        if result_type is not None:
            body["result_type"] = result_type
        if system_message is not None:
            body["system_message"] = system_message

        url = f"{self.__base_url}/desearch/ai/search"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return AISearchResponse.model_validate(data)

    async def web_links_search(
        self,
        prompt: str,
        model: typing.Optional[str] = None,
        tools: typing.Optional[list[str]] = None,
        count: int = 10,
    ) -> WebLinksResponse:
        body = {
            "prompt": prompt,
            "tools": tools or ["web"],
            "count": count,
            "model": model or "NOVA",
        }

        url = f"{self.__base_url}/desearch/ai/search/links/web"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return WebLinksResponse.model_validate(data)

    async def web_search(
        self, query: str, num_results: int = 10, start: int = 0
    ) -> WebSearchResponse:
        params = {
            "query": query,
            "num_results": num_results,
            "start": start,
        }

        url = f"{self.__base_url}/web"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return WebSearchResponse.model_validate(data)

    async def web_crawl(self, url: str) -> WebCrawlResponse:
        crawl_url = f"{self.__base_url}/web/crawl"

        async with aiohttp.ClientSession(timeout=self.__timeout) as session:
            async with session.get(
                crawl_url, params={"url": url}, headers=self.__headers
            ) as response:
                response.raise_for_status()
                data = await response.text()
                return WebCrawlResponse(url=url, content=data)
