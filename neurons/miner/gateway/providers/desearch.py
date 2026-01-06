import datetime
import typing

import aiohttp

from neurons.validator.models.desearch import (
    AISearchResponse,
    WebCrawlResponse,
    WebLinksResponse,
    WebSearchResponse,
    XPostResponse,
    XPostSummary,
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

    async def x_search(
        self,
        query: str,
        sort: typing.Optional[typing.Literal["Top", "Latest"]] = "Top",
        user: typing.Optional[str] = None,
        start_date: typing.Optional[datetime.datetime] = None,
        end_date: typing.Optional[datetime.datetime] = None,
        lang: typing.Optional[str] = None,
        verified: typing.Optional[bool] = None,
        blue_verified: typing.Optional[bool] = None,
        is_quote: typing.Optional[bool] = None,
        is_video: typing.Optional[bool] = None,
        is_image: typing.Optional[bool] = None,
        min_retweets: typing.Optional[int] = None,
        min_replies: typing.Optional[int] = None,
        min_likes: typing.Optional[int] = None,
        count: int = 20,
    ) -> list[XPostSummary]:
        params = {
            "query": query,
            "sort": sort,
            "user": user,
            "start_date": start_date,
            "end_date": end_date,
            "lang": lang,
            "verified": verified,
            "blue_verified": blue_verified,
            "is_quote": is_quote,
            "is_video": is_video,
            "is_image": is_image,
            "min_retweets": min_retweets,
            "min_replies": min_replies,
            "min_likes": min_likes,
            "count": count,
        }

        params = {k: v for k, v in params.items() if v is not None}

        url = f"{self.__base_url}/twitter"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return [XPostSummary.model_validate(item) for item in data]

    async def fetch_x_post(self, post_id: str) -> XPostResponse:
        url = f"{self.__base_url}/twitter/post"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url, params={"id": post_id}) as response:
                response.raise_for_status()
                data = await response.json()
                return XPostResponse.model_validate(data)
