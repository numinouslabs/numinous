import typing

import aiohttp

from neurons.validator.models.perplexity import PerplexityCompletion


class PerplexityClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Perplexity API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://api.perplexity.ai"
        self.__timeout = aiohttp.ClientTimeout(total=300)
        self.__headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, typing.Any]],
        temperature: typing.Optional[float] = None,
        max_tokens: typing.Optional[int] = None,
        search_recency_filter: typing.Optional[str] = None,
        **kwargs: typing.Any,
    ) -> PerplexityCompletion:
        body = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if search_recency_filter is not None:
            body["search_recency_filter"] = search_recency_filter

        body.update(kwargs)

        url = f"{self.__base_url}/chat/completions"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return PerplexityCompletion.model_validate(data)
