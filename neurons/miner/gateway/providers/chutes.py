import typing

import aiohttp

from neurons.validator.models.chutes import ChutesCompletion, ChuteStatus


class ChutesClient:
    __api_key: str
    __base_inference_url: str
    __base_api_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Chutes API key is not set")

        self.__api_key = api_key
        self.__base_inference_url = "https://llm.chutes.ai/v1"
        self.__base_api_url = "https://api.chutes.ai"
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
        tools: typing.Optional[list[dict[str, typing.Any]]] = None,
        tool_choice: typing.Optional[typing.Any] = None,
        **kwargs: typing.Any,
    ) -> ChutesCompletion:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(kwargs)

        url = f"{self.__base_inference_url}/chat/completions"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return ChutesCompletion.model_validate(data)

    async def get_chutes_status(self) -> list[ChuteStatus]:
        url = f"{self.__base_api_url}/chutes/utilization"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                return [ChuteStatus.model_validate(item) for item in data]
