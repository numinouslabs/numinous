import typing

import aiohttp

from neurons.validator.models.openai import OpenAIResponse


class OpenAIClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://api.openai.com/v1"
        self.__timeout = aiohttp.ClientTimeout(total=300)
        self.__headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def create_response(
        self,
        model: str,
        input: list[dict[str, typing.Any]],
        temperature: typing.Optional[float] = None,
        max_output_tokens: typing.Optional[int] = None,
        tools: typing.Optional[list[dict[str, typing.Any]]] = None,
        tool_choice: typing.Optional[typing.Any] = None,
        instructions: typing.Optional[str] = None,
        **kwargs: typing.Any,
    ) -> OpenAIResponse:
        body = {
            "model": model,
            "input": input,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if instructions is not None:
            body["instructions"] = instructions

        body.update(kwargs)

        url = f"{self.__base_url}/responses"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return OpenAIResponse.model_validate(data)
