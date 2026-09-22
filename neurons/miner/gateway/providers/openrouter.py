import typing

import aiohttp

from neurons.validator.models.openrouter import (
    DecisionQuestion,
    OpenRouterCompletion,
    OpenRouterDecision,
)


class OpenRouterClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenRouter API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://openrouter.ai/api/v1"
        self.__timeout = aiohttp.ClientTimeout(total=300)
        self.__headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Identifier": "Numinous",
        }

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, typing.Any]],
        temperature: float = 0.7,
        max_tokens: typing.Optional[int] = None,
        tools: typing.Optional[list[dict[str, typing.Any]]] = None,
        tool_choice: typing.Optional[typing.Any] = None,
        **kwargs: typing.Any,
    ) -> OpenRouterCompletion:
        body: dict[str, typing.Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        body.update(kwargs)

        url = f"{self.__base_url}/chat/completions"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return OpenRouterCompletion.model_validate(data)

    async def decisions(
        self,
        model: str,
        state: typing.Any,
        questions: dict[str, DecisionQuestion],
    ) -> OpenRouterDecision:
        body: dict[str, typing.Any] = {
            "model": model,
            "state": state,
            "questions": {
                key: question.model_dump(exclude_none=True) for key, question in questions.items()
            },
        }

        url = f"{self.__base_url}/systemone"

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                return OpenRouterDecision.model_validate(data)
