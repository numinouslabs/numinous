import typing

import aiohttp

from neurons.validator.models.azure_openai import AzureOpenAIResponse


class AzureOpenAIClient:
    
    __api_key: str
    __resource_name: str
    __api_version: typing.Optional[str]
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(
        self,
        api_key: str,
        resource_name: str,
        api_version: typing.Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("Azure OpenAI API key is not set")
        if not resource_name:
            raise ValueError("Azure OpenAI resource name is not set")

        self.__api_key = api_key
        self.__resource_name = resource_name
        self.__api_version = api_version
        self.__timeout = aiohttp.ClientTimeout(total=300)
        self.__headers = {
            "api-key": self.__api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_base_url(self) -> str:
        return f"https://{self.__resource_name}.openai.azure.com"

    async def create_response(
        self,
        deployment: str,
        input: list[dict[str, typing.Any]],
        temperature: typing.Optional[float] = None,
        max_output_tokens: typing.Optional[int] = None,
        tools: typing.Optional[list[dict[str, typing.Any]]] = None,
        tool_choice: typing.Optional[typing.Any] = None,
        instructions: typing.Optional[str] = None,
        **kwargs: typing.Any,
    ) -> AzureOpenAIResponse:
        body = {
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

        base_url = f"{self._get_base_url()}/openai/deployments/{deployment}/responses"
        
        if self.__api_version:
            url = f"{base_url}?api-version={self.__api_version}"
        else:
            url = base_url

        async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "model" not in data:
                    data["model"] = deployment
                    
                return AzureOpenAIResponse.model_validate(data)