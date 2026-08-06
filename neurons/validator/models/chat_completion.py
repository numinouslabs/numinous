import typing

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Role of the message author")
    content: typing.Optional[str] = Field(None, description="Message content")
    tool_calls: typing.Optional[list[dict[str, typing.Any]]] = Field(
        None, description="Tool calls made by the model"
    )

    model_config = ConfigDict(extra="allow")


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    message: ChatCompletionMessage = Field(..., description="Chat completion message")
    finish_reason: typing.Optional[str] = Field(None, description="Reason for completion stop")

    model_config = ConfigDict(extra="allow")
