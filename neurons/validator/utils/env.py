import os

from pydantic import BaseModel


class EnvironmentVariables(BaseModel):
    INLINE_LOGS: bool
    GIT_COMMIT_HASH: str


ENVIRONMENT_VARIABLES = EnvironmentVariables(
    INLINE_LOGS=os.getenv("INLINE_LOGS", "false").lower()
    in [
        "true",
        "1",
    ],
    GIT_COMMIT_HASH=os.getenv("GIT_COMMIT_HASH", "-"),
)
