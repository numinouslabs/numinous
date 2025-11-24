import functools
import logging
from typing import Any, Callable

import aiohttp
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def handle_provider_errors(provider: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                error_message = f"{provider} API error: {str(e)}"
                logger.error(error_message)

                if isinstance(e, aiohttp.ClientResponseError):
                    raise HTTPException(status_code=e.status, detail=error_message)

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_message,
                )

        return wrapper

    return decorator
