import hashlib
import json
import typing
from functools import wraps
from threading import Lock

from pydantic import BaseModel

T = typing.TypeVar("T")

_cache: dict[str, typing.Any] = {}
_cache_lock = Lock()


def generate_request_hash(endpoint: str, request_payload: dict) -> str:
    def normalize_value(value: typing.Any) -> typing.Any:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return json.dumps(parsed, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                return value
        elif isinstance(value, dict):
            return {k: normalize_value(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value

    normalized_payload = normalize_value(request_payload)
    cache_str = json.dumps(
        {"endpoint": endpoint, "params": normalized_payload},
        sort_keys=True,
    )
    return hashlib.sha256(cache_str.encode()).hexdigest()


def cached_gateway_call(func: typing.Callable[..., T]) -> typing.Callable[..., T]:
    @wraps(func)
    async def wrapper(request: BaseModel, *args: typing.Any, **kwargs: typing.Any) -> T:
        request_payload = request.model_dump(exclude_none=True, exclude=("run_id",))
        request_hash = generate_request_hash(func.__name__, request_payload)

        with _cache_lock:
            if request_hash in _cache:
                return _cache[request_hash]

        result = await func(request, *args, **kwargs)
        response_data = result.model_dump() if isinstance(result, BaseModel) else result

        with _cache_lock:
            _cache[request_hash] = response_data

        return result

    return wrapper
