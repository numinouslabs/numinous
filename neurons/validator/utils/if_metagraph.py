import asyncio
import time
from typing import Any

from bittensor.core.metagraph import AsyncMetagraph


class IfMetagraph(AsyncMetagraph):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._last_sync_time: float = 0.0
        self._sync_lock = asyncio.Lock()

    async def sync(self) -> None:
        sync_throttle_seconds = 30

        now = time.time()

        if now - self._last_sync_time < sync_throttle_seconds:
            return

        async with self._sync_lock:
            now = time.time()

            if now - self._last_sync_time < sync_throttle_seconds:
                return

            await super().sync(lite=True)

            self._last_sync_time = time.time()
