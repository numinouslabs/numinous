import asyncio
from typing import Any, Dict

from neurons.validator.utils.if_metagraph import IfMetagraph


class TestIfMetagraph:
    async def test_sync_throttles_and_sets_lite(self, monkeypatch):
        # Create instance without running base __init__
        obj = IfMetagraph.__new__(IfMetagraph)

        obj._last_sync_time = 0.0
        obj._sync_lock = asyncio.Lock()

        call_count = {"n": 0}
        last_kwargs: Dict[str, Any] = {}

        async def fake_super_sync(self, *args, **kwargs):
            call_count["n"] += 1
            last_kwargs.clear()
            last_kwargs.update(kwargs)

        # Patch parent sync to fake_super_sync
        from bittensor.core.metagraph import AsyncMetagraph as _AsyncMetagraph

        monkeypatch.setattr(_AsyncMetagraph, "sync", fake_super_sync, raising=True)

        current = {"t": 1_000.0}

        def fake_time():
            return current["t"]

        monkeypatch.setattr(
            "neurons.validator.utils.if_metagraph.time.time", fake_time, raising=True
        )

        # First call triggers sync
        await obj.sync()
        assert call_count["n"] == 1
        assert last_kwargs.get("lite") is True

        # Second call within a minute does nothing
        await obj.sync()
        assert call_count["n"] == 1

        # Advance time beyond Xs, should trigger again
        current["t"] = 1_031.0
        await obj.sync()

        # Assert that sync was called again
        assert call_count["n"] == 2
        assert last_kwargs.get("lite") is True

    async def test_sync_concurrent_calls_single_super_sync(self, monkeypatch):
        obj = IfMetagraph.__new__(IfMetagraph)
        obj._last_sync_time = 0.0
        obj._sync_lock = asyncio.Lock()

        call_count = {"n": 0}

        async def fake_super_sync(self, *args, **kwargs):
            await asyncio.sleep(0.01)

            call_count["n"] += 1

        from bittensor.core.metagraph import AsyncMetagraph as _AsyncMetagraph

        monkeypatch.setattr(_AsyncMetagraph, "sync", fake_super_sync, raising=True)

        # Keep time constant
        monkeypatch.setattr(
            "neurons.validator.utils.if_metagraph.time.time", lambda: 2_000.0, raising=True
        )

        await asyncio.gather(*(obj.sync() for _ in range(5)))

        # Assert that only one call to super sync was made
        assert call_count["n"] == 1
