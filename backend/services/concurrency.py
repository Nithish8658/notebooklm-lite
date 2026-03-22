import asyncio
from typing import Dict
import logging

_LOG = logging.getLogger("reranker_batch")

class BatchRerankManager:
    """
    AC-25: Batch Cross-Encoder for Cache Verification.
    Delegates verification to the dedicated Phase 2 rerank runtime.
    """
    def __init__(self, reranker_getter=None, batch_size: int = 16, wait_time_ms: int = 15):
        from services.phase2_runtime import get_phase2_runtime

        self._runtime = get_phase2_runtime()

    async def verify_logic(self, query: str, cached_text: str) -> float:
        """
        Public API to verify if two strings are logically equivalent.
        Returns the Cross-Encoder score.
        """
        return await self._runtime.verify_logic(query, cached_text)

from infrastructure import get_infra

class SemaphoreProxy:
    """
    Acts as a proxy for the db_semaphore.
    Delegates to the infrastructure container assigned to the current event loop.
    """
    async def __aenter__(self):
        return await get_infra().db_semaphore.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await get_infra().db_semaphore.__aexit__(exc_type, exc_val, exc_tb)

class MetadataSemaphoreProxy:
    async def __aenter__(self):
        return await get_infra().db_metadata_semaphore.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await get_infra().db_metadata_semaphore.__aexit__(exc_type, exc_val, exc_tb)

class SingleFlightManager:
    """
    Prevents "Cache Stampedes" by coalescing identical concurrent requests
    into a single execution.
    """
    def __init__(self):
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, coro_fn, *args, **kwargs):
        """
        Executes coro_fn and returns the result. If another call with the same
        key is already in flight, it waits for the first one and returns its result.
        """
        leader = False
        async with self._lock:
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
                leader = True

        if not leader:
            return await asyncio.shield(future)

        try:
            result = await coro_fn(*args, **kwargs)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
                # Mark as retrieved to avoid "Future exception was never retrieved" 
                # if there are no followers awaiting this future.
                try:
                    future.exception()
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    pass
            raise e
        finally:
            async with self._lock:
                if self._inflight.get(key) == future:
                    del self._inflight[key]

# Singletons
db_semaphore = SemaphoreProxy()
db_metadata_semaphore = MetadataSemaphoreProxy()
single_flight = SingleFlightManager()
