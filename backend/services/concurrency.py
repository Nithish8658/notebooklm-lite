
import asyncio
import time
from typing import List, Tuple
import logging

_LOG = logging.getLogger("reranker_batch")

class BatchRerankManager:
    """
    AC-25: Batch Cross-Encoder for Cache Verification.
    Groups concurrent cache-hit validations into a single model call.
    Fixes the 7s bottleneck in Semantic Cache Lookups.
    """
    def __init__(self, reranker_getter, batch_size: int = 16, wait_time_ms: int = 15):
        self.reranker_getter = reranker_getter
        self.batch_size = batch_size
        self.wait_time_ms = wait_time_ms / 1000.0
        self.queue = asyncio.Queue()
        self._loop_task = None
        self._lock = asyncio.Lock()

    async def verify_logic(self, query: str, cached_text: str) -> float:
        """
        Public API to verify if two strings are logically equivalent.
        Returns the Cross-Encoder score.
        """
        if self._loop_task is None:
            async with self._lock:
                if self._loop_task is None:
                    self._loop_task = asyncio.create_task(self._batch_loop())

        future = asyncio.get_running_loop().create_future()
        await self.queue.put(((query, cached_text), future))
        return await future

    async def _batch_loop(self):
        _LOG.info("Rerank Batch Loop Started.")
        while True:
            pair, future = await self.queue.get()
            batch = [(pair, future)]

            start_wait = time.time()
            while len(batch) < self.batch_size and (time.time() - start_wait) < self.wait_time_ms:
                try:
                    pair, future = self.queue.get_nowait()
                    batch.append((pair, future))
                except asyncio.QueueEmpty:
                    break
            
            try:
                pairs = [item[0] for item in batch]
                futures = [item[1] for item in batch]
                
                t_start = time.perf_counter()
                model = self.reranker_getter()
                
                # predict handles list of pairs internally
                scores = await asyncio.to_thread(model.predict, pairs)
                
                duration = time.perf_counter() - t_start
                if len(batch) > 1:
                    _LOG.info("BATCH VERIFY: Validated %d cache pairs in %.4fs", len(batch), duration)

                for i, score in enumerate(scores):
                    if not futures[i].done():
                        futures[i].set_result(float(score))

            except Exception as e:
                _LOG.error("BATCH VERIFY ERROR: %s", e)
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                for _ in range(len(batch)): 
                    self.queue.task_done()

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

# Connection Guard for PostgreSQL (Dynamic Proxy)
db_semaphore = SemaphoreProxy()
