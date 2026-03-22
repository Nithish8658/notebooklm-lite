import asyncio
import time
import json
from typing import Dict, Optional, Any
from collections import OrderedDict
import logging
from enum import Enum
from contextlib import asynccontextmanager

from sqlalchemy import select
from database import AsyncSessionLocal, Batch
from retrieval.bm25_index import BM25ChunkIndex
from services.concurrency import db_metadata_semaphore

_LOG = logging.getLogger("index_state")

class ResourceState(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    EVICTING = "evicting"

class BrainHandle:
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.brain: Optional[Dict[str, Any]] = None
        self.ref_count = 0
        self.last_access = time.time()
        self.size_bytes = 0
        self.state = ResourceState.LOADING
        self.error: Optional[str] = None

class IndexStateManager:
    def __init__(self, max_memory_mb: int = 2048):
        self._cache: OrderedDict[str, BrainHandle] = OrderedDict()
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory_bytes = 0
        self._global_lock = asyncio.Lock()
        self._loading_futures: Dict[str, asyncio.Future] = {}
        self._eviction_task: Optional[asyncio.Task] = None

    def _get_handle(self, batch_id: str) -> Optional[BrainHandle]:
        if batch_id in self._cache:
            handle = self._cache[batch_id]
            self._cache.move_to_end(batch_id)
            return handle
        return None

    async def _estimate_size(self, bm25_dict: Optional[Dict], graph_dict: Optional[Dict]) -> int:
        """Offloads heavy JSON serialization to a thread to avoid blocking the event loop."""
        def calc():
            size = 0
            if bm25_dict: size += len(json.dumps(bm25_dict))
            if graph_dict: size += len(json.dumps(graph_dict))
            return int(size * 1.5) # 1.5x buffer for Python object overhead
        return await asyncio.to_thread(calc)

    async def _load_resource(self, batch_id: str) -> BrainHandle:
        """
        Decoupled Single-Flight Loader.
        The future lifecycle is tied to the loading task, not the requester's cancellation.
        """
        async with self._global_lock:
            if batch_id in self._loading_futures:
                future = self._loading_futures[batch_id]
            else:
                future = asyncio.Future()
                self._loading_futures[batch_id] = future
                # Detached task ensures the build finishes even if this request is cancelled
                asyncio.create_task(self._loading_task_wrapper(batch_id, future))
        
        return await future

    async def _loading_task_wrapper(self, batch_id: str, future: asyncio.Future):
        """Manages the loading future and ensures cleanup in the tracker."""
        try:
            handle = await self._perform_build(batch_id)
            if not future.done():
                future.set_result(handle)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            async with self._global_lock:
                # Only delete if this is still the active future for this batch
                if self._loading_futures.get(batch_id) == future:
                    del self._loading_futures[batch_id]

    async def _perform_build(self, batch_id: str) -> BrainHandle:
        _LOG.info("IMS MISS: Loading batch '%s' into RAM.", batch_id)
        handle = BrainHandle(batch_id)
        
        try:
            async with asyncio.timeout(60):
                async with db_metadata_semaphore:
                    async with AsyncSessionLocal() as db:
                        result = await db.execute(select(Batch).where(Batch.id == batch_id))
                        batch = result.scalars().first()
                
                if not batch or (not batch.bm25_data and not batch.graph_data):
                    handle.state = ResourceState.FAILED
                    handle.error = "No metadata found in DB"
                    return handle

                # Non-blocking size estimation
                handle.size_bytes = await self._estimate_size(batch.bm25_data, batch.graph_data)

                # CPU Intensive instantiation
                bm25 = await asyncio.to_thread(BM25ChunkIndex.from_dict, batch.bm25_data) if batch.bm25_data else None
                graph = batch.graph_data or {}
                
                handle.brain = {"bm25": bm25, "graph": graph}
                handle.state = ResourceState.READY
                
                async with self._global_lock:
                    self._cache[batch_id] = handle
                    self._current_memory_bytes += handle.size_bytes

                return handle

        except Exception as e:
            handle.state = ResourceState.FAILED
            handle.error = str(e)
            _LOG.error("IMS CRITICAL: Failed to load batch '%s': %s", batch_id, e)
            return handle

    async def _eviction_worker(self):
        _LOG.info("IMS EVICTOR: Background worker active.")
        while True:
            try:
                if self._current_memory_bytes > self._max_memory_bytes:
                    async with self._global_lock:
                        candidates = [
                            bid for bid, h in self._cache.items()
                            if h.ref_count == 0 and h.state == ResourceState.READY
                        ]
                        if candidates:
                            evict_id = candidates[0]
                            handle = self._cache[evict_id]
                            _LOG.info("IMS EVICT: Removing batch '%s' (%d MB reclaimed).", 
                                      evict_id, handle.size_bytes // 1024 // 1024)
                            handle.state = ResourceState.EVICTING
                            del self._cache[evict_id]
                            self._current_memory_bytes -= handle.size_bytes
                            handle.brain = None 
                await asyncio.sleep(10)
            except asyncio.CancelledError: break
            except Exception as e:
                _LOG.error("IMS EVICTOR: Loop error: %s", e)
                await asyncio.sleep(10)

    def start_evictor(self):
        if not self._eviction_task:
            self._eviction_task = asyncio.create_task(self._eviction_worker())

    async def stop_evictor(self):
        if self._eviction_task:
            self._eviction_task.cancel()
            try: await self._eviction_task
            except asyncio.CancelledError: pass
            self._eviction_task = None

    @asynccontextmanager
    async def acquire(self, batch_id: str):
        handle = self._get_handle(batch_id)
        if not handle or handle.state != ResourceState.READY:
            handle = await self._load_resource(batch_id)
        
        if handle.state != ResourceState.READY:
            _LOG.error("IMS ACQUIRE FAIL: Batch '%s' state is %s", batch_id, handle.state.value)
            yield None
            return

        async with self._global_lock:
            handle.ref_count += 1
            handle.last_access = time.time()
        
        try:
            yield handle.brain
        finally:
            async with self._global_lock:
                handle.ref_count -= 1

    async def invalidate_batch(self, batch_id: str):
        async with self._global_lock:
            if batch_id in self._cache:
                handle = self._cache[batch_id]
                self._current_memory_bytes -= handle.size_bytes
                handle.state = ResourceState.EVICTING
                del self._cache[batch_id]
                handle.brain = None

    async def clear_all(self):
        async with self._global_lock:
            for handle in self._cache.values():
                handle.state = ResourceState.EVICTING
                handle.brain = None
            self._cache.clear()
            self._current_memory_bytes = 0

index_manager = IndexStateManager(max_memory_mb=2048)
