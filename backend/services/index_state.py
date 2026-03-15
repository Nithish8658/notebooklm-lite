
import asyncio
import time
from typing import Dict, Optional, Any
from collections import OrderedDict
import logging

from sqlalchemy import select
from database import AsyncSessionLocal, Batch
from retrieval.bm25_index import BM25ChunkIndex

_LOG = logging.getLogger("index_state")

class IndexStateManager:
    """
    AC-21: Intelligent Metadata Singleton (IMS).
    Manages in-memory batch brains (BM25 + Graph) with LRU eviction.
    Reduces Phase 1 latency from 7s to <1ms.
    """
    def __init__(self, max_batches: int = 50):
        # Stores {batch_id: {"bm25": BM25ChunkIndex, "graph": dict, "loaded_at": float}}
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_batches = max_batches
        self._lock = asyncio.Lock()

    async def get_brain(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the brain from RAM. Loads from DB on miss.
        Thread-safe to prevent multiple users from loading the same batch at once.
        """
        # 1. Fast path: already in RAM
        if batch_id in self._cache:
            self._cache.move_to_end(batch_id) # Update LRU status
            return self._cache[batch_id]

        # 2. Slow path: load from DB (locked to prevent redundant work)
        async with self._lock:
            # Re-check in case another request loaded it while we were waiting for the lock
            if batch_id in self._cache:
                return self._cache[batch_id]

            _LOG.info("IMS MISS: Loading batch '%s' into RAM singleton.", batch_id)
            try:
                async with AsyncSessionLocal() as db:
                    result = await db.execute(select(Batch).where(Batch.id == batch_id))
                    batch = result.scalars().first()
                    
                    if not batch or (not batch.bm25_data and not batch.graph_data):
                        _LOG.warning("IMS ALERT: No metadata found in DB for batch '%s'", batch_id)
                        return None

                    # Instantiate objects (CPU intensive, but done only once)
                    bm25 = BM25ChunkIndex.from_dict(batch.bm25_data) if batch.bm25_data else None
                    graph = batch.graph_data or {}
                    
                    brain = {
                        "bm25": bm25,
                        "graph": graph,
                        "loaded_at": time.time()
                    }

                    # 3. Manage Cache Size (LRU Eviction)
                    if len(self._cache) >= self._max_batches:
                        evicted_id, _ = self._cache.popitem(last=False)
                        _LOG.info("IMS EVICT: Removing batch '%s' from RAM to save memory.", evicted_id)

                    self._cache[batch_id] = brain
                    return brain

            except Exception as e:
                _LOG.error("IMS CRITICAL: Failed to load batch '%s' metadata: %s", batch_id, e)
                return None

    async def invalidate_batch(self, batch_id: str):
        """
        Clears a batch from RAM. Called when new documents are uploaded.
        """
        async with self._lock:
            if batch_id in self._cache:
                _LOG.info("IMS FLUSH: Invaliding batch '%s' due to data update.", batch_id)
                del self._cache[batch_id]

# Singleton instance
index_manager = IndexStateManager(max_batches=50)
