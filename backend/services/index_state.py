
import asyncio
import time
from typing import Dict, Optional, Any
from collections import OrderedDict
import logging

from sqlalchemy import select
from database import AsyncSessionLocal, Cohort
from retrieval.bm25_index import BM25ChunkIndex

_LOG = logging.getLogger("index_state")

class IndexStateManager:
    """
    AC-21: Intelligent Metadata Singleton (IMS).
    Manages in-memory cohort brains (BM25 + Graph) with LRU eviction.
    Reduces Phase 1 latency from 7s to <1ms.
    """
    def __init__(self, max_cohorts: int = 50):
        # Stores {cohort_id: {"bm25": BM25ChunkIndex, "graph": dict, "loaded_at": float}}
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_cohorts = max_cohorts
        self._lock = asyncio.Lock()

    async def get_brain(self, cohort_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the brain from RAM. Loads from DB on miss.
        Thread-safe to prevent multiple users from loading the same cohort at once.
        """
        # 1. Fast path: already in RAM
        if cohort_id in self._cache:
            self._cache.move_to_end(cohort_id) # Update LRU status
            return self._cache[cohort_id]

        # 2. Slow path: load from DB (locked to prevent redundant work)
        async with self._lock:
            # Re-check in case another request loaded it while we were waiting for the lock
            if cohort_id in self._cache:
                return self._cache[cohort_id]

            _LOG.info("IMS MISS: Loading cohort '%s' into RAM singleton.", cohort_id)
            try:
                async with AsyncSessionLocal() as db:
                    result = await db.execute(select(Cohort).where(Cohort.id == cohort_id))
                    cohort = result.scalars().first()
                    
                    if not cohort or (not cohort.bm25_data and not cohort.graph_data):
                        _LOG.warning("IMS ALERT: No metadata found in DB for cohort '%s'", cohort_id)
                        return None

                    # Instantiate objects (CPU intensive, but done only once)
                    bm25 = BM25ChunkIndex.from_dict(cohort.bm25_data) if cohort.bm25_data else None
                    graph = cohort.graph_data or {}
                    
                    brain = {
                        "bm25": bm25,
                        "graph": graph,
                        "loaded_at": time.time()
                    }

                    # 3. Manage Cache Size (LRU Eviction)
                    if len(self._cache) >= self._max_cohorts:
                        evicted_id, _ = self._cache.popitem(last=False)
                        _LOG.info("IMS EVICT: Removing cohort '%s' from RAM to save memory.", evicted_id)

                    self._cache[cohort_id] = brain
                    return brain

            except Exception as e:
                _LOG.error("IMS CRITICAL: Failed to load cohort '%s' metadata: %s", cohort_id, e)
                return None

    async def invalidate_cohort(self, cohort_id: str):
        """
        Clears a cohort from RAM. Called when new documents are uploaded.
        """
        async with self._lock:
            if cohort_id in self._cache:
                _LOG.info("IMS FLUSH: Invaliding cohort '%s' due to data update.", cohort_id)
                del self._cache[cohort_id]

# Singleton instance
index_manager = IndexStateManager(max_cohorts=50)
