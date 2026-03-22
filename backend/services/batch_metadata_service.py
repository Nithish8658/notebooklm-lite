import asyncio
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple

from sqlalchemy import delete, select

from database import AsyncSessionLocal, Batch, UserEnrollment
from graph.chunk_graph import build_chunk_graph
from retrieval.bm25_index import BM25ChunkIndex
from services.concurrency import SingleFlightManager, db_metadata_semaphore
from services.index_state import index_manager
from services.runtime_config import Phase2RuntimeConfig, get_phase2_config
from services.search_repository import SearchRepository

_LOG = logging.getLogger("batch_metadata_service")


@dataclass
class _LoopState:
    executor: ThreadPoolExecutor
    semaphore: asyncio.Semaphore


class BatchMetadataService:
    def __init__(self, repository: Optional[SearchRepository] = None, config: Optional[Phase2RuntimeConfig] = None):
        self.repository = repository or SearchRepository("document_chunks")
        self.config = config or get_phase2_config()
        self._registry_lock = threading.Lock()
        self._loop_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._single_flight = SingleFlightManager()

    async def shutdown(self):
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.pop(loop, None)

        if state:
            state.executor.shutdown(wait=False, cancel_futures=False)

    async def ensure_batch_metadata(self, batch_id: str) -> bool:
        if await self._batch_metadata_ready(batch_id):
            return True
        return await self.rebuild_batch_metadata(batch_id, purge_orphaned=True)

    async def rebuild_batch_metadata(
        self,
        batch_id: str,
        *,
        batch_name: Optional[str] = None,
        force: bool = True,
        purge_orphaned: bool = False,
    ) -> bool:
        if not force and await self._batch_metadata_ready(batch_id):
            return True

        return await self._single_flight.do(
            f"metadata:{batch_id}",
            self._rebuild_internal,
            batch_id,
            batch_name,
            purge_orphaned,
        )

    async def clear_batch_metadata(
        self,
        batch_id: str,
        *,
        integrity_status: str = "REPAIR_REQUIRED",
        delete_batch: bool = False,
        clear_enrollments: bool = False,
    ):
        async with db_metadata_semaphore:
            async with AsyncSessionLocal() as db:
                if delete_batch:
                    await db.execute(delete(Batch).where(Batch.id == batch_id))
                else:
                    result = await db.execute(select(Batch).where(Batch.id == batch_id))
                    batch = result.scalars().first()
                    if batch:
                        batch.bm25_data = None
                        batch.graph_data = None
                        batch.integrity_status = integrity_status

                if clear_enrollments:
                    await db.execute(delete(UserEnrollment).where(UserEnrollment.batch_id == batch_id))

                await db.commit()

        await index_manager.invalidate_batch(batch_id)

    async def purge_orphaned_batch(self, batch_id: str):
        await self.clear_batch_metadata(
            batch_id,
            delete_batch=True,
            clear_enrollments=True,
        )

        from services.platform_sync import PlatformSyncService

        sync_service = PlatformSyncService()
        sync_service.remove_all_files_for_batch(batch_id)

    async def _batch_metadata_ready(self, batch_id: str) -> bool:
        async with db_metadata_semaphore:
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Batch).where(Batch.id == batch_id))
                batch = result.scalars().first()

        return bool(batch and batch.bm25_data and batch.graph_data)

    async def _rebuild_internal(self, batch_id: str, batch_name: Optional[str], purge_orphaned: bool) -> bool:
        chunks = await self.repository.fetch_batch_chunks(
            batch_id,
            page_size=self.config.metadata_scroll_page_size,
        )

        if not chunks:
            _LOG.warning("No indexed chunks found for batch %s during metadata rebuild.", batch_id)
            if purge_orphaned:
                await self.purge_orphaned_batch(batch_id)
            else:
                await self.clear_batch_metadata(batch_id)
            return False

        bm25_dict, graph_dict = await self._build_metadata_structures(chunks)
        await self._persist_metadata(batch_id, bm25_dict, graph_dict, batch_name=batch_name)
        return True

    async def _build_metadata_structures(self, chunks: list[Dict]) -> Tuple[Dict, Dict]:
        state = await self._get_loop_state()
        async with state.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                state.executor,
                partial(self._build_metadata_sync, chunks),
            )

    @staticmethod
    def _build_metadata_sync(chunks: list[Dict]) -> Tuple[Dict, Dict]:
        bm25_index = BM25ChunkIndex(chunks)
        graph = build_chunk_graph(chunks)
        return bm25_index.to_dict(), graph

    async def _persist_metadata(
        self,
        batch_id: str,
        bm25_dict: Dict,
        graph_dict: Dict,
        *,
        batch_name: Optional[str] = None,
    ):
        async with db_metadata_semaphore:
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Batch).where(Batch.id == batch_id))
                batch = result.scalars().first()
                if not batch:
                    batch = Batch(id=batch_id, name=batch_name or f"Course {batch_id}")
                    db.add(batch)
                elif batch_name and batch.name != batch_name:
                    batch.name = batch_name

                batch.bm25_data = bm25_dict
                batch.graph_data = graph_dict
                batch.integrity_status = "HEALTHY"
                await db.commit()

        await index_manager.invalidate_batch(batch_id)
        _LOG.info("Batch metadata rebuilt and persisted for %s.", batch_id)

    async def _get_loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    executor=ThreadPoolExecutor(
                        max_workers=self.config.metadata_build_workers,
                        thread_name_prefix="metadata-build",
                    ),
                    semaphore=asyncio.Semaphore(
                        max(1, min(self.config.metadata_build_parallelism, self.config.metadata_build_workers))
                    ),
                )
                self._loop_states[loop] = state
                _LOG.info(
                    "Batch metadata service initialized: workers=%d parallelism=%d page_size=%d",
                    self.config.metadata_build_workers,
                    max(1, min(self.config.metadata_build_parallelism, self.config.metadata_build_workers)),
                    self.config.metadata_scroll_page_size,
                )
        return state


_BATCH_METADATA_SERVICE: Optional[BatchMetadataService] = None


def get_batch_metadata_service() -> BatchMetadataService:
    global _BATCH_METADATA_SERVICE
    if _BATCH_METADATA_SERVICE is None:
        _BATCH_METADATA_SERVICE = BatchMetadataService()
    return _BATCH_METADATA_SERVICE
