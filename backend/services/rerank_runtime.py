import asyncio
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

from rerank import OnnxCrossEncoder, ScoreCache, create_reranker, rerank_with_cross_encoder
from services.runtime_config import Phase2RuntimeConfig, get_phase2_config

_LOG = logging.getLogger("rerank_runtime")


@dataclass
class _LoopState:
    executor: ThreadPoolExecutor
    score_cache: ScoreCache
    session_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    verify_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    models: List[OnnxCrossEncoder] = field(default_factory=list)
    verify_task: Optional[asyncio.Task] = None
    init_task: Optional[asyncio.Task] = None


class RerankRuntime:
    def __init__(self, config: Optional[Phase2RuntimeConfig] = None):
        self.config = config or get_phase2_config()
        self._loop_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._registry_lock = threading.Lock()

    async def warmup(self):
        state = await self._get_loop_state()
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                state.executor,
                partial(state.models[idx].predict, [("warmup query", "warmup context")] * 2),
            )
            for idx in range(len(state.models))
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self):
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.pop(loop, None)

        if not state:
            return

        if state.verify_task:
            state.verify_task.cancel()
            try:
                await state.verify_task
            except asyncio.CancelledError:
                pass

        state.executor.shutdown(wait=False, cancel_futures=False)

    async def rerank(
        self,
        *,
        query: str,
        candidates: List[Dict],
        alternative_queries: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        use_fusion: bool = True,
    ) -> List[Dict]:
        if not candidates:
            return []

        state = await self._get_loop_state()
        session_id = await state.session_queue.get()
        try:
            loop = asyncio.get_running_loop()
            job = partial(
                rerank_with_cross_encoder,
                query=query,
                candidates=candidates,
                alternative_queries=alternative_queries,
                batch_size=self.config.rerank_batch_size,
                top_k=top_k,
                use_fusion=use_fusion,
                model=state.models[session_id],
                score_cache=state.score_cache,
            )
            return await loop.run_in_executor(state.executor, job)
        finally:
            await state.session_queue.put(session_id)

    async def verify_logic(self, query: str, cached_text: str) -> float:
        state = await self._get_loop_state()
        future = asyncio.get_running_loop().create_future()
        await state.verify_queue.put(((query, cached_text), future))
        return await future

    async def _get_loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    executor=ThreadPoolExecutor(
                        max_workers=self.config.rerank_executor_workers,
                        thread_name_prefix="rerank-cpu",
                    ),
                    score_cache=ScoreCache(maxsize=self.config.rerank_score_cache_size),
                )
                state.init_task = loop.create_task(self._initialize_loop_state(state))
                self._loop_states[loop] = state

        if state.init_task is not None:
            await state.init_task

        if state.verify_task is None or state.verify_task.done():
            state.verify_task = asyncio.create_task(self._verify_loop(state))

        return state

    async def _initialize_loop_state(self, state: _LoopState):
        models = await asyncio.to_thread(self._build_models)
        state.models.extend(models)
        for index in range(len(models)):
            await state.session_queue.put(index)
        state.init_task = None
        _LOG.info("Rerank runtime initialized with %d pooled sessions.", len(models))

    def _build_models(self) -> List[OnnxCrossEncoder]:
        return [
            create_reranker(
                num_threads=self.config.rerank_threads_per_session,
                num_streams=self.config.rerank_openvino_streams,
            )
            for _ in range(self.config.rerank_pool_size)
        ]

    async def _verify_loop(self, state: _LoopState):
        try:
            while True:
                pair, future = await state.verify_queue.get()
                batch = [(pair, future)]
                deadline = asyncio.get_running_loop().time() + (self.config.rerank_batch_window_ms / 1000.0)

                while len(batch) < self.config.rerank_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        pair, future = await asyncio.wait_for(state.verify_queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append((pair, future))

                try:
                    session_id = await state.session_queue.get()
                    try:
                        loop = asyncio.get_running_loop()
                        pairs = [item[0] for item in batch]
                        scores = await loop.run_in_executor(
                            state.executor,
                            partial(
                                state.models[session_id].predict,
                                pairs,
                                batch_size=self.config.rerank_batch_size,
                            ),
                        )
                    finally:
                        await state.session_queue.put(session_id)

                    for idx, score in enumerate(scores):
                        if not batch[idx][1].done():
                            batch[idx][1].set_result(float(score))

                    if len(batch) > 1:
                        _LOG.info("BATCH VERIFY: Validated %d cache pairs in one pooled inference.", len(batch))
                except Exception as exc:
                    _LOG.error("BATCH VERIFY ERROR: %s", exc)
                    for _, item_future in batch:
                        if not item_future.done():
                            item_future.set_exception(exc)
                finally:
                    for _ in range(len(batch)):
                        state.verify_queue.task_done()
        except asyncio.CancelledError:
            raise


_RERANK_RUNTIME: Optional[RerankRuntime] = None


def get_rerank_runtime() -> RerankRuntime:
    global _RERANK_RUNTIME
    if _RERANK_RUNTIME is None:
        _RERANK_RUNTIME = RerankRuntime()
    return _RERANK_RUNTIME
