import asyncio
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

from services.embeddings import get_embedding_model, load_embedding_model
from services.runtime_config import Phase2RuntimeConfig, get_phase2_config

_LOG = logging.getLogger("embedding_runtime")


@dataclass
class EmbeddingRequestContext:
    query_vectors: Dict[str, List[float]] = field(default_factory=dict)

    @staticmethod
    def _key(text: str) -> str:
        return str(text or "").strip()

    def get_query_vector(self, text: str) -> Optional[List[float]]:
        return self.query_vectors.get(self._key(text))

    def store_query_vector(self, text: str, vector: List[float]) -> List[float]:
        vector_list = list(vector)
        self.query_vectors[self._key(text)] = vector_list
        return vector_list


@dataclass
class _EmbeddingJob:
    text: str
    future: asyncio.Future


@dataclass
class _LoopState:
    executor: ThreadPoolExecutor
    online_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    background_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    batch_task: Optional[asyncio.Task] = None
    init_task: Optional[asyncio.Task] = None


class EmbeddingRuntime:
    def __init__(self, config: Optional[Phase2RuntimeConfig] = None):
        self.config = config or get_phase2_config()
        self._loop_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._registry_lock = threading.Lock()

    async def warmup(self):
        await self.get_embeddings(["Embedding runtime warmup"], priority="background")

    async def shutdown(self):
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.pop(loop, None)

        if not state:
            return

        if state.batch_task:
            state.batch_task.cancel()
            try:
                await state.batch_task
            except asyncio.CancelledError:
                pass

        state.executor.shutdown(wait=False, cancel_futures=False)

    async def get_embedding(
        self,
        text: str,
        *,
        request_context: Optional[EmbeddingRequestContext] = None,
        priority: str = "online",
    ) -> List[float]:
        results = await self.get_embeddings(
            [text],
            request_context=request_context,
            priority=priority,
        )
        return results[0] if results else []

    async def get_embeddings(
        self,
        texts: List[str],
        *,
        request_context: Optional[EmbeddingRequestContext] = None,
        priority: str = "online",
    ) -> List[List[float]]:
        if not texts:
            return []

        normalized_texts = [str(text or "").strip() for text in texts]
        results: List[Optional[List[float]]] = [None] * len(normalized_texts)
        missing_positions: Dict[str, List[int]] = {}
        ordered_missing: List[str] = []

        for idx, text in enumerate(normalized_texts):
            cached = request_context.get_query_vector(text) if request_context else None
            if cached is not None:
                results[idx] = cached
                continue

            if text not in missing_positions:
                missing_positions[text] = []
                ordered_missing.append(text)
            missing_positions[text].append(idx)

        if ordered_missing:
            state = await self._get_loop_state()
            queue = state.background_queue if priority == "background" else state.online_queue
            loop = asyncio.get_running_loop()
            futures = []

            for text in ordered_missing:
                future = loop.create_future()
                await queue.put(_EmbeddingJob(text=text, future=future))
                futures.append(future)

            vectors = await asyncio.gather(*futures)
            for text, vector in zip(ordered_missing, vectors):
                vector_list = request_context.store_query_vector(text, vector) if request_context else list(vector)
                for idx in missing_positions[text]:
                    results[idx] = vector_list

        return [vector or [] for vector in results]

    async def _get_loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    executor=ThreadPoolExecutor(
                        max_workers=self.config.embedding_executor_workers,
                        thread_name_prefix="embed-cpu",
                    ),
                )
                state.init_task = loop.create_task(self._initialize_loop_state(state))
                self._loop_states[loop] = state

        if state.init_task is not None:
            await state.init_task

        if state.batch_task is None or state.batch_task.done():
            state.batch_task = asyncio.create_task(self._batch_loop(state))

        return state

    async def _initialize_loop_state(self, state: _LoopState):
        await asyncio.to_thread(load_embedding_model, num_threads=self.config.embedding_threads)
        state.init_task = None
        _LOG.info(
            "Embedding runtime initialized: threads=%d executor_workers=%d batch_size=%d",
            self.config.embedding_threads,
            self.config.embedding_executor_workers,
            self.config.embedding_batch_size,
        )

    def _encode_queries(self, texts: List[str]):
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=self.config.embedding_encode_batch_size,
            is_query=True,
        )
        if len(texts) == 1 and getattr(embeddings, "ndim", 0) == 1:
            return [embeddings]
        return embeddings

    async def _next_job(self, state: _LoopState) -> tuple[str, asyncio.Queue, _EmbeddingJob]:
        if not state.online_queue.empty():
            return "online", state.online_queue, await state.online_queue.get()
        if not state.background_queue.empty():
            return "background", state.background_queue, await state.background_queue.get()

        online_get = asyncio.create_task(state.online_queue.get())
        background_get = asyncio.create_task(state.background_queue.get())

        done, pending = await asyncio.wait(
            {online_get, background_get},
            return_when=asyncio.FIRST_COMPLETED,
        )

        try:
            if online_get in done:
                job = online_get.result()
                if background_get in done:
                    await state.background_queue.put(background_get.result())
                return "online", state.online_queue, job

            return "background", state.background_queue, background_get.result()
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _batch_loop(self, state: _LoopState):
        try:
            while True:
                queue_name, queue, job = await self._next_job(state)
                batch = [job]
                deadline = asyncio.get_running_loop().time() + (self.config.embedding_batch_window_ms / 1000.0)

                while len(batch) < self.config.embedding_batch_size:
                    if queue_name == "background" and not state.online_queue.empty():
                        break

                    try:
                        batch.append(queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        pass

                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break

                    try:
                        batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                try:
                    texts = [item.text for item in batch]
                    loop = asyncio.get_running_loop()
                    embeddings = await loop.run_in_executor(
                        state.executor,
                        partial(self._encode_queries, texts),
                    )

                    for idx, emb in enumerate(embeddings):
                        vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                        if not batch[idx].future.done():
                            batch[idx].future.set_result(vector)

                    if len(batch) > 1:
                        _LOG.info(
                            "Embedding runtime batched %d %s requests in one inference.",
                            len(batch),
                            queue_name,
                        )
                except Exception as exc:
                    _LOG.error("Embedding batch error: %s", exc)
                    for item in batch:
                        if not item.future.done():
                            item.future.set_exception(exc)
                finally:
                    for _ in range(len(batch)):
                        queue.task_done()
        except asyncio.CancelledError:
            raise


_EMBEDDING_RUNTIME: Optional[EmbeddingRuntime] = None


def get_embedding_runtime() -> EmbeddingRuntime:
    global _EMBEDDING_RUNTIME
    if _EMBEDDING_RUNTIME is None:
        _EMBEDDING_RUNTIME = EmbeddingRuntime()
    return _EMBEDDING_RUNTIME
