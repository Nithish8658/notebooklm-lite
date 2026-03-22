import asyncio
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

from services.rerank_runtime import get_rerank_runtime
from services.runtime_config import Phase2RuntimeConfig, get_phase2_config

_LOG = logging.getLogger("phase2_runtime")


@dataclass
class _LoopState:
    cpu_executor: ThreadPoolExecutor
    cpu_semaphore: asyncio.Semaphore


class Phase2Runtime:
    def __init__(self, config: Optional[Phase2RuntimeConfig] = None):
        self.config = config or get_phase2_config()
        self._loop_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._registry_lock = threading.Lock()
        self._rerank_runtime = get_rerank_runtime()

    async def run_cpu_stage(self, stage_name: str, func: Callable[..., Any], *args, **kwargs):
        state = await self._get_loop_state()
        async with state.cpu_semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                state.cpu_executor,
                partial(func, *args, **kwargs),
            )

    async def rerank(self, **kwargs):
        return await self._rerank_runtime.rerank(**kwargs)

    async def verify_logic(self, query: str, cached_text: str) -> float:
        return await self._rerank_runtime.verify_logic(query, cached_text)

    async def warmup(self):
        await self._rerank_runtime.warmup()

    async def shutdown(self):
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.pop(loop, None)

        if state:
            state.cpu_executor.shutdown(wait=False, cancel_futures=False)

        await self._rerank_runtime.shutdown()

    async def _get_loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    cpu_executor=ThreadPoolExecutor(
                        max_workers=self.config.retrieval_cpu_workers,
                        thread_name_prefix="phase2-cpu",
                    ),
                    cpu_semaphore=asyncio.Semaphore(self.config.retrieval_cpu_parallelism),
                )
                self._loop_states[loop] = state
                _LOG.info(
                    "Phase 2 runtime initialized: retrieval_workers=%d, retrieval_parallelism=%d, rerank_pool=%d",
                    self.config.retrieval_cpu_workers,
                    self.config.retrieval_cpu_parallelism,
                    self.config.rerank_pool_size,
                )
        return state


_PHASE2_RUNTIME: Optional[Phase2Runtime] = None


def get_phase2_runtime() -> Phase2Runtime:
    global _PHASE2_RUNTIME
    if _PHASE2_RUNTIME is None:
        _PHASE2_RUNTIME = Phase2Runtime()
    return _PHASE2_RUNTIME
