import asyncio
import logging
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import HTTPException

from metrics import log_metric
from services.runtime_config import Phase2RuntimeConfig, get_phase2_config

_LOG = logging.getLogger("tenant_runtime")


@dataclass
class _TenantState:
    online_semaphore: asyncio.Semaphore
    background_semaphore: asyncio.Semaphore
    online_active: int = 0
    background_active: int = 0
    last_used: float = field(default_factory=time.monotonic)


@dataclass
class _LoopState:
    online_semaphore: asyncio.Semaphore
    background_semaphore: asyncio.Semaphore
    tenant_states: Dict[str, _TenantState] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class TenantAdmissionRuntime:
    def __init__(self, config: Optional[Phase2RuntimeConfig] = None):
        self.config = config or get_phase2_config()
        self._loop_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._registry_lock = threading.Lock()

    @asynccontextmanager
    async def acquire(
        self,
        tenant_id: str,
        *,
        lane: str = "online",
        feature: str = "request",
        http_on_timeout: bool = False,
    ):
        tenant_key = str(tenant_id or "global").strip() or "global"
        state = await self._get_loop_state()
        tenant_state = await self._get_tenant_state(state, tenant_key)
        tenant_semaphore = self._get_tenant_semaphore(tenant_state, lane)
        global_semaphore = self._get_global_semaphore(state, lane)
        timeout_seconds = self.config.tenant_admission_timeout_ms / 1000.0 if self.config.tenant_admission_timeout_ms > 0 else None

        t_wait_start = time.perf_counter()
        await self._acquire_slot(
            tenant_semaphore,
            timeout_seconds=timeout_seconds,
            tenant_id=tenant_key,
            lane=lane,
            feature=feature,
            http_on_timeout=http_on_timeout,
        )

        try:
            await self._acquire_slot(
                global_semaphore,
                timeout_seconds=timeout_seconds,
                tenant_id=tenant_key,
                lane=lane,
                feature=feature,
                http_on_timeout=http_on_timeout,
            )
        except Exception:
            tenant_semaphore.release()
            raise

        wait_ms = round((time.perf_counter() - t_wait_start) * 1000, 2)
        await self._mark_acquired(state, tenant_key, lane)
        self._log_wait(wait_ms, tenant_key, lane, feature)

        try:
            yield
        finally:
            global_semaphore.release()
            tenant_semaphore.release()
            await self._mark_released(state, tenant_key, lane)

    async def shutdown(self):
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            self._loop_states.pop(loop, None)

    async def _get_loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        with self._registry_lock:
            state = self._loop_states.get(loop)
            if state is None:
                state = _LoopState(
                    online_semaphore=asyncio.Semaphore(self.config.tenant_online_global_limit),
                    background_semaphore=asyncio.Semaphore(self.config.tenant_background_global_limit),
                )
                self._loop_states[loop] = state
                _LOG.info(
                    "Tenant admission runtime initialized: online_global=%d online_per_tenant=%d background_global=%d background_per_tenant=%d timeout_ms=%d",
                    self.config.tenant_online_global_limit,
                    self.config.tenant_online_per_tenant_limit,
                    self.config.tenant_background_global_limit,
                    self.config.tenant_background_per_tenant_limit,
                    self.config.tenant_admission_timeout_ms,
                )
        return state

    async def _get_tenant_state(self, state: _LoopState, tenant_id: str) -> _TenantState:
        async with state.lock:
            self._prune_idle_tenants_locked(state)
            tenant_state = state.tenant_states.get(tenant_id)
            if tenant_state is None:
                tenant_state = _TenantState(
                    online_semaphore=asyncio.Semaphore(self.config.tenant_online_per_tenant_limit),
                    background_semaphore=asyncio.Semaphore(self.config.tenant_background_per_tenant_limit),
                )
                state.tenant_states[tenant_id] = tenant_state
            tenant_state.last_used = time.monotonic()
            return tenant_state

    async def _mark_acquired(self, state: _LoopState, tenant_id: str, lane: str):
        async with state.lock:
            tenant_state = state.tenant_states.get(tenant_id)
            if not tenant_state:
                return
            if lane == "online":
                tenant_state.online_active += 1
            else:
                tenant_state.background_active += 1
            tenant_state.last_used = time.monotonic()

    async def _mark_released(self, state: _LoopState, tenant_id: str, lane: str):
        async with state.lock:
            tenant_state = state.tenant_states.get(tenant_id)
            if not tenant_state:
                return
            if lane == "online":
                tenant_state.online_active = max(0, tenant_state.online_active - 1)
            else:
                tenant_state.background_active = max(0, tenant_state.background_active - 1)
            tenant_state.last_used = time.monotonic()
            self._prune_idle_tenants_locked(state)

    async def _acquire_slot(
        self,
        semaphore: asyncio.Semaphore,
        *,
        timeout_seconds: float | None,
        tenant_id: str,
        lane: str,
        feature: str,
        http_on_timeout: bool,
    ):
        try:
            if timeout_seconds is None:
                await semaphore.acquire()
            else:
                await asyncio.wait_for(semaphore.acquire(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            detail = "The server is busy processing requests for this workspace. Please retry shortly."
            if http_on_timeout:
                raise HTTPException(status_code=429, detail=detail) from exc
            raise RuntimeError(detail) from exc

    def _get_tenant_semaphore(self, tenant_state: _TenantState, lane: str) -> asyncio.Semaphore:
        if lane == "online":
            return tenant_state.online_semaphore
        if lane == "background":
            return tenant_state.background_semaphore
        raise ValueError(f"Unsupported admission lane: {lane}")

    def _get_global_semaphore(self, state: _LoopState, lane: str) -> asyncio.Semaphore:
        if lane == "online":
            return state.online_semaphore
        if lane == "background":
            return state.background_semaphore
        raise ValueError(f"Unsupported admission lane: {lane}")

    def _prune_idle_tenants_locked(self, state: _LoopState):
        now = time.monotonic()
        ttl_seconds = float(self.config.tenant_state_ttl_sec)
        to_delete = []

        for tenant_id, tenant_state in state.tenant_states.items():
            if tenant_state.online_active or tenant_state.background_active:
                continue
            if (now - tenant_state.last_used) < ttl_seconds:
                continue
            to_delete.append(tenant_id)

        for tenant_id in to_delete:
            state.tenant_states.pop(tenant_id, None)

    def _log_wait(self, wait_ms: float, tenant_id: str, lane: str, feature: str):
        if wait_ms < 1.0:
            return

        log_metric(
            {
                "category": "runtime",
                "operation": f"{lane}_admission_wait",
                "model_name": "tenant_admission",
                "duration_ms": wait_ms,
                "metadata": {
                    "tenant_id": tenant_id,
                    "feature": feature,
                    "lane": lane,
                    "online_global_limit": self.config.tenant_online_global_limit,
                    "online_per_tenant_limit": self.config.tenant_online_per_tenant_limit,
                    "background_global_limit": self.config.tenant_background_global_limit,
                    "background_per_tenant_limit": self.config.tenant_background_per_tenant_limit,
                },
            }
        )


_TENANT_RUNTIME: Optional[TenantAdmissionRuntime] = None


def get_tenant_runtime() -> TenantAdmissionRuntime:
    global _TENANT_RUNTIME
    if _TENANT_RUNTIME is None:
        _TENANT_RUNTIME = TenantAdmissionRuntime()
    return _TENANT_RUNTIME
