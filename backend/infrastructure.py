import os
import asyncio
import weakref
from typing import Optional
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# --- CONFIGURATION ---
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://notebook:notebook@localhost:5432/notebooklm"
)

class AppInfrastructure:
    """
    Sophisticated Infrastructure Container.
    Enforces 'Upgrade 3' Connection Budgets and 'Upgrade 1' Process Scoping.
    """
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.db_engine = None
        self.db_sessionmaker = None
        self.db_semaphore: Optional[asyncio.Semaphore] = None
        self.qdrant_semaphore: Optional[asyncio.Semaphore] = None

    async def initialize(self):
        """Initializes all bounded connection pools for the current loop."""
        # Redis: Explicit Budget (40 connections)
        self.redis_client = redis.from_url(
            REDIS_URL, 
            decode_responses=True, 
            max_connections=40,
            socket_timeout=5,
            retry_on_timeout=True
        )
        
        # Qdrant: Explicit Budget (50 concurrent requests via Semaphore)
        self.qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
        self.qdrant_semaphore = asyncio.Semaphore(50)
        
        # PostgreSQL: Explicit Budget (30 pool + 10 overflow)
        self.db_engine = create_async_engine(
            SQLALCHEMY_DATABASE_URL,
            pool_size=30,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.db_sessionmaker = async_sessionmaker(
            bind=self.db_engine, 
            class_=AsyncSession, 
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        # Database Guard: Industrial limit to prevent DB crash
        self.db_semaphore = asyncio.Semaphore(80)

    async def close(self):
        """Upgrade 2: Explicit Lifecycle Management (Graceful Shutdown)"""
        if self.redis_client:
            await self.redis_client.close()
        if self.qdrant_client:
            # AsyncQdrantClient doesn't have an explicit close() in 1.7.3,
            # but we nullify it to prevent further usage.
            self.qdrant_client = None
        if self.db_engine:
            await self.db_engine.dispose()

# --- LOOP-AWARE REGISTRY ---
# Uses WeakKeyDictionary to ensure that when a loop is garbage collected, 
# its infrastructure reference is also cleared.
_infra_registry = weakref.WeakKeyDictionary()

def get_infra() -> AppInfrastructure:
    """Retrieves the infrastructure container for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError("Infrastructure accessed outside of an asyncio event loop.")
    
    if loop not in _infra_registry:
        raise RuntimeError(
            "Infrastructure not initialized for the current loop. "
            "Ensure 'init_infra()' was called in this context (e.g., in lifespan or task wrapper)."
        )
    return _infra_registry[loop]

async def init_infra():
    """Initializes infrastructure for the current event loop."""
    loop = asyncio.get_running_loop()
    if loop not in _infra_registry:
        infra = AppInfrastructure()
        await infra.initialize()
        _infra_registry[loop] = infra

async def close_infra():
    """Closes infrastructure for the current event loop and removes it from registry."""
    loop = asyncio.get_running_loop()
    infra = _infra_registry.pop(loop, None)
    if infra:
        await infra.close()

# --- SOPHISTICATED COMPATIBILITY PROXIES ---
# These allow main.py to keep using 'redis_client' and 'qdrant' without changes.
class InfrastructureProxy:
    def __init__(self, attr_name: str):
        self._attr_name = attr_name

    def _get_target(self):
        return getattr(get_infra(), self._attr_name)

    def __getattr__(self, name):
        return getattr(self._get_target(), name)

# Exported singletons that act as proxies to the loop-scoped clients
redis_client = InfrastructureProxy("redis_client")
qdrant = InfrastructureProxy("qdrant_client")
