import asyncio
from celery_app import celery_app
from main import process_ingestion_background, process_batch_ingestion_background, generate_podcast_background
from infrastructure import init_infra, close_infra

from services.platform_sync import PlatformSyncService

# Celery Worker: Startup is now model-free.
# Models will be loaded on-demand during task execution if not already present.
print("\n--- CELERY WORKER ACTIVE ---")
print("Status: Waiting for tasks...\n")

async def task_lifecycle(coro):
    """
    Ensures 'Upgrade 1' and 'Upgrade 2' compliance for transient Celery tasks.
    Initializes a private pool for the task and closes it immediately after.
    """
    await init_infra()
    try:
        return await coro
    finally:
        await close_infra()

@celery_app.task(name="sync_platform_task", queue="platform_sync")
def sync_platform_task():
    """Background task for Platform Sync & Purge cycle.
    Uses DistributedLock to ensure Exactly-Once execution.
    """
    from infrastructure import DistributedLock
    
    async def run_sync_with_lock():
        await init_infra()
        # Execution Lock: 5 minute timeout
        exec_lock = DistributedLock("active_platform_sync", timeout=300)
        
        try:
            if await exec_lock.acquire():
                print("CELERY: Acquired EXECUTION LOCK. Starting Sync...")
                sync_service = PlatformSyncService()
                await sync_service.run_sync_cycle()
                print("CELERY: Sync Completed.")
            else:
                print("CELERY: Sync already in progress elsewhere. Skipping.")
        finally:
            await exec_lock.release()
            await close_infra()

    asyncio.run(run_sync_with_lock())

@celery_app.task(name="process_ingestion", queue="ingestion")
def process_ingestion_task(job_id: str, input_path: str, file_type: str, ingestion_id: str, file_id: str, filename: str, batch_id: str, mode: str = "single"):
    """
    Synchronous wrapper for Celery to run the async ingestion pipeline.
    """
    asyncio.run(task_lifecycle(process_ingestion_background(job_id, input_path, file_type, ingestion_id, file_id, filename, batch_id, mode)))

@celery_app.task(name="process_batch_ingestion", queue="ingestion")
def process_batch_ingestion_task(job_id: str, urls: list, batch_id: str):
    """
    Synchronous wrapper for Celery to run the async batch ingestion pipeline.
    """
    asyncio.run(task_lifecycle(process_batch_ingestion_background(job_id, urls, batch_id)))

@celery_app.task(name="generate_podcast_task", queue="podcast")
def generate_podcast_task(job_id: str, username: str, batch_id: str, topic: str, complexity: str):
    """
    Background task for podcast generation.
    """
    asyncio.run(task_lifecycle(generate_podcast_background(job_id, username, batch_id, topic, complexity)))
