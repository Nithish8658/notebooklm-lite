import os
import threading
import time

from celery import Celery, signals
from kombu import Queue

# Use Redis as the broker and result backend
redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "notebooklm_tasks",
    broker=redis_url,
    backend=redis_url,
    include=['tasks'] # We will define our tasks here
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # AC-62: Explicit task discovery
    imports=("tasks",),
    # Worker concurrency controls (to prevent CPU stampedes)
    worker_concurrency=4, 
    worker_prefetch_multiplier=1,
    
    # AC-30: Task Reliability - only ack after completion to prevent loss,
    # and reject if worker dies to prevent it being stuck in 'reserved' state
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Routing: keep platform sync isolated to its own queue so it cannot block ingestion
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("platform_sync", routing_key="platform_sync"),
        Queue("ingestion", routing_key="ingestion"),
        Queue("podcast", routing_key="podcast"),
    ),
    task_routes={
        "sync_platform": {"queue": "platform_sync", "routing_key": "platform_sync"},
        "process_ingestion": {"queue": "ingestion", "routing_key": "ingestion"},
        "process_batch_ingestion": {"queue": "ingestion", "routing_key": "ingestion"},
        "generate_podcast_task": {"queue": "podcast", "routing_key": "podcast"},
    },
)

# Platform Sync is managed via a single scheduler thread inside the Celery worker.
# This avoids requiring `celery beat`, and prevents duplicate schedules from multiple
# uvicorn reload processes.

_SYNC_INTERVAL_SECONDS = 120

@signals.worker_ready.connect
def _start_platform_sync_scheduler(sender=None, **kwargs):
    """Starts a background scheduler thread when the worker is ready."""
    def _scheduler_loop():
        # Run once immediately, then sleep between runs.
        while True:
            try:
                celery_app.send_task("sync_platform")
            except Exception as e:
                # We don't want the thread to die; log and retry
                print(f"WARNING: Failed to enqueue platform sync: {e}")
            time.sleep(_SYNC_INTERVAL_SECONDS)

    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
