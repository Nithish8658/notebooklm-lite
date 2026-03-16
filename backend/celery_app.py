import os
from celery import Celery

from celery.schedules import crontab

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
    # Worker concurrency controls (to prevent CPU stampedes)
    worker_concurrency=4, 
    worker_prefetch_multiplier=1,
    
    # AC-30: Task Reliability - only ack after completion to prevent loss,
    # and reject if worker dies to prevent it being stuck in 'reserved' state
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Schedule Platform Sync every 2 minutes
celery_app.conf.beat_schedule = {
    'sync-platform-every-2-minutes': {
        'task': 'sync_platform',
        'schedule': 120.0, # seconds
    },
}
