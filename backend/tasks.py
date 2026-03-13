import asyncio
from celery_app import celery_app
from main import process_ingestion_background, process_batch_ingestion_background, generate_podcast_background

# Celery Worker: Startup is now model-free.
# Models will be loaded on-demand during task execution if not already present.
print("\n--- CELERY WORKER ACTIVE ---")
print("Status: Waiting for tasks...\n")

@celery_app.task(name="process_ingestion")
def process_ingestion_task(job_id: str, input_path: str, file_type: str, ingestion_id: str, file_id: str, filename: str, cohort_id: str, mode: str = "single"):
    """
    Synchronous wrapper for Celery to run the async ingestion pipeline.
    """
    asyncio.run(process_ingestion_background(job_id, input_path, file_type, ingestion_id, file_id, filename, cohort_id, mode))

@celery_app.task(name="process_batch_ingestion")
def process_batch_ingestion_task(job_id: str, urls: list, cohort_id: str):
    """
    Synchronous wrapper for Celery to run the async batch ingestion pipeline.
    """
    asyncio.run(process_batch_ingestion_background(job_id, urls, cohort_id))

@celery_app.task(name="generate_podcast_task")
def generate_podcast_task(job_id: str, user_id: str, cohort_id: str, topic: str, complexity: str):
    """
    Background task for podcast generation.
    """
    asyncio.run(generate_podcast_background(job_id, user_id, cohort_id, topic, complexity))
