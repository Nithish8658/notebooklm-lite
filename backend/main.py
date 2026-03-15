import os
import sys
import json
import asyncio
import traceback
import time
import datetime
from pathlib import Path

import torchaudio_patch

# --- HARDWARE ACCELERATION (Must be set before any ML imports) ---
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List
import hashlib

from ingestion.pdf_ingestor import ingest_pdf_layout_aware
from ingestion.pptx_ingestor import ingest_pptx_structure_aware
from ingestion.excel_ingestor import ingest_excel
from ingestion.video_ingestor import ingest_video_file
from ingestion.youtube_ingestor_whisperx import ingest_youtube_video, load_whisperx_model
from ingestion.web_ingestor import WebIngestor
from chunking.semantic_chunker import semantic_chunk_blocks
from graph.chunk_graph import build_chunk_graph
from retrieval.bm25_index import BM25ChunkIndex
from retrieval.query_rewriter import rewrite_query_ensemble
from services.retrieval import retrieve_candidates
from services.llm import call_gemini_async
from rerank import load_reranker, get_reranker, rerank_with_cross_encoder
from services.flashcards import generate_flashcards
from services.quiz import generate_quiz_for_topic
from services.podcast import generate_podcast, PodcastRefusalError

from database import init_db, get_db, User, Flashcard, Quiz, Podcast, Batch, UserEnrollment, IngestionJob, AsyncSessionLocal, engine
from services.index_state import index_manager
from services.embeddings import BatchEmbeddingManager, get_embedding_model, load_embedding_model
from services.concurrency import BatchRerankManager, db_semaphore
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import Depends

import torch
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import time
from metrics import log_metric, get_metrics
import redis.asyncio as redis

# ===== ENV =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# --- COOLDOWN HELPERS ---
async def set_cooldown(batch_id: str, topic: str, cache_type: str, duration: int = 180):
    """Sets a 3-minute (180s) cooldown for a specific topic, type, and BATCH."""
    try:
        key = f"cooldown:{batch_id}:{cache_type}:{hashlib.md5(topic.encode()).hexdigest()}"
        await redis_client.set(key, "1", ex=duration)
    except Exception as e:
        print(f"DEBUG: Failed to set cooldown: {e}")

async def check_cooldown(batch_id: str, topic: str, cache_type: str):
    """Raises HTTPException if cooldown is active for this BATCH."""
    try:
        key = f"cooldown:{batch_id}:{cache_type}:{hashlib.md5(topic.encode()).hexdigest()}"
        if await redis_client.exists(key):
            raise HTTPException(
                status_code=429, 
                detail="The provided text contains no information regarding the topic. Please wait 3 minutes before retrying this specific query."
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Cooldown check error (ignoring): {e}")

# ===== CACHE & DB =====
class RedisProxy:
    def __init__(self, url):
        self._url = url
        self._client = None
        self._loop = None

    def _get_client(self):
        try:
            loop = asyncio.get_running_loop()
            if self._client is None or self._loop != loop or loop.is_closed():
                self._client = redis.from_url(self._url, decode_responses=True)
                self._loop = loop
            return self._client
        except RuntimeError:
            # Fallback for sync or non-loop contexts
            if self._client is None:
                self._client = redis.from_url(self._url, decode_responses=True)
            return self._client

    def __getattr__(self, name):
        return getattr(self._get_client(), name)

class QdrantProxy:
    def __init__(self, url):
        self._url = url
        self._client = None
        self._loop = None

    def _get_client(self):
        try:
            loop = asyncio.get_running_loop()
            if self._client is None or self._loop != loop or loop.is_closed():
                self._client = AsyncQdrantClient(url=self._url)
                self._loop = loop
            return self._client
        except RuntimeError:
            if self._client is None:
                self._client = AsyncQdrantClient(url=self._url)
            return self._client

    def __getattr__(self, name):
        return getattr(self._get_client(), name)

redis_client = RedisProxy(REDIS_URL)
qdrant = QdrantProxy(QDRANT_URL)
QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"

# AC-22: Instantiate the Global Batch Embedding Manager
embedding_manager = BatchEmbeddingManager(model_getter=get_embedding_model)
# AC-25: Instantiate the Global Batch Rerank Manager for Cache Verification
rerank_manager = BatchRerankManager(reranker_getter=get_reranker)

async def initialize_qdrant():
    """Ensures necessary Qdrant collections are ready."""
    print("SYNC: Initializing Qdrant collections...")
    for coll in [QDRANT_COLLECTION, CACHE_COLLECTION]:
        try:
            collections_res = await qdrant.get_collections()
            exists = any(c.name == coll for c in collections_res.collections)
            
            if not exists:
                print(f"SYNC: Collection {coll} not recognized. Creating fresh...")
                await qdrant.create_collection(
                    collection_name=coll,
                    vectors_config={"default": VectorParams(size=768, distance=Distance.COSINE)}
                )
        except Exception as e:
            print(f"WARNING: Qdrant initialization failed for {coll}: {e}")

# ===== LIFESPAN =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- HW ACCELERATION DIAGNOSTICS ---")
    try:
        import torch
        print("PyTorch Config:")
        print(torch.__config__.show())
    except ImportError:
        print("PyTorch not found.")

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime Version: {ort.get_version_string()}")
        print(f"Available Providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime not found.")
    print("-----------------------------------\n")

    print("STARTUP: Initializing Database...")
    try:
        init_db()
    except Exception as e:
        print(f"WARNING: DB connection failed: {e}")

    print("STARTUP: Initializing Qdrant (Vector DB)...")
    try:
        await initialize_qdrant()
    except Exception as e:
        print(f"WARNING: Initial Qdrant check failed (might still be booting): {e}")

    print("\n--- MASTER STARTUP: Verifying & Loading Models ---")
    try:
        from services.bootstrap import verify_all_models
        from services.embeddings import load_embedding_model, get_embedding_model
        from rerank import load_reranker, get_reranker
        
        # 1. Master Download (Only main process handles this)
        print("STARTUP: Verifying all required models on disk...")
        await asyncio.to_thread(verify_all_models)
        
        # 2. Load into memory
        print("STARTUP: Loading models into RAM...")
        load_embedding_model()
        load_whisper_model()
        load_reranker()
        
        # 3. Warm-up Inference
        print("STARTUP: Running model sanity checks (Warm-up)...")
        emb_model = get_embedding_model()
        _ = emb_model.encode(["Sanity check for BGE Embedding"])
        
        reranker = get_reranker()
        _ = reranker.predict([("test query", "test document context")])
        print("STARTUP: All models verified and warmed up correctly.")
        
        # --- AUTO-SYNC STARTUP ---
        # 1. Run one-time sync cycle for Users and Enrollments ONLY (Database only)
        from services.platform_sync import PlatformSyncService
        sync_service = PlatformSyncService()
        print("STARTUP: Running initial Platform Sync (Users & Enrollments)...")
        await sync_service.sync_users_and_enrollments()
        print("STARTUP: Initial User Sync Complete.")

        # 2. Start the Continuous Poller in a background thread
        import threading
        def run_continuous_poller():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(sync_service.start())

        poller_thread = threading.Thread(target=run_continuous_poller, daemon=True)
        poller_thread.start()
        print("STARTUP: Platform Sync Service background poller active.")
    except Exception as e:
        print(f" CRITICAL ERROR during model loading: {e}")
        traceback.print_exc()
    yield

# ===== CACHE HELPERS =====
async def get_semantic_cache(batch_id: str, topic: str, cache_type: str, filters: Optional[Dict[str, Any]] = None):
    """
    Looks up a semantically similar topic in the cache.
    Supports tiered caching via optional metadata filters (e.g. complexity).
    """
    try:
        # AC-22: Use Batch Embedding Manager for efficient concurrent processing
        query_vec_list = await embedding_manager.get_embedding(topic)

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build strict filters
        must_conditions = [
            FieldCondition(key="batch_id", match=MatchValue(value=batch_id)),
            FieldCondition(key="cache_type", match=MatchValue(value=cache_type))
        ]
        
        # Add dynamic metadata filters (e.g., complexity for chat_response)
        if filters:
            for key, val in filters.items():
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))

        cache_results = await qdrant.search(
            collection_name=CACHE_COLLECTION,
            query_vector=("default", query_vec_list),
            query_filter=Filter(must=must_conditions),
            limit=1,
            score_threshold=0.96
        )

        if cache_results:
            hit = cache_results[0]
            point_id = hit.id
            score = hit.score
            cached_query = hit.payload.get("query", "")

            # --- CROSS-ENCODER VALIDATION ---
            is_valid_hit = True
            if score < 0.999 and cached_query:
                ce_score = await rerank_manager.verify_logic(topic, cached_query)
                if ce_score < 0.5:
                    is_valid_hit = False

            if is_valid_hit:
                cached_json = await redis_client.get(f"studio_cache:{point_id}")
                if cached_json:
                    print(f"CACHE HIT [{cache_type}]: '{topic[:30]}...' (Score: {score:.4f})")
                    return json.loads(cached_json)
    except Exception as e:
        print(f"WARNING: Cache lookup failed: {e}")
    return None

async def set_semantic_cache(batch_id: str, topic: str, cache_type: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
    """Saves a result to the semantic cache with optional tiered metadata."""
    try:
        model = get_embedding_model()
        query_vec = await asyncio.to_thread(
            model.encode, 
            "Represent this question for searching relevant passages: " + topic
        )
        query_vec_list = query_vec.tolist()
        
        # Build payload
        payload = {"batch_id": batch_id, "query": topic, "cache_type": cache_type}
        if metadata:
            payload.update(metadata)

        new_point_id = str(uuid4())
        await qdrant.upsert(
            collection_name=CACHE_COLLECTION,
            points=[PointStruct(
                id=new_point_id,
                vector={"default": query_vec_list},
                payload=payload
            )]
        )
        # Cache for 24 hours
        await redis_client.setex(
            f"studio_cache:{new_point_id}",
            86400,
            json.dumps(data)
        )
        print(f"SUCCESS: Cache Tier saved: {cache_type}")
    except Exception as e:
        print(f"WARNING: Failed to save to studio cache: {e}")

# ===== APP =====
app = FastAPI(title="NotebookLM Lite API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PODCAST_DIR = Path("uploads/podcasts")
PODCAST_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/podcasts", StaticFiles(directory=PODCAST_DIR), name="podcasts")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error."})

# ===== STATE & JOBS =====
# Stateless Architecture: No more global STATE. 
# BM25 and Graph are stored in Postgres per batch.
# Chunk lookups are performed directly against Qdrant.
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ===== METRICS =====
@app.get("/metrics")
async def get_performance_metrics():
    raw_metrics = get_metrics()
    llm_metrics = [m for m in raw_metrics if m.get("category") == "llm"]
    total_tokens = sum(m.get("metadata", {}).get("total_tokens", 0) for m in llm_metrics)
    estimated_cost = (total_tokens / 1_000_000) * 0.10
    return {
        "raw": raw_metrics[-100:],
        "summary": {
            "total_calls": len(llm_metrics),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4)
        }
    }

@app.delete("/metrics")
async def clear_usage_metrics():
    from metrics import METRICS_FILE
    if METRICS_FILE.exists(): METRICS_FILE.unlink()
    return {"status": "success"}

# ===== STUDIO ROUTES =====
@app.get("/studio/flashcards")
async def get_flashcards(username: str, batch_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Flashcard).where(Flashcard.username == username, Flashcard.batch_id == batch_id))
    cards = result.scalars().all()
    all_cards = []
    for c in cards:
        if isinstance(c.payload, list): all_cards.extend(c.payload)
    return {"cards": all_cards}

class FlashcardRequest(BaseModel):
    username: str
    active_batch_id: str
    topics: Optional[List[str]] = None
    complexity: str = Field("Undergrad", description="Complexity level")

@app.post("/studio/flashcards")
async def create_flashcards(req: FlashcardRequest, db: AsyncSession = Depends(get_db)):
    await verify_user_enrollment(req.username, req.active_batch_id, db)
    if not await ensure_batch_is_loaded(req.active_batch_id, db):
        raise HTTPException(status_code=400, detail="No documents indexed.")

    # --- SEMANTIC CACHE LOOKUP ---
    # Normalize topics and include complexity in the cache key
    sorted_topics = sorted(req.topics) if req.topics else ["general"]
    topic_str = f"{','.join(sorted_topics)} [Level: {req.complexity}]"
    
    # AC-45: Rate Limit - Check for 3min cooldown on empty/invalid attempts
    await check_cooldown(req.active_batch_id, topic_str, "flashcards")

    cached_cards = await get_semantic_cache(req.active_batch_id, topic_str, "flashcards")
    if cached_cards:
        # Save to DB for this user too so it shows up in their list
        db.add(Flashcard(username=req.username, batch_id=req.active_batch_id, payload=cached_cards, complexity=req.complexity))
        await db.commit()
        return {"cards": cached_cards}

    # Fetch Metadata (Optimized RAM Singleton)
    brain = await index_manager.get_brain(req.active_batch_id)
    if not brain:
        raise HTTPException(status_code=404, detail="Course resources not found.")
    
    bm25 = brain["bm25"]
    graph = brain["graph"]

    try:
        cards = await generate_flashcards(
            bm25=bm25,
            graph=graph,
            chunk_fetcher=lambda ids: fetch_chunk_details(ids, req.active_batch_id),
            batch_id=req.active_batch_id,
            dense_retrieve_fn=dense_retrieve,
            topics=req.topics,
            complexity=req.complexity
        )
        if cards and len(cards) > 0:
            db.add(Flashcard(username=req.username, batch_id=req.active_batch_id, payload=cards, complexity=req.complexity))
            await db.commit()
            # --- SAVE TO SEMANTIC CACHE (Background) ---
            # Only cache proper answers
            asyncio.create_task(set_semantic_cache(req.active_batch_id, topic_str, "flashcards", cards))
        else:
            # AC-45: Negative Cache / Cooldown - Prevent retrying for 3 mins if no info found
            await set_cooldown(req.active_batch_id, topic_str, "flashcards")
            
        return {"cards": cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/studio/quizzes")
async def get_quizzes(username: str, batch_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Quiz).where(Quiz.username == username, Quiz.batch_id == batch_id))
    quizzes = result.scalars().all()
    return {"quizzes": [q.payload for q in quizzes]}

class QuizGenerateRequest(BaseModel):
    username: str
    active_batch_id: str
    topic: str
    complexity: str = Field("Undergrad", description="Complexity level")

@app.post("/studio/quiz/generate")
async def generate_quiz_endpoint(req: QuizGenerateRequest, db: AsyncSession = Depends(get_db)):
    await verify_user_enrollment(req.username, req.active_batch_id, db)
    if not await ensure_batch_is_loaded(req.active_batch_id, db):
        raise HTTPException(status_code=400, detail="No documents indexed.")

    # --- SEMANTIC CACHE LOOKUP ---
    topic_with_level = f"{req.topic} [Level: {req.complexity}]"
    
    # AC-45: Rate Limit - Check for 3min cooldown
    await check_cooldown(req.active_batch_id, req.topic, "quiz")

    cached_quiz = await get_semantic_cache(req.active_batch_id, topic_with_level, "quiz")
    if cached_quiz:
        db.add(Quiz(username=req.username, batch_id=req.active_batch_id, payload=cached_quiz, complexity=req.complexity))
        await db.commit()
        return cached_quiz

    # Fetch Metadata (Optimized RAM Singleton)
    brain = await index_manager.get_brain(req.active_batch_id)
    if not brain:
        raise HTTPException(status_code=404, detail="Course resources not found.")
    
    bm25 = brain["bm25"]
    graph = brain["graph"]

    try:
        quiz = await generate_quiz_for_topic(
            topic=req.topic,
            bm25=bm25,
            graph=graph,
            chunk_fetcher=lambda ids: fetch_chunk_details(ids, req.active_batch_id),
            dense_retrieve_fn=dense_retrieve,
            batch_id=req.active_batch_id,
            complexity=req.complexity
        )
        if quiz:
            db.add(Quiz(username=req.username, batch_id=req.active_batch_id, payload=quiz.dict(), complexity=req.complexity))
            await db.commit()
            # --- SAVE TO SEMANTIC CACHE (Background) ---
            # Only cache proper answers
            asyncio.create_task(set_semantic_cache(req.active_batch_id, topic_with_level, "quiz", quiz.dict()))
            return quiz.dict()
        else:
            # AC-45: Negative Cache / Cooldown - Prevent retrying for 3 mins if no info found
            await set_cooldown(req.active_batch_id, req.topic, "quiz")
            raise HTTPException(status_code=400, detail="The provided text contains no information regarding the topic.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PodcastGenerateRequest(BaseModel):
    username: str
    active_batch_id: str
    topic: str = Field(..., max_length=500)
    complexity: str = Field("Undergrad")

@app.get("/studio/podcasts")
async def get_podcasts(username: str, batch_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Podcast).where(Podcast.username == username, Podcast.batch_id == batch_id))
    podcasts = result.scalars().all()
    formatted = []
    for p in podcasts:
        script_data = p.transcript
        if isinstance(script_data, str):
            try:
                import json
                script_data = json.loads(script_data)
            except:
                try:
                    import ast
                    script_data = ast.literal_eval(script_data)
                except:
                    script_data = []
        
        # Ensure it's a list for the frontend .map()
        if not isinstance(script_data, list):
            script_data = []

        formatted.append({
            "id": p.id,
            "username": p.username, 
            "audio_path": p.filename, 
            "script": script_data, 
            "topic": p.topic,
            "duration_seconds": 0
        })
    return {"podcasts": formatted}


@app.post("/studio/podcast/generate")
async def generate_podcast_endpoint(req: PodcastGenerateRequest, db: AsyncSession = Depends(get_db)):
    await verify_user_enrollment(req.username, req.active_batch_id, db)
    if not await ensure_batch_is_loaded(req.active_batch_id, db):
        raise HTTPException(status_code=400, detail="No documents indexed.")

    # --- SEMANTIC CACHE LOOKUP ---
    topic_with_level = f"{req.topic} [Level: {req.complexity}]"
    
    # AC-45: Rate Limit - Check for 3min cooldown
    await check_cooldown(req.active_batch_id, req.topic, "podcast")

    cached_podcast = await get_semantic_cache(req.active_batch_id, topic_with_level, "podcast")
    
    # Validation: Ensure cache has the required script and the audio file actually exists on disk
    is_cache_valid = False
    if cached_podcast:
        script_list = cached_podcast.get("script") if isinstance(cached_podcast, dict) else cached_podcast
        audio_path = cached_podcast.get("audio_path") if isinstance(cached_podcast, dict) else None
        
        if script_list and audio_path:
            full_audio_path = PODCAST_DIR / audio_path
            if full_audio_path.exists():
                is_cache_valid = True
                
    if is_cache_valid:
        # Re-verify normalize for DB insertion
        script_list = cached_podcast.get("script") if isinstance(cached_podcast, dict) else cached_podcast
        audio_path = cached_podcast.get("audio_path") if isinstance(cached_podcast, dict) else None
        
        db.add(Podcast(
            username=req.username, 
            batch_id=req.active_batch_id, 
            filename=audio_path, 
            transcript=json.dumps(script_list), 
            topic=req.topic, 
            complexity=req.complexity
        ))
        await db.commit()
        # Return as a completed job so frontend knows it's ready immediately
        return JobResponse(job_id="cached", status=JobStatus.COMPLETED, filename=req.topic)

    # Dispatch to Celery
    job_id = f"podcast_{uuid4().hex[:8]}"
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename=f"Podcast: {req.topic}")
    db.add(new_job)
    await db.commit()

    from celery_app import celery_app
    celery_app.send_task("generate_podcast_task", args=[
        job_id, req.username, req.active_batch_id, req.topic, req.complexity
    ])

    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename=req.topic)

async def generate_podcast_background(job_id: str, username: str, batch_id: str, topic: str, complexity: str):
    try:
        # 1. Ensure metadata is ready
        async with AsyncSessionLocal() as db:
            if not await ensure_batch_is_loaded(batch_id, db):
                 raise RuntimeError("Batch metadata not ready.")

        # 2. Fetch Metadata (RAM Singleton)
        brain = await index_manager.get_brain(batch_id)
        if not brain: raise RuntimeError("Course resources not found.")
        
        bm25 = brain["bm25"]
        graph = brain["graph"]

        podcast = await generate_podcast(
            topic=topic,
            bm25=bm25,
            graph=graph,
            chunk_fetcher=lambda ids: fetch_chunk_details(ids, batch_id),
            dense_retrieve_fn=dense_retrieve,
            batch_id=batch_id,
            complexity=complexity
        )
        
        if not podcast:
            raise RuntimeError("Not enough resource to generate podcast")

        script_list = [s.dict() for s in podcast.script]
        
        # 3. Save to DB
        async with AsyncSessionLocal() as db:
            db.add(Podcast(
                username=username, 
                batch_id=batch_id, 
                filename=podcast.audio_path, 
                transcript=json.dumps(script_list), 
                topic=topic, 
                complexity=complexity
            ))
            
            # Update Job
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.COMPLETED
                job.message = "Success"
            
            await db.commit()

        # 4. Cache
        topic_with_level = f"{topic} [Level: {complexity}]"
        await set_semantic_cache(batch_id, topic_with_level, "podcast", {
            "audio_path": podcast.audio_path,
            "script": script_list,
            "topic": topic,
            "duration_seconds": podcast.duration_seconds
        })

    except PodcastRefusalError as pre:
        # AC-45: Security Gate refusal / No Info Found
        # Set a 3min cooldown to prevent spamming the same invalid topic
        await set_cooldown(batch_id, topic, "podcast")
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.FAILED
                job.message = str(pre)
                await db.commit()

    except ValueError as ve:
        # AC-45: Security Gate refusal / No Info Found
        # Set a 3min cooldown to prevent spamming the same invalid topic
        await set_cooldown(batch_id, topic, "podcast")
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.FAILED
                job.message = str(ve)
                await db.commit()

    except Exception as e:
        traceback.print_exc()
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.FAILED
                job.message = str(e)
                await db.commit()
    finally:
        from database import _db_proxy
        await _db_proxy.get_engine().dispose()

# ===== DELETE STUDIO TOOLS =====
@app.delete("/studio/flashcards")
async def delete_flashcards(username: str, batch_id: str, topic: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Flashcard).where(Flashcard.username == username, Flashcard.batch_id == batch_id))
    records = result.scalars().all()
    
    deleted_any = False
    for rec in records:
        original_len = len(rec.payload)
        new_payload = [card for card in rec.payload if card.get("topic") != topic]
        if len(new_payload) != original_len:
            deleted_any = True
            if not new_payload:
                await db.delete(rec)
            else:
                rec.payload = new_payload
                
    if deleted_any:
        await db.commit()
        return {"status": "success"}
    return {"status": "not_found"}

@app.delete("/studio/quiz")
async def delete_quiz(username: str, batch_id: str, topic: str, db: AsyncSession = Depends(get_db)):
    # Since Quiz payload is a single object with a topic field
    result = await db.execute(select(Quiz).where(Quiz.username == username, Quiz.batch_id == batch_id))
    records = result.scalars().all()
    
    deleted_any = False
    for rec in records:
        if rec.payload.get("topic") == topic:
            await db.delete(rec)
            deleted_any = True
            
    if deleted_any:
        await db.commit()
        return {"status": "success"}
    return {"status": "not_found"}

@app.delete("/studio/podcast")
async def delete_podcast(username: str, batch_id: str, topic: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Podcast).where(Podcast.username == username, Podcast.batch_id == batch_id, Podcast.topic == topic))
    records = result.scalars().all()
    
    if not records:
        return {"status": "not_found"}
        
    for rec in records:
        # Delete the audio file
        if rec.filename:
            file_path = PODCAST_DIR / rec.filename
            if file_path.exists():
                try:
                    file_path.unlink()
                except: pass
        await db.delete(rec)
        
    await db.commit()
    return {"status": "success"}
# ===== SOURCES MANAGEMENT =====
async def list_sources(batch_id: Optional[str] = None):
    all_stored = []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        scroll_filter = None
        if batch_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="batch_id", match=MatchValue(value=batch_id))]
            )

        response = await qdrant.scroll(
            collection_name=QDRANT_COLLECTION, 
            scroll_filter=scroll_filter,
            limit=100, 
            with_payload=True
        )
        points, _ = response
        seen = {}
        for p in points:
            doc_id = p.payload.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": p.payload.get("metadata", {}).get("filename", "Unknown"),
                    "type": p.payload.get("type", "persistent"),
                    "is_active": True, # In stateless mode, if it's in Qdrant, it's available
                    "ingested_at": p.payload.get("metadata", {}).get("ingested_at")
                }
        all_stored = list(seen.values())
    except Exception: pass
    return {"sources": all_stored}

@app.get("/sources")
async def list_sources_endpoint(batch_id: Optional[str] = None):
    return await list_sources(batch_id)

class ActivateRequest(BaseModel):
    document_ids: List[str]

@app.post("/sources/activate")
async def activate_sources(req: ActivateRequest):
    # In stateless mode, 'activating' is effectively a no-op as everything is 
    # loaded on-demand from Qdrant/Postgres per request.
    return {"status": "success", "message": "Stateless activation complete."}

@app.post("/sources/deactivate")
async def deactivate_sources(req: ActivateRequest):
    # In stateless mode, deactivation is a no-op.
    return {"status": "success", "message": "Stateless deactivation complete."}

@app.delete("/sources/{document_id}")
async def delete_source(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Fetch metadata for physical deletion
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        response = await qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
            limit=1,
            with_payload=True
        )
        points = response[0]
        if not points:
            raise HTTPException(status_code=404, detail="Source not found")
        
        doc_payload = points[0].payload
        batch_id = doc_payload.get("batch_id") or doc_payload.get("metadata", {}).get("batch_id")
        filename = doc_payload.get("metadata", {}).get("filename")

        # 2. Remove from Qdrant
        await qdrant.delete(
            collection_name=QDRANT_COLLECTION, 
            points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        )
        
        # 3. Physical File Deletion
        if batch_id and filename:
            target_file = UPLOAD_DIR / batch_id / f"{document_id}_{filename}"
            if target_file.exists():
                try:
                    target_file.unlink()
                except: pass

        # 4. Invalidate Metadata in DB so it's re-built next time
        if batch_id:
            result = await db.execute(select(Batch).where(Batch.id == batch_id))
            batch = result.scalars().first()
            if batch:
                batch.bm25_data = None
                batch.graph_data = None
                await db.commit()
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources")
async def clear_all_sources(db: AsyncSession = Depends(get_db)):
    try:
        await qdrant.delete_collection(QDRANT_COLLECTION)
        await qdrant.create_collection(collection_name=QDRANT_COLLECTION, vectors_config={"default": VectorParams(size=768, distance=Distance.COSINE)})
        
        # Wipe all batch metadata
        from sqlalchemy import update
        await db.execute(update(Batch).values(bm25_data=None, graph_data=None))
        await db.commit()
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== HELPERS =====
from services.embeddings import get_embedding_model
_whisper_model = None
def load_whisper_model():
    global _whisper_model
    if _whisper_model is None: _whisper_model = load_whisperx_model()
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None: 
        print("INFO: WhisperX model requested but not loaded. Loading now...")
        load_whisper_model()
    return _whisper_model

async def dense_retrieve(query: str, batch_id: str, top_k: int = 50) -> dict:
    # AC-22: Use Batch Embedding Manager for efficient concurrent processing
    query_vec = await embedding_manager.get_embedding(query)
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    results = await qdrant.search(
        collection_name=QDRANT_COLLECTION, 
        query_vector=("default", query_vec), 
        query_filter=Filter(must=[FieldCondition(key="batch_id", match=MatchValue(value=batch_id))]), 
        limit=top_k, 
        with_payload=True
    )
    return {p.payload["chunk_id"]: float(p.score) for p in results if "chunk_id" in p.payload}

async def fetch_chunk_details(chunk_ids: List[str], batch_id: str = None) -> List[Dict]:
    """
    Helper to fetch full chunk data from Qdrant by IDs.
    AC-20: Security Hard-Lock - Always filter by batch_id to prevent cross-tenant data leakage.
    """
    if not chunk_ids:
        return []
    
    from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
    
    # Base filter: match any of the requested chunk IDs
    must_conditions = [FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]
    
    # Security Gate: If batch_id is provided, enforce it strictly
    if batch_id:
        must_conditions.append(FieldCondition(key="batch_id", match=MatchValue(value=batch_id)))

    response = await qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(must=must_conditions),
        limit=len(chunk_ids),
        with_payload=True
    )
    points, _ = response
    return [p.payload for p in points]

async def verify_user_enrollment(username: str, batch_id: str, db: AsyncSession):
    """
    AC-24: Optimized Enrollment Verification with Redis Caching and DB Throttling.
    Eliminates redundant DB hits for every message.
    """
    cache_key = f"enrollment:{username}:{batch_id}"
    try:
        cached = await redis_client.get(cache_key)
        if cached == "1":
            return # Validated via cache
    except Exception as e:
        print(f"WARNING: Redis Enrollment Cache check failed: {e}")

    # AC-26: Database Guard - only allow N physical connections at once
    async with db_semaphore:
        # Fallback to DB
        result = await db.execute(select(UserEnrollment).where(UserEnrollment.username == username, UserEnrollment.batch_id == batch_id))
        if not result.scalars().first(): 
            raise HTTPException(status_code=403, detail="Access Denied")
    
    # Success: Cache for 5 minutes (300s) to balance security and performance
    try:
        await redis_client.setex(cache_key, 300, "1")
    except Exception as e:
        print(f"WARNING: Failed to set Redis enrollment cache: {e}")

async def ensure_batch_is_loaded(batch_id: str, db: AsyncSession):
    """
    Checks if the batch has its BM25/Graph metadata ready in Postgres.
    """
    # AC-26: Database Guard
    async with db_semaphore:
        result = await db.execute(select(Batch).where(Batch.id == batch_id))
        batch = result.scalars().first()
    
    if not batch or not batch.bm25_data or not batch.graph_data:
        print(f"INFO: Batch {batch_id} metadata missing. Building from Qdrant...")
        # 1. Scroll all chunks for this batch
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        response = await qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="batch_id", match=MatchValue(value=batch_id))]),
            limit=10000,
            with_payload=True
        )
        points, _ = response
        
        if not points:
            print(f"WARN: No data found in Qdrant for batch {batch_id}")
            return False

        chunks = [p.payload for p in points]
        
        # 2. Build Index and Graph (Offloaded to separate thread)
        new_bm25 = await asyncio.to_thread(BM25ChunkIndex, chunks)
        new_graph = await asyncio.to_thread(build_chunk_graph, chunks)
        
        # 3. Save to DB
        if not batch:
            batch = Batch(id=batch_id, name=f"Course {batch_id}")
            db.add(batch)
        
        batch.bm25_data = new_bm25.to_dict()
        batch.graph_data = new_graph
        await db.commit()
        
        print(f"SUCCESS: Built and persisted stateless metadata for batch {batch_id}.")
        return True
    
    return True

# ===== API MODELS =====
class UserLogin(BaseModel):
    username: str # Unified Username

class SwitchBatchRequest(BaseModel):
    username: str
    batch_id: str

class ChatRequest(BaseModel):
    username: str
    active_batch_id: str
    message: str = Field(..., min_length=1)
    complexity: str = Field("Undergrad", description="Complexity level: 5-Year-Old, High School, Undergrad, PhD Expert")
    tutor_mode: bool = Field(False, description="Enable Socratic Tutor mode")
    min_dense_score: float = Field(0.30, description="Strategy 2: Early Thresholding for dense matches")

COMPLEXITY_MAP = {
    "5-Year-Old": "Explain like I'm five. Use 1-syllable words and analogies, Use 30 – 60 words (1-2 short paragraphs) elaborate only when asked.",
    "High School": "Explain at a high school level. Use clear, accessible language and avoid overly technical jargon unless explained, Use 80 – 120 words (2-3 clear paragraphs) elaborate only when asked",
    "Undergrad": "Explain at a college undergraduate level. Provide a balanced, detailed, succinct overview with standard academic terminology, Use 150 – 250 words (Structured with headers) elaborate only when asked.",
    "PhD Expert": "Provide a highly technical, rigorous, and nuanced analysis suitable for a PhD expert. Use advanced terminology and address subtle complexities, Use 100 – 180 words (Highly Dense) elaborate only when asked."
}
TUTOR_PROMPT = "Use a Socratic tutoring style. Give the answer directly. Ask guiding questions to help the user discover the answer more detailed based on the context."

class ChatResponse(BaseModel):
    reply: str
    citations: list[str] | None = None
class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None

class YoutubeIngestRequest(BaseModel):
    url: str      
    batch_id: str = "default_batch"
    document_id: str | None = None
class WebIngestMode(str, Enum):
    SINGLE = "single"                                                                                           
    CRAWL = "crawl"                                                                                             
    SITEMAP = "sitemap"                                                                                                                                                                                            
class WebIngestRequest(BaseModel):                                                                              
    url: str                                                                                                    
    batch_id: str = "default_batch"                                                                           
    document_id: str | None = None                                                                              
    mode: WebIngestMode = WebIngestMode.SINGLE 
class UniversalURLRequest(BaseModel):
    url: str
    batch_id: str = "default_batch"
    document_id: str | None = None
    mode: str = "single"
    filename: Optional[str] = None
class BatchUrlRequest(BaseModel):
    urls: List[str]
    batch_id: str = "default_batch"

# ===== BACKGROUND WORKER & INGESTION =====
async def process_ingestion_background(job_id: str, input_path: str, file_type: str, ingestion_id: str, file_id: str, filename: str, batch_id: str, mode: str = "single"):
    try:
        from urllib.parse import urlparse
        import httpx
        
        # 1. SMART ROUTING FOR AUTO-URLS
        t_start = time.time()
        print(f"\n[INGESTION PHASE 1/6] Smart Routing & Download: {input_path}")
        if file_type == "url_auto":
            if "youtube.com" in input_path or "youtu.be" in input_path:
                file_type = "youtube"
            else:
                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                        resp = await client.head(input_path)
                        ct = resp.headers.get("Content-Type", "").lower()
                        if "application/pdf" in ct: file_type = ".pdf"
                        elif "application/vnd.openxmlformats-officedocument.presentationml.presentation" in ct: file_type = ".pptx"
                        elif "video/" in ct: file_type = ".mp4"
                        elif "text/html" in ct: file_type = "web"
                        else:
                            pure_path = urlparse(input_path).path.lower()
                            if pure_path.endswith(".pdf"): file_type = ".pdf"
                            elif pure_path.endswith((".pptx", ".ppt")): file_type = ".pptx"
                            elif pure_path.endswith((".mp4", ".mkv")): file_type = ".mp4"
                            elif pure_path.endswith((".xlsx", ".xls")): file_type = ".xlsx"
                            else: file_type = "web"
                except: file_type = "web"

            if file_type not in ["youtube", "web"]:
                async with AsyncSessionLocal() as db:
                    job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
                    job = job_result.scalars().first()
                    if job:
                        job.message = f"Downloading {file_type}..."
                        await db.commit()
                
                from urllib.parse import unquote
                url_filename = Path(urlparse(input_path).path).name or f"remote_file{file_type}"
                
                # BATCH ISOLATION: Nest file in batch directory
                batch_dir = UPLOAD_DIR / batch_id
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                local_file = batch_dir / f"{file_id}_{uuid4().hex[:6]}_{unquote(url_filename)}"
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    download_resp = await client.get(input_path)
                    download_resp.raise_for_status()
                    with open(local_file, "wb") as f: f.write(download_resp.content)
                input_path = str(local_file)
        t_routing = time.time()
        print(f"   -> Phase 1 Completed in {t_routing - t_start:.2f}s (Total: {t_routing - t_start:.2f}s)")

        print(f"[INGESTION PHASE 2/6] Extraction ({file_type})")
        blocks = []
        model_name = "Unknown"
        
        if file_type == "youtube":
            blocks = await asyncio.to_thread(ingest_youtube_video, input_path, file_id, model=get_whisper_model())
            model_name = "WhisperX"
        elif file_type == ".pdf":
            blocks = await asyncio.to_thread(ingest_pdf_layout_aware, Path(input_path), file_id)
            model_name = "PDF"
        elif file_type in [".pptx", ".ppt"]:
            blocks = await asyncio.to_thread(ingest_pptx_structure_aware, Path(input_path), file_id)
            model_name = "PPTX"
        elif file_type in [".xlsx", ".xls"]:
            blocks = await asyncio.to_thread(ingest_excel, Path(input_path), file_id)
            model_name = "Excel"
        elif file_type in [".mp4", ".mkv", ".mov"]:
            blocks = await asyncio.to_thread(ingest_video_file, Path(input_path), file_id, model=get_whisper_model())
            model_name = "WhisperX"
        elif file_type == "web":
            ingestor = WebIngestor()
            if mode == "crawl": raw_blocks = await asyncio.to_thread(ingestor.crawl, input_path)
            elif mode == "sitemap": raw_blocks = await asyncio.to_thread(ingestor.ingest_sitemap, input_path)
            else: raw_blocks = await asyncio.to_thread(ingestor.ingest, input_path)
            model_name = "Web"
            blocks = [{"block_id": rb.block_id, "block_type": rb.block_type, "text": rb.text, "document_id": file_id, "page": 1, "metadata": {**rb.metadata}} for rb in raw_blocks if rb.block_type != "error"]
        
        for b in blocks:
            b["batch_id"] = batch_id
            if "metadata" not in b: b["metadata"] = {}
            b["metadata"]["batch_id"] = batch_id
            b["metadata"]["filename"] = filename
            b["metadata"]["ingested_at"] = int(time.time())
        t_extraction = time.time()
        print(f"   -> Phase 2 Completed in {t_extraction - t_routing:.2f}s (Blocks: {len(blocks)}, Total: {t_extraction - t_start:.2f}s)")

        print(f"[INGESTION PHASE 3/6] Semantic Chunking")
        if blocks and blocks[0].get("is_prechunked"): chunks = blocks
        else: chunks = await asyncio.to_thread(semantic_chunk_blocks, blocks)
        
        for c in chunks:
            c["batch_id"] = batch_id
            if "metadata" not in c: c["metadata"] = {}
            c["metadata"]["batch_id"] = batch_id
            c["metadata"]["filename"] = filename
            c["metadata"]["ingested_at"] = int(time.time())
        t_chunking = time.time()
        print(f"   -> Phase 3 Completed in {t_chunking - t_extraction:.2f}s (Chunks: {len(chunks)}, Total: {t_chunking - t_start:.2f}s)")

        # Filter chunks that are eligible for embedding
        eligible_chunks = [c for c in chunks if c.get("embedding_eligible") and c.get("text")]
        texts = [c["text"] for c in eligible_chunks]
        
        print(f"[INGESTION PHASE 4/6] Embedding Generation")
        if texts:
            # Safety: Ensure collection exists before upsert
            collections = await qdrant.get_collections()
            if not any(c.name == QDRANT_COLLECTION for c in collections.collections):
                await qdrant.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config={"default": VectorParams(size=768, distance=Distance.COSINE)}
                )

            model = get_embedding_model()
            # Use to_thread for CPU bound embedding generation
            vectors = await asyncio.to_thread(model.encode, texts, batch_size=32)
            t_embedding = time.time()
            print(f"   -> Phase 4 Completed in {t_embedding - t_chunking:.2f}s (Embeddings: {len(texts)}, Total: {t_embedding - t_start:.2f}s)")

            print(f"[INGESTION PHASE 5/6] Vector Store Upsert")
            points = []
            for i, vec in enumerate(vectors):
                points.append(PointStruct(
                    id=int(uuid4().int >> 64), 
                    vector={"default": vec.tolist()}, 
                    payload=eligible_chunks[i]
                ))
            await qdrant.upsert(QDRANT_COLLECTION, points)
            t_upsert = time.time()
            print(f"   -> Phase 5 Completed in {t_upsert - t_embedding:.2f}s (Total: {t_upsert - t_start:.2f}s)")
        else:
            t_upsert = time.time()
            print(f"   -> Phase 4 & 5 Skipped (No eligible text, Total: {t_upsert - t_start:.2f}s)")
            
        # STATELESS TRANSITION: Re-build and Persist Metadata to Postgres
        # We fetch ALL chunks for this batch to ensure the BM25/Graph is complete
        print(f"[INGESTION PHASE 6/6] Metadata & Index Building")
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        response = await qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="batch_id", match=MatchValue(value=batch_id))]),
            limit=10000,
            with_payload=True
        )
        all_batch_points = response[0]
        all_batch_chunks = [p.payload for p in all_batch_points]
        
        async with AsyncSessionLocal() as db:
            if all_batch_chunks:
                new_bm25 = await asyncio.to_thread(BM25ChunkIndex, all_batch_chunks)
                new_graph = await asyncio.to_thread(build_chunk_graph, all_batch_chunks)
                
                result = await db.execute(select(Batch).where(Batch.id == batch_id))
                batch = result.scalars().first()
                if not batch:
                    batch = Batch(id=batch_id, name=f"Course {batch_id}")
                    db.add(batch)
                
                batch.bm25_data = new_bm25.to_dict()
                batch.graph_data = new_graph
                await db.commit()
                
                # IMS INVALIDATION: Clear RAM singleton so next request loads fresh data
                await index_manager.invalidate_batch(batch_id)
                
                # CACHE INVALIDATION: Clear Redis so chat uses the new index
                await redis_client.delete(f"batch_meta:{batch_id}")
                t_metadata = time.time()
                print(f"   -> Phase 6 Completed in {t_metadata - t_upsert:.2f}s (Total: {t_metadata - t_start:.2f}s)")
                print(f"SUCCESS: Ingestion total time: {t_metadata - t_start:.2f}s\n")


            # Update Celery Job Status
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.COMPLETED
                job.document_id = file_id
                await db.commit()

    except Exception as e:
        traceback.print_exc()
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.FAILED
                job.message = str(e)
                await db.commit()
    finally:
        if mode == "single":
            from database import _db_proxy
            await _db_proxy.get_engine().dispose()

async def process_batch_ingestion_background(job_id: str, urls: List[str], batch_id: str):
    try:
        for i, url in enumerate(urls):
            await process_ingestion_background(job_id, url, "url_auto", uuid4().hex[:8], f"url_{uuid4().hex[:8]}", url, batch_id, mode="batch")
        
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.COMPLETED
                await db.commit()
    except Exception as e:
        traceback.print_exc()
        async with AsyncSessionLocal() as db:
            job_result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
            job = job_result.scalars().first()
            if job:
                job.status = JobStatus.FAILED
                job.message = f"Batch failed: {str(e)}"
                await db.commit()
    finally:
        from database import _db_proxy
        await _db_proxy.get_engine().dispose()

@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    if job_id == "cached":
        return JobResponse(job_id="cached", status=JobStatus.COMPLETED, message="Returned from cache")
        
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    job = result.scalars().first()
    if not job: raise HTTPException(404, "Job not found")
    return JobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        message=job.message,
        document_id=job.document_id,
        filename=job.filename
    )

@app.post("/ingest/youtube", response_model=JobResponse)
async def ingest_youtube(req: YoutubeIngestRequest, db: AsyncSession = Depends(get_db)):
    job_id = f"job_{uuid4().hex[:8]}"
    file_id = req.document_id or f"yt_{uuid4().hex[:8]}"
    
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename=req.url, document_id=file_id)
    db.add(new_job)
    await db.commit()
    
    from celery_app import celery_app
    celery_app.send_task("process_ingestion", args=[job_id, req.url, "youtube", uuid4().hex[:8], file_id, req.url, req.batch_id, "single"])
    
    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename=req.url)

@app.post("/ingest/web", response_model=JobResponse)
async def ingest_web(req: WebIngestRequest, db: AsyncSession = Depends(get_db)):
    job_id = f"job_{uuid4().hex[:8]}"
    file_id = req.document_id or f"web_{uuid4().hex[:8]}"
    
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename=req.url, document_id=file_id)
    db.add(new_job)
    await db.commit()
    
    from celery_app import celery_app
    celery_app.send_task("process_ingestion", args=[job_id, req.url, "web", uuid4().hex[:8], file_id, req.url, req.batch_id, req.mode.value])
    
    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename=req.url)

@app.post("/ingest/url", response_model=JobResponse)
async def ingest_universal_url(req: UniversalURLRequest, db: AsyncSession = Depends(get_db)):
    job_id = f"job_{uuid4().hex[:8]}"
    file_id = req.document_id or f"url_{uuid4().hex[:8]}"
    filename = req.filename or req.url
    
    # 1. Create DB Record
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename=filename, document_id=file_id)
    db.add(new_job)
    await db.commit()
    
    # 2. Dispatch to Celery Queue
    from celery_app import celery_app
    celery_app.send_task("process_ingestion", args=[job_id, req.url, "url_auto", uuid4().hex[:8], file_id, filename, req.batch_id, req.mode])
    
    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename=filename)

@app.post("/ingest/batch-urls", response_model=JobResponse)
async def ingest_batch_urls(req: BatchUrlRequest, db: AsyncSession = Depends(get_db)):
    job_id = f"batch_{uuid4().hex[:8]}"
    
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename="Batch")
    db.add(new_job)
    await db.commit()
    
    from celery_app import celery_app
    celery_app.send_task("process_batch_ingestion", args=[job_id, req.urls, req.batch_id])
    
    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename="Batch")

@app.post("/upload", response_model=JobResponse)
async def upload_file(file: UploadFile = File(...), batch_id: str = "default_batch", db: AsyncSession = Depends(get_db)):
    job_id = f"job_{uuid4().hex[:8]}"
    file_id = str(uuid4())
    
    # BATCH ISOLATION: Create subdirectory
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = batch_dir / f"{file_id}_{file.filename}"
    with open(file_path, "wb") as f: f.write(await file.read())
    
    new_job = IngestionJob(id=job_id, status=JobStatus.PROCESSING.value, filename=file.filename, document_id=file_id)
    db.add(new_job)
    await db.commit()
    
    from celery_app import celery_app
    celery_app.send_task("process_ingestion", args=[job_id, str(file_path), Path(file.filename).suffix.lower(), uuid4().hex[:8], file_id, file.filename, batch_id, "single"])
    
    return JobResponse(job_id=job_id, status=JobStatus.PROCESSING, filename=file.filename)

# ===== AUTH ENDPOINTS =====
@app.post("/auth/login")
async def login(req: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please wait for sync.")
    
    # Enrich batch data with friendly names
    e_res = await db.execute(
        select(Batch.id, Batch.name)
        .join(UserEnrollment, UserEnrollment.batch_id == Batch.id)
        .where(UserEnrollment.username == user.username)
    )
    enrolled_data = [{"id": r[0], "name": r[1]} for r in e_res.all()]
    batch_ids = [c["id"] for c in enrolled_data]
    
    # If user has no enrollments yet, we return empty list
    active_batch = user.last_active_batch_id
    if not active_batch and batch_ids:
        active_batch = batch_ids[0]
        
    return {
        "status": "success", 
        "username": user.username, 
        "active_batch_id": active_batch, 
        "enrolled_batches": enrolled_data
    }

@app.post("/auth/switch-batch")
async def switch_batch(req: SwitchBatchRequest, db: AsyncSession = Depends(get_db)):
    # 1. Verify enrollment exists
    e_res = await db.execute(select(UserEnrollment).where(UserEnrollment.username == req.username, UserEnrollment.batch_id == req.batch_id))
    if not e_res.scalars().first():
        raise HTTPException(status_code=403, detail="User not enrolled in this batch")
    
    # 2. Update last active
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.last_active_batch_id = req.batch_id
    await db.commit()
    
    return {"status": "success", "active_batch_id": req.batch_id}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    t_start = time.time()
    print(f"\n[INFERENCE] Starting Chat Inference for user {req.username}")
    
    # AC-45: Rate Limit - Check for 3min cooldown
    await check_cooldown(req.active_batch_id, req.message, "chat")

    await verify_user_enrollment(req.username, req.active_batch_id, db)
    
    # 1. Ensure metadata is ready
    if not await ensure_batch_is_loaded(req.active_batch_id, db):
        return ChatResponse(reply="Currently we do not have the specific resources to answer this question.", citations=[])
    
    # --- TIER 1: FULL RESPONSE CACHE (LLM BYPASS) ---
    response_filters = {"complexity": req.complexity, "tutor_mode": str(req.tutor_mode)}
    cached_response = await get_semantic_cache(req.active_batch_id, req.message, "chat_response", filters=response_filters)
    
    if cached_response:
        print(f"   -> [TIER 1 HIT] Inference Bypassed via Response Cache in {time.time() - t_start:.2f}s")
        return ChatResponse(reply=cached_response["reply"], citations=cached_response.get("citations", []))

    # --- TIER 2: KNOWLEDGE CONTEXT CACHE (RETRIEVAL BYPASS) ---
    candidates = None
    cached_context = await get_semantic_cache(req.active_batch_id, req.message, "chat_context")
    
    if cached_context:
        print(f"   -> [TIER 2 HIT] Retrieval Bypassed via Context Cache")
        candidates = cached_context
    else:
        # 2. Fetch Metadata (IMS RAM Brain)
        brain = await index_manager.get_brain(req.active_batch_id)
        if not brain:
            return ChatResponse(reply="Currently we do not have the specific resources to answer this question.", citations=[])
        
        bm25 = brain["bm25"]
        graph = brain["graph"]

        # 3. Full Retrieval Pipeline
        print(f"[INFERENCE PHASE 2/4] Query Rewriting")
        rewrite_result = await rewrite_query_ensemble(query=req.message, call_llm_fn=call_gemini_async)
        
        print(f"[INFERENCE PHASE 3/4] Retrieval & Reranking")
        candidates = await retrieve_candidates(
            query=req.message, 
            rewrites=rewrite_result["rewrites"], 
            bm25=bm25,
            graph=graph,
            chunk_fetcher=lambda ids: fetch_chunk_details(ids, req.active_batch_id),
            dense_fn=dense_retrieve, 
            batch_id=req.active_batch_id, 
            min_dense_score=req.min_dense_score,
            t_inference_start=t_start
        )
        
        if candidates:
            # Save to Context Cache (Background) - Available for ANY complexity level
            asyncio.create_task(set_semantic_cache(req.active_batch_id, req.message, "chat_context", candidates))

    if not candidates: 
        return ChatResponse(reply="Currently we do not have the answer for this question.", citations=[])
    
    # 4. Response Generation
    print(f"[INFERENCE PHASE 4/4] Response Generation (Complexity: {req.complexity})")
    complexity_instr = COMPLEXITY_MAP.get(req.complexity, COMPLEXITY_MAP["Undergrad"])
    tutor_instr = TUTOR_PROMPT if req.tutor_mode else "Answer clearly based on the context."
    context_text = "\n\n".join(f"[DOC_ID: {c['doc_id']}]\n{c['text']}" for c in candidates)
    
    final_prompt = f"""
Instructions:
- {complexity_instr}
- {tutor_instr}
- Use only the provided context to answer the question. Do not use any external knowledge.
- If context is insufficient, say only "Currently we do not have the answer for this question."
- Always cite your sources by [DOC_ID: id].
Context:
{context_text}
Question: {req.message}
""".strip()

    reply = await call_gemini_async(final_prompt)
    
    # Extract citations
    doc_ids = list(set([c['doc_id'] for c in candidates]))
    found_citations = [did for did in doc_ids if f"[DOC_ID: {did}]" in reply]
    
    # --- SAVE TO TIER 1 CACHE (Background) ---
    NO_INFO_MSG = "Currently we do not have the answer for this question."
    if reply.strip() != NO_INFO_MSG:
        asyncio.create_task(set_semantic_cache(
            req.active_batch_id, 
            req.message, 
            "chat_response", 
            {"reply": reply, "citations": found_citations},
            metadata={"complexity": req.complexity, "tutor_mode": str(req.tutor_mode)}
        ))
    else:
        await set_cooldown(req.active_batch_id, req.message, "chat")

    print(f"SUCCESS: Total Inference Time: {time.time() - t_start:.2f}s\n")
    return ChatResponse(reply=reply, citations=found_citations)