import os
import time
import json
import sqlite3
import requests
import logging
import asyncio
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Database imports for PostgreSQL sync
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database import AsyncSessionLocal, User, Batch, UserEnrollment
from sqlalchemy import select, delete
from infrastructure import qdrant, redis_client

# Load env vars for configuration
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PlatformSync")

# --- CONFIGURATION ---
PLATFORM_BASE_URL = os.getenv("PLATFORM_BASE_URL", "http://34.122.167.196:4000/v1")
PLATFORM_AUTH_URL = os.getenv("PLATFORM_AUTH_URL", "http://34.122.167.196:4000/v1/login")
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# New Endpoints
GET_LEARNERS_URL = f"{PLATFORM_BASE_URL}/getlearners"
GET_MENTORS_URL = f"{PLATFORM_BASE_URL}/getmentors"
GET_BATCHES_URL = f"{PLATFORM_BASE_URL}/getbatches"
LEARNER_PROFILE_URL = f"{PLATFORM_BASE_URL}/learnerProfile/"
MENTOR_PROFILE_URL = f"{PLATFORM_BASE_URL}/mentorProfile/"

# Credentials from .env
EMAIL = os.getenv("PLATFORM_EMAIL")
PASSWORD = os.getenv("PLATFORM_PASSWORD")

# Absolute path for SQLite DB to ensure consistency across execution contexts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "platform_sync.db") 
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "120")) # Default 2 minutes
class PlatformSyncService:
    def __init__(self):
        self.access_token = None
        self.batch_name_map = {} # Lookup for batchId -> courseName
        self.current_cycle_id = None
        self._init_db()

    def _log(self, msg, level=logging.INFO, *args):
        prefix = f"[Cycle: {self.current_cycle_id}] " if self.current_cycle_id else ""
        logger.log(level, f"{prefix}{msg}", *args)

    def _init_db(self):
        """Initializes deduplication database for Files."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_id TEXT PRIMARY KEY,
                    url TEXT,
                    batch_id TEXT,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def login(self):
        """Phase 0: Authenticate and get Access Token."""
        user_identifier = EMAIL or os.getenv("PLATFORM_USERNAME")
        if not user_identifier or not PASSWORD:
            logger.error("Credentials (Email/Username and Password) not set in environment!")
            return False

        try:
            logger.info(f"Attempting login for: {user_identifier}...")
            login_headers = {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/json",
                "X-Source": "web",
                "X-Device-Fingerprint": "cf6428af1b4ea26b020acc8a902156e9"
            }
            
            id_key = "username" if os.getenv("PLATFORM_USE_USERNAME") == "true" else "email"
            payload = {id_key: user_identifier, "password": PASSWORD}
            
            org_id = os.getenv("PLATFORM_ORG_ID")
            if org_id: payload["organizationId"] = org_id

            response = requests.post(PLATFORM_AUTH_URL, json=payload, headers=login_headers, timeout=15)
            if response.status_code != 200:
                logger.error(f"Login failed (Status {response.status_code}): {response.text}")
                return False

            res_data = response.json()
            if res_data.get("status") == "success":
                self.access_token = res_data["data"]["accessToken"]
                logger.info("Login successful. Token acquired.")
                return True
            return False
        except Exception as e:
            logger.error(f"Login exception: {e}")
            return False

    def get_headers(self):
        return {
            "Accept": "application/json, text/plain, */*",
            "Authorization": f"Bearer {self.access_token}",
            "X-Source": "web",
            "X-Device-Fingerprint": "cf6428af1b4ea26b020acc8a902156e9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        }

    async def sync_users_and_enrollments(self):
        """Syncs all Learners and Mentors from Platform to PostgreSQL."""
        logger.info("--- SYNCING USERS & ENROLLMENTS ---")
        seen_usernames = set()
        
        # 1. Sync Learners
        learners = self._fetch_all(GET_LEARNERS_URL)
        logger.info(f"Fetched {len(learners)} learners from platform.")
        for l in learners:
            l_id = l.get("id")
            profile = self._fetch_one(f"{LEARNER_PROFILE_URL}{l_id}")
            if profile:
                username = await self._upsert_user_data(profile, "learner")
                if username: seen_usernames.add(username)

        # 2. Sync Mentors
        mentors = self._fetch_all(GET_MENTORS_URL)
        logger.info(f"Fetched {len(mentors)} mentors from platform.")
        for m in mentors:
            m_id = m.get("id")
            profile = self._fetch_one(f"{MENTOR_PROFILE_URL}{m_id}")
            if profile:
                username = await self._upsert_user_data(profile, "mentor")
                if username: seen_usernames.add(username)
        
        # --- GLOBAL USER PURGE ---
        if seen_usernames:
            async with AsyncSessionLocal() as db:
                res = await db.execute(
                    select(User).where(User.username.not_in(list(seen_usernames)))
                )
                to_delete = res.scalars().all()
                if to_delete:
                    logger.info(f"Purging {len(to_delete)} stale users.")
                    for u in to_delete:
                        await db.delete(u)
                    await db.commit()
        logger.info("--- USER SYNC COMPLETE ---")

        logger.info("--- USER SYNC COMPLETE ---")

    def _fetch_all(self, url):
        try:
            resp = requests.get(url, headers=self.get_headers(), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
            return []
        except Exception as e:
            logger.error(f"Error fetching from {url}: {e}")
            return []

    def _fetch_one(self, url):
        try:
            resp = requests.get(url, headers=self.get_headers(), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data")
            return None
        except Exception as e:
            logger.error(f"Error fetching from {url}: {e}")
            return None

    async def _upsert_user_data(self, profile_data, role):
        """Upserts User, Batches, and Enrollments into PostgreSQL."""
        # Note: Platform profile is nested: {"learner": {...}, "enrolledBatches": [...]} 
        # OR for mentor: {"mentor": {...}, "enrolledBatches": [...]}
        
        user_info = profile_data.get("learner") or profile_data.get("mentor")
        if not user_info:
            logger.warning(f"Could not find user info in profile data for role {role}")
            return

        username = user_info.get("username")
        if not username:
            logger.warning(f"Profile for {role} has no username: {user_info.get('id')}")
            return

        enrolled_batches = profile_data.get("enrolledBatches", [])
        logger.info(f"Syncing {role}: {username} (Enrolled in {len(enrolled_batches)} batches)")
        
        async with AsyncSessionLocal() as db:
            # 1. Upsert User
            res = await db.execute(select(User).where(User.username == username))
            user = res.scalars().first()
            if not user:
                logger.info(f"Creating new user: {username}")
                user = User(username=username, role=role)
                db.add(user)
                await db.flush()
            else:
                user.role = role # Update role if changed
            
            # 2. Sync Batches and Enrollments
            # First, get existing enrollments to avoid duplicates
            e_res = await db.execute(select(UserEnrollment).where(UserEnrollment.username == user.username))
            existing_batch_ids = {e.batch_id for e in e_res.scalars().all()}
            
            # Extract cloud state
            cloud_batch_ids = set()
            for batch in enrolled_batches:
                b_id = batch.get("batchId") or batch.get("id") # Try both
                if not b_id: continue
                cloud_batch_ids.add(b_id)
                
                # ENRICHMENT: Use the master name map for the Batch name, fallback to platform provided names
                b_name = self.batch_name_map.get(b_id) or batch.get("courseName") or batch.get("batchName") or f"Course {b_id}"
                
                # Upsert Batch
                c_res = await db.execute(select(Batch).where(Batch.id == b_id))
                batch = c_res.scalars().first()
                if not batch:
                    logger.info(f"Creating new batch: {b_name} ({b_id})")
                    db.add(Batch(id=b_id, name=b_name))
                else:
                    # Update name if it changed or was previously an ID
                    if batch.name != b_name:
                        logger.info(f"Updating batch name: {batch.name} -> {b_name}")
                        batch.name = b_name
                
                # Add Enrollment if missing
                if b_id not in existing_batch_ids:
                    logger.info(f"Enrolling {username} in {b_id}")
                    db.add(UserEnrollment(username=user.username, batch_id=b_id))
            
            # --- PURGE PHASE: Remove stale local enrollments ---
            stale_enrollments = existing_batch_ids - cloud_batch_ids
            if stale_enrollments:
                logger.info(f"Purging {len(stale_enrollments)} stale enrollments for {username}")
                await db.execute(
                    delete(UserEnrollment)
                    .where(UserEnrollment.username == user.username)
                    .where(UserEnrollment.batch_id.in_(list(stale_enrollments)))
                )
            
            await db.commit()

    def get_live_batches(self):
        """Fetches all batches/batches marked as 'live'."""
        try:
            url = f"{PLATFORM_BASE_URL}/getbatches"
            response = requests.get(url, headers=self.get_headers(), timeout=15)
            if response.status_code == 401:
                if self.login(): return self.get_live_batches()
                return []
            data = response.json()
            if data.get("status") == "success":
                return [b for b in data.get("data", []) if b.get("batchType") == "live"]
            return []
        except Exception as e:
            logger.error(f"Error fetching batches: {e}")
            return []

    def get_batch_files(self, batch_id):
        """Fetches all files associated with a specific batch ID."""
        try:
            # Note: External platform API still uses 'cohorts' in the URL path
            url = f"{PLATFORM_BASE_URL}/cohorts/files/{batch_id}"
            response = requests.get(url, headers=self.get_headers(), timeout=15)
            if response.status_code == 401:
                if self.login(): return self.get_batch_files(batch_id)
                return []
            
            if response.status_code != 200:
                logger.error(f"Error fetching files for batch {batch_id}: Status {response.status_code}, Body: {response.text[:200]}")
                return []

            try:
                data = response.json()
            except Exception as je:
                logger.error(f"JSON Parse Error for batch {batch_id}: {je}. Body: {response.text[:200]}")
                return []

            if data.get("status") == "success":
                return data.get("data", [])
            return []
        except Exception as e:
            logger.error(f"Request Error fetching files for batch {batch_id}: {e}")
            return []

    def get_all_local_ids_for_batch(self, batch_id):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT file_id FROM processed_files WHERE batch_id = ?", (batch_id,))
            return [row[0] for row in cursor.fetchall()]

    def trigger_ingestion(self, file_url, batch_id, file_name, file_id):
        try:
            payload = {
                "url": file_url,
                "batch_id": batch_id,
                "mode": "single",
                "filename": file_name,
                "document_id": file_id
            }
            resp = requests.post(f"{RAG_API_URL}/ingest/url", json=payload, timeout=20)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"RAG Ingestion Error: {e}")
            return False

    def trigger_deletion(self, file_id):
        try:
            resp = requests.delete(f"{RAG_API_URL}/sources/{file_id}", timeout=20)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"RAG Deletion Error: {e}")
            return False

    def mark_processed(self, file_id, url, batch_id):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR REPLACE INTO processed_files (file_id, url, batch_id) VALUES (?, ?, ?)", 
                         (file_id, url, batch_id))
            conn.commit()

    def remove_from_local_db(self, file_id):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM processed_files WHERE file_id = ?", (file_id,))
            conn.commit()

    async def run_sync_cycle(self):
        """Main loop: Detects new users, files to ingest and stale files to purge."""
        self.current_cycle_id = uuid.uuid4().hex[:8]
        lock_key = "lock:platform_sync_cycle"
        
        # AC-60: Distributed Lock to prevent overlapping runs
        lock_acquired = await redis_client.set(lock_key, self.current_cycle_id, nx=True, ex=600)
        
        if not lock_acquired:
            existing_cycle = await redis_client.get(lock_key)
            logger.info(f"SYNC SKIP: Another cycle ({existing_cycle}) is already in progress. Exiting.")
            return

        try:
            self._log("--- STARTING SYNC & PURGE CYCLE ---")
            if not self.access_token and not self.login():
                self._log("Authentication failed. Skipping cycle.", logging.ERROR)
                return

            # 0. Build Master Batch Name Map (Friendly names for batches)
            self._log("Fetching Master Batch metadata...")
            batches = self._fetch_all(GET_BATCHES_URL)
            self.batch_name_map = {
                b["batchId"]: b.get("courseName") or b.get("batchName") or f"Course {b['batchId']}" 
                for b in batches if "batchId" in b
            }
            self._log(f"Master Batch Map built with {len(self.batch_name_map)} entries.")

            # 1. Sync Users and Enrollments
            await self.sync_users_and_enrollments()

            # --- GLOBAL BATCH PURGE: Remove batches no longer on the platform ---
            valid_batch_ids = set(self.batch_name_map.keys())
            if valid_batch_ids:
                async with AsyncSessionLocal() as db:
                    res = await db.execute(select(Batch))
                    local_batches = res.scalars().all()
                    for batch in local_batches:
                        if batch.id not in valid_batch_ids:
                            self._log(f"Purging dead batch: {batch.name} ({batch.id})")
                            
                            # A. Remove from Qdrant
                            try:
                                from qdrant_client.models import Filter, FieldCondition, MatchValue
                                await qdrant.delete(
                                    collection_name="document_chunks",
                                    points_selector=Filter(must=[FieldCondition(key="batch_id", match=MatchValue(value=batch.id))])
                                )
                            except Exception as e: self._log(f"Failed to purge Qdrant for batch {batch.id}: {e}", logging.ERROR)

                            # B. Remove Physical Files
                            import shutil
                            batch_dir = os.path.join("backend", "uploads", batch.id)
                            if os.path.exists(batch_dir):
                                try:
                                    shutil.rmtree(batch_dir)
                                    self._log(f"Deleted files for batch {batch.id}")
                                except Exception as e: self._log(f"Failed to delete files for {batch.id}: {e}", logging.ERROR)

                            # C. Remove from DB
                            await db.delete(batch)
                    await db.commit()

            # 2. Sync Files (Existing logic)
            self._log("--- SYNCING FILES ---")
            live_batches = self.get_live_batches()
            self._log(f"Found {len(live_batches)} 'live' batches for file sync.")
            
            for batch in live_batches:
                batch_id = batch["batchId"]
                batch_name = self.batch_name_map.get(batch_id, batch_id)
                current_platform_files = self.get_batch_files(batch_id)
                platform_file_map = {f["id"]: f for f in current_platform_files if not f.get("isDeleted")}
                platform_ids = set(platform_file_map.keys())
                
                self._log(f"Batch {batch_name} ({batch_id}): {len(platform_ids)} active files on platform.")
                
                local_stored_ids = self.get_all_local_ids_for_batch(batch_id)
                
                # A. Deletion
                for file_id in local_stored_ids:
                    if file_id not in platform_ids:
                        self._log(f"Detected deleted file {file_id} in batch {batch_id}. Triggering RAG purge...")
                        if self.trigger_deletion(file_id):
                            self.remove_from_local_db(file_id)

                # B. Ingestion
                for file_id, f in platform_file_map.items():
                    if file_id not in local_stored_ids:
                        file_name = f.get("fileName", "Unnamed Resource")
                        self._log(f"Detected new file '{file_name}' ({file_id}) in batch {batch_id}. Triggering RAG ingestion...")
                        if self.trigger_ingestion(f["fileUrl"], batch_id, file_name, file_id):
                            self.mark_processed(file_id, f["fileUrl"], batch_id)
            
            self._log("--- SYNC & PURGE CYCLE COMPLETE ---")
        finally:
            # Release lock safely
            try:
                current_lock_val = await redis_client.get(lock_key)
                if current_lock_val == self.current_cycle_id:
                    await redis_client.delete(lock_key)
            except:
                pass

    async def start(self):
        """Entry point for standalone execution (python backend/services/platform_sync.py)."""
        logger.info(f"Platform Sync Service (User + File) active (Interval: {POLL_INTERVAL}s)")
        from infrastructure import init_infra, close_infra
        await init_infra()
        try:
            while True:
                try:
                    await self.run_sync_cycle()
                except Exception as e:
                    logger.error(f"CRITICAL: Sync loop crashed: {e}")
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            await close_infra()

if __name__ == "__main__":
    service = PlatformSyncService()
    asyncio.run(service.start())