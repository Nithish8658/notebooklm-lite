from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import uuid
import datetime

import os

import asyncio
import weakref

# Database URL for PostgreSQL in Docker
# Using postgresql+asyncpg for async support
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://notebook:notebook@localhost:5432/notebooklm"
)

class AsyncDatabaseProxy:
    def __init__(self, url):
        self._url = url
        # Use WeakKeyDictionary to allow loops to be GC'd
        self._engines = weakref.WeakKeyDictionary()
        self._sessionmakers = weakref.WeakKeyDictionary()
        # Fallback for when there's no running loop (e.g. during import)
        self._fallback_engine = None
        self._fallback_sessionmaker = None

    def get_engine(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._fallback_engine is None:
                self._fallback_engine = create_async_engine(
                    self._url,
                    pool_pre_ping=True,
                    pool_size=20,
                    max_overflow=10,
                    pool_recycle=3600,
                )
            return self._fallback_engine
        
        if loop not in self._engines:
            self._engines[loop] = create_async_engine(
                self._url,
                pool_pre_ping=True,
                pool_size=200,       # AC-23: Increased from 100 for massive concurrency
                max_overflow=100,    # AC-23: Increased from 50
                pool_recycle=3600,
            )
        return self._engines[loop]

    def get_sessionmaker(self):
        try:
            loop = asyncio.get_running_loop()
            engine = self.get_engine()
            if loop not in self._sessionmakers:
                self._sessionmakers[loop] = async_sessionmaker(
                    bind=engine, 
                    class_=AsyncSession, 
                    expire_on_commit=False,
                    autocommit=False,
                    autoflush=False
                )
            return self._sessionmakers[loop]
        except RuntimeError:
            if self._fallback_sessionmaker is None:
                engine = self.get_engine()
                self._fallback_sessionmaker = async_sessionmaker(
                    bind=engine, 
                    class_=AsyncSession, 
                    expire_on_commit=False,
                    autocommit=False,
                    autoflush=False
                )
            return self._fallback_sessionmaker

    def __call__(self, **kwargs):
        """Allows using AsyncSessionLocal() as before."""
        return self.get_sessionmaker()(**kwargs)

    def dispose_current(self):
        """Optional: call this to cleanup the engine for the current loop."""
        try:
            loop = asyncio.get_running_loop()
            if loop in self._engines:
                # We can't easily await here if called from sync finally,
                # but we can remove it from our cache at least.
                # The caller should ideally call await proxy.get_engine().dispose()
                pass
        except RuntimeError:
            pass

_db_proxy = AsyncDatabaseProxy(SQLALCHEMY_DATABASE_URL)
engine = _db_proxy.get_engine() # For backward compatibility where used directly
AsyncSessionLocal = _db_proxy

Base = declarative_base()

from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

# ... (Models User, Cohort, UserEnrollment, Flashcard, Quiz, Podcast unchanged)

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False) # Stores Username
    role = Column(String, nullable=True) # learner or mentor
    last_active_cohort_id = Column(String, nullable=True) # Remembers user's current context
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Cohort(Base):
# ... (rest of the models remain same until init_db)
    __tablename__ = "cohorts"
    id = Column(String, primary_key=True, index=True) # e.g. "CS101"
    name = Column(String, nullable=False) # e.g. "Computer Science 101"
    graph_data = Column(JSON, nullable=True) # Persisted Knowledge Graph
    bm25_data = Column(JSON, nullable=True)  # Persisted BM25 Statistics (IDF, etc.)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class UserEnrollment(Base):
    """Link table for Many-to-Many relationship"""
    __tablename__ = "user_enrollments"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    cohort_id = Column(String, ForeignKey("cohorts.id", ondelete="CASCADE"), index=True)
    enrolled_at = Column(DateTime, default=datetime.datetime.utcnow)

class Flashcard(Base):
    __tablename__ = "flashcards"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    cohort_id = Column(String, index=True, nullable=False)
    complexity = Column(String, nullable=False, default="Undergrad")
    payload = Column(JSON, nullable=False) # Stores the array of flashcards
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Quiz(Base):
    __tablename__ = "quizzes"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    cohort_id = Column(String, index=True, nullable=False)
    complexity = Column(String, nullable=False, default="Undergrad")
    payload = Column(JSON, nullable=False) # Stores the quiz questions
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Podcast(Base):
    __tablename__ = "podcasts"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    cohort_id = Column(String, index=True, nullable=False)
    complexity = Column(String, nullable=False, default="Undergrad")
    topic = Column(String, nullable=True)
    filename = Column(String, nullable=False)
    transcript = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id = Column(String, primary_key=True, index=True) # The job_id
    status = Column(String, nullable=False, default="processing")
    message = Column(String, nullable=True)
    document_id = Column(String, nullable=True)
    filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create all tables (Synchronous helper for startup)
def init_db():
    from sqlalchemy import create_engine, text
    # Fallback to sync engine just for table creation at startup
    sync_url = SQLALCHEMY_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    sync_engine = create_engine(sync_url)
    
    print("DB: Ensuring all tables exist...")
    Base.metadata.create_all(bind=sync_engine)
    
    # Manual Migration for existing 'cohorts' table if it was created before stateless refactor
    with sync_engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS graph_data JSON;"))
            conn.execute(text("ALTER TABLE cohorts ADD COLUMN IF NOT EXISTS bm25_data JSON;"))
            conn.execute(text("ALTER TABLE podcasts ADD COLUMN IF NOT EXISTS topic VARCHAR;"))
            
            # Add complexity column to Studio tables
            conn.execute(text("ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS complexity VARCHAR DEFAULT 'Undergrad';"))
            conn.execute(text("ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS complexity VARCHAR DEFAULT 'Undergrad';"))
            conn.execute(text("ALTER TABLE podcasts ADD COLUMN IF NOT EXISTS complexity VARCHAR DEFAULT 'Undergrad';"))
            
            # Username-only transition: Add role, remove hashed_password
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR;"))
            conn.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS hashed_password;"))
            
            # TRUNCATE tables to prepare for new platform sync (Optional: Remove if you want to keep existing users)
            # conn.execute(text("TRUNCATE TABLE user_enrollments CASCADE;"))
            # conn.execute(text("TRUNCATE TABLE users CASCADE;"))
            
            conn.commit()
            print("DB: Migration (role, remove password) applied.")
        except Exception as e:
            print(f"DB: Migration note (can usually ignore): {e}")

async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()
