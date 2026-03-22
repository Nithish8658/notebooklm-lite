from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
import uuid
import datetime
import os
from infrastructure import get_infra

# Database URL for PostgreSQL in Docker
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://notebook:notebook@localhost:5432/notebooklm"
)

# --- SOPHISTICATED DYNAMIC SESSION ACCESS ---
class AsyncSessionLocalProxy:
    """
    Acts as a proxy for the AsyncSessionLocal sessionmaker.
    Delegates to the infrastructure container assigned to the current event loop.
    """
    def __call__(self, **kwargs):
        return get_infra().db_sessionmaker(**kwargs)

AsyncSessionLocal = AsyncSessionLocalProxy()

Base = declarative_base()

from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

# ... (Models User, Batch, UserEnrollment, Flashcard, Quiz, Podcast defined below)

class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True) # Now the primary identifier
    role = Column(String, nullable=True) # learner or mentor
    last_active_batch_id = Column(String, nullable=True) # Remembers user's current context
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Batch(Base):
    __tablename__ = "batches"
    id = Column(String, primary_key=True, index=True) # e.g. "CS101"
    name = Column(String, nullable=False) # e.g. "Computer Science 101"
    graph_data = Column(JSON, nullable=True) # Persisted Knowledge Graph
    bm25_data = Column(JSON, nullable=True)  # Persisted BM25 Statistics (IDF, etc.)
    integrity_status = Column(String, nullable=False, default="HEALTHY") # HEALTHY, REPAIR_REQUIRED, REPAIRING
    last_repair_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class UserEnrollment(Base):
    """Link table for Many-to-Many relationship"""
    __tablename__ = "user_enrollments"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), index=True)
    batch_id = Column(String, ForeignKey("batches.id", ondelete="CASCADE"), index=True)
    enrolled_at = Column(DateTime, default=datetime.datetime.utcnow)

class Flashcard(Base):
    __tablename__ = "flashcards"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, index=True, nullable=False)
    batch_id = Column(String, index=True, nullable=False)
    complexity = Column(String, nullable=False, default="Undergrad")
    payload = Column(JSON, nullable=False) # Stores the array of flashcards
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Quiz(Base):
    __tablename__ = "quizzes"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, index=True, nullable=False)
    batch_id = Column(String, index=True, nullable=False)
    complexity = Column(String, nullable=False, default="Undergrad")
    payload = Column(JSON, nullable=False) # Stores the quiz questions
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Podcast(Base):
    __tablename__ = "podcasts"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, index=True, nullable=False)
    batch_id = Column(String, index=True, nullable=False)
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
    
    # Manual Migration for existing 'batches' table if it was created before stateless refactor
    with sync_engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS graph_data JSON;"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS bm25_data JSON;"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS integrity_status VARCHAR DEFAULT 'HEALTHY';"))
            conn.execute(text("ALTER TABLE batches ADD COLUMN IF NOT EXISTS last_repair_at TIMESTAMP;"))
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