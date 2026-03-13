import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import os

from .youtube_ingestor_whisperx import (
    transcribe_with_whisperx,
    flatten_words,
    normalize_segments,
    create_blocks
)

logger = logging.getLogger(__name__)

def ingest_video_file(file_path: Path, document_id: str, model=None) -> List[Dict]:
    """
    Ingests a local video file, transcribes it, and produces RAG-compatible blocks.
    Reuses the WhisperX pipeline from the YouTube ingestor.

    Args:
        file_path: Path to the local video file.
        document_id: A unique ID for the document.
        model: Optional pre-loaded WhisperX model.

    Returns:
        List[Dict]: A list of block dictionaries.
    """
    # Create a temporary directory if we need to do any pre-processing
    # though whisperx.load_audio handles most video formats via ffmpeg directly.
    logger.info(f"Starting local video ingestion for {document_id}: {file_path.name}")
    
    try:
        # 1. Transcribe
        # WhisperX load_audio uses ffmpeg to read audio from video files directly
        logger.info("Phase 1: Transcribing & Diarizing local video...")
        raw_result = transcribe_with_whisperx(file_path, model=model)
        
        # 2. Normalize
        logger.info("Phase 2: Normalizing segments...")
        flat_words = flatten_words(raw_result)
        normalized_segments = normalize_segments(flat_words)
        
        # 3. Generate Blocks
        logger.info("Phase 3: Generating blocks...")
        blocks = create_blocks(normalized_segments, document_id, source_tag="video_upload")
        
        logger.info(f"Ingestion complete. Generated {len(blocks)} blocks for {file_path.name}")
        return blocks
        
    except Exception as e:
        logger.error(f"Local video ingestion failed for {document_id}: {e}", exc_info=True)
        return []
