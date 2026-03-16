import os
import logging
import uuid
import tempfile
import shutil
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sys
import time

# Add backend to sys.path if not present for torchaudio_patch
backend_dir = str(Path(__file__).parent.parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    import torchaudio_patch
except ImportError:
    print("Warning: torchaudio_patch not found.")

# Initialize optional dependencies to None
yt_dlp = None
whisperx = None
torch = None
gc = None
pd = None

try:
    import yt_dlp
except ImportError:
    pass

try:
    import whisperx
    import torch
    import gc
    import pandas as pd
    # AC-105: OpenVINO Industrial Acceleration
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline
except ImportError as e:
    print(f"DEBUG: Failed to import dependencies (whisperx/optimum): {e}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Max duration for a single block (seconds) to ensure retrieval granularity
MAX_BLOCK_DURATION = 60.0

# Silence gap to trigger a new block (seconds)
SILENCE_GAP_THRESHOLD = 1.2

# Confidence fallback if model doesn't provide it
DEFAULT_CONFIDENCE = 0.75

# Device configuration
DEVICE = "cpu" # lowercase for PyTorch/Transformers compatibility
COMPUTE_TYPE = "int4" # Optimization 3: INT4 Quantization

logger = logging.getLogger(__name__)

# ==============================================================================
# UTILITIES
# ==============================================================================

def _generate_block_id(prefix: str = "t") -> str:
    """Generates a unique, sequential-looking ID for blocks."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def _cleanup_temp_files(paths: List[Path]):
    """Safely removes temporary files."""
    for p in paths:
        try:
            if p.exists():
                os.remove(p)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {p}: {e}")

def _get_hf_token() -> Optional[str]:
    """Retrieves Hugging Face token for Pyannote diarization."""
    return os.getenv("HF_TOKEN")

# ==============================================================================
# PHASE 1: AUDIO EXTRACTION (yt-dlp)
# ==============================================================================

def download_audio_from_youtube(url: str, temp_dir: Path) -> Path:
    """
    Downloads audio from a YouTube video using yt-dlp.
    Converts to 16kHz Mono WAV (Whisper requirement).
    
    Args:
        url: YouTube URL.
        temp_dir: Directory to store the temp file.
        
    Returns:
        Path to the generated .wav file.
    """
    if yt_dlp is None:
        raise ImportError("The 'yt_dlp' library is missing. Please run: pip install yt-dlp")

    # Check for local FFmpeg binaries to avoid system PATH issues
    ffmpeg_location = None
    script_dir = Path(__file__).parent.absolute()
    possible_paths = [
        script_dir / "ffmpeg.exe",
        Path("ingestion/ffmpeg.exe"),
        Path("ffmpeg.exe"),
        Path("bin/ffmpeg.exe"),
        Path("../ffmpeg.exe"),
    ]
    for p in possible_paths:
        if p.exists():
            ffmpeg_location = str(p.parent.absolute())
            logger.info(f"Using local FFmpeg at: {ffmpeg_location}")
            break

    output_template = str(temp_dir / "%(id)s.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'concurrent_fragment_downloads': 10,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ac', '1',                 # mono
            '-ar', '16000',              # 16 kHz
            '-acodec', 'pcm_s16le',      # explicit PCM 16-bit
        ],
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
        'noplaylist': True,
    }

    if ffmpeg_location:
        ydl_opts['ffmpeg_location'] = ffmpeg_location

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', 'unknown')
            expected_filename = f"{video_id}.wav"
            output_path = temp_dir / expected_filename
            
            if not output_path.exists():
                wavs = list(temp_dir.glob("*.wav"))
                if wavs:
                    return wavs[0]
                raise FileNotFoundError("Audio extraction failed: No WAV file found.")
            
            return output_path
            
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed: {str(e)}")

# ==============================================================================
# PHASE 2: TRANSCRIPTION (OpenVINO Optimized)
# ==============================================================================

def load_whisperx_model():
    """
    Optimization 1 & 3: Load Native OpenVINO Whisper with INT4 Quantization.
    Replaces standard WhisperX load with Optimum-Intel.
    """
    model_id = "openai/whisper-small"
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "whisper-small-ov-int4"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading OpenVINO Native Whisper (INT4) on {DEVICE}...")
    
    try:
        # Load OpenVINO model with INT4 weights
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            device=DEVICE,
            load_in_4bit=True, # Optimization 3
            export=True,       # Optimization 1
            compile=True,
            cache_dir=str(model_dir)
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Wrap in a HF Pipeline for seamless transcription
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device=DEVICE,
        )
        return pipe
    except Exception as e:
        logger.error(f"OpenVINO Model Load Failed: {e}")
        raise RuntimeError(f"Failed to initialize OpenVINO hardware acceleration: {e}")

def transcribe_with_whisperx(audio_path: Path, model=None) -> Dict[str, Any]:
    """
    Runs the optimized OpenVINO Pipeline.
    Optimization 2: Alignment is skipped entirely.
    """
    audio_file = str(audio_path)
    
    try:
        t_start_p2 = time.time()
        
        # 1. Load Model (On-demand if not pre-cached)
        if model is None:
            logger.info("[PHASE 2.1] Initializing OpenVINO Hardware Acceleration...")
            model = load_whisperx_model()
        
        # 2. Transcribe (Native OpenVINO)
        logger.info("[PHASE 2.3] Starting ASR (OpenVINO INT4)...")
        t_asr = time.time()
        
        # Optimization: Pin language to 'en' to skip 17s detection overhead
        # Note: In a production env, this could be passed as a param
        result = model(
            audio_file, 
            return_timestamps=True, 
            generate_kwargs={"language": "en", "task": "transcribe"}
        )
        
        logger.info(f"[PHASE 2.3] ASR Execution Time: {time.time() - t_asr:.2f}s")

        # 3. Schema Mapping (Map OV format to Legacy WhisperX format)
        # OV Output: {"text": "...", "chunks": [{"timestamp": (s, e), "text": "..."}]}
        # WhisperX Input for Phase 3: {"segments": [{"start": s, "end": e, "text": "...", "words": [...]}]}
        
        segments = []
        for chunk in result.get("chunks", []):
            start, end = chunk["timestamp"]
            # Optimization 2: Use segment-level data directly (Skip Alignment)
            segments.append({
                "start": start,
                "end": end or start + 5.0, # Fallback for open-ended segments
                "text": chunk["text"],
                "speaker": "SPEAKER_00",
                "words": [] # Alignment skipped, so word-level is empty
            })
            
        logger.info(f"Phase 2 Optimized Total Time: {time.time() - t_start_p2:.2f}s (Alignment Skipped)")
        
        return {"segments": segments, "language": "en"}
        
    except Exception as e:
        logger.error(f"Transcription Failed: {e}")
        return {"segments": [], "language": "en"}

# ==============================================================================
# PHASE 3: SEGMENT NORMALIZATION (Adapted for Segment-Level Timestamps)
# ==============================================================================

def flatten_words(whisper_result: Dict[str, Any]) -> List[Dict]:
    """
    Extracts a flat list of units from the Whisper result.
    ADAPTED: Now handles both word-level (legacy) and segment-level (OpenVINO) data.
    """
    all_units = []
    segments = whisper_result.get("segments", [])
    
    for seg in segments:
        words = seg.get("words", [])
        
        if words:
            # Legacy Path (Word-level timestamps present)
            for w in words:
                if "start" in w and "end" in w:
                    all_units.append({
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "score": w.get("score", 0.0),
                        "speaker": w.get("speaker", seg.get("speaker", "SPEAKER_00"))
                    })
        else:
            # Optimized Path: Use the Segment itself as the base unit
            # Since alignment is skipped, the segment is our smallest timestamped unit.
            all_units.append({
                "word": seg["text"], # The "unit" is the whole segment text
                "start": seg["start"],
                "end": seg["end"],
                "score": DEFAULT_CONFIDENCE,
                "speaker": seg.get("speaker", "SPEAKER_00")
            })
            
    return sorted(all_units, key=lambda x: x["start"])

def normalize_segments(words: List[Dict]) -> List[Dict]:
    """
    Regroups words into semantic blocks based on:
    1. Speaker Identity (Strict separation)
    2. Silence Gaps (> 1.2s)
    3. Max Duration (60s)
    """
    if not words:
        return []

    blocks = []
    
    # Current block buffer
    current_words = []
    current_speaker = words[0]["speaker"]
    block_start_time = words[0]["start"]
    last_word_end = words[0]["start"] # initialize to start to avoid initial gap trigger
    
    for word in words:
        # 1. Calculate Gap
        gap = word["start"] - last_word_end
        
        # 2. Check Triggers
        speaker_changed = (word["speaker"] != current_speaker)
        is_silence_gap = (gap > SILENCE_GAP_THRESHOLD)
        
        # Check potential duration if we add this word
        # Note: We use word['end'] as the potential new end
        potential_duration = word["end"] - block_start_time
        is_duration_exceeded = (potential_duration > MAX_BLOCK_DURATION)
        
        should_split = (speaker_changed or is_silence_gap or is_duration_exceeded)
        
        if should_split and current_words:
            # Finalize current block
            blocks.append({
                "words": current_words,
                "speaker": current_speaker,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"]
            })
            
            # Start new block
            current_words = [word]
            current_speaker = word["speaker"]
            block_start_time = word["start"]
        else:
            # Continue block
            current_words.append(word)
        
        last_word_end = word["end"]
    
        
    # Flush final block
    if current_words:
        blocks.append({
            "words": current_words,
            "speaker": current_speaker,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"]
        })
        
    return blocks

# ==============================================================================
# PHASE 4: BLOCK GENERATION (Schema Compliance)
# ==============================================================================
def create_blocks(
    normalized_segments: List[Dict], 
    document_id: str, 
    source_tag: str = "youtube_whisperx"
) -> List[Dict]:
    """
    Converts normalized segments into the strict RAG System Block Schema.
    """
    final_output = []
    
    for i, seg in enumerate(normalized_segments):
        words = seg.get("words", [])
        
        # Optimization 2 Recovery: If alignment was skipped, words is empty.
        # We reconstruct the text from the segment's summarized content (stored in 'word' by flatten_words)
        if not words:
            # When alignment is skipped, flatten_words puts the segment text into a single unit's 'word' field.
            # We must ensure we don't end up with empty text.
            text_content = " ".join([w.get("word", "").strip() for w in words]).strip()
            if not text_content and "text" in seg: # Fallback to segment-level text if available
                text_content = seg["text"].strip()
            
            # Format Metadata Words (Placeholder for segment-level block)
            metadata_words = [{"w": text_content, "s": seg["start"], "e": seg["end"]}]
        else:
            # Reconstruct text with proper punctuation spacing
            text_content = " ".join([w["word"].strip() for w in words]).strip()
            metadata_words = [
                {"w": w["word"], "s": w["start"], "e": w["end"]}
                for w in words
            ]
        
        # Calculate Average Confidence
        scores = [w.get("score", DEFAULT_CONFIDENCE) for w in (words or [{"score": DEFAULT_CONFIDENCE}])]
        avg_confidence = sum(scores) / len(scores) if scores else DEFAULT_CONFIDENCE
        
        block = {
            "document_id": document_id,
            "page": None,
            "block_id": _generate_block_id(),
            "block_type": "paragraph",
            "text": text_content,
            "bbox": None,
            "source": source_tag,
            "confidence": round(avg_confidence, 4),
            "metadata": {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "words": metadata_words
            }
        }
        
        final_output.append(block)
        logger.info(f"Created block {block['block_id']} with {len(text_content.split())} words, speaker {block['metadata']['speaker']}, duration {seg['end'] - seg['start']:.2f}s")
        
    return final_output

# ==============================================================================
# PUBLIC ENTRYPOINT
# ==============================================================================

def ingest_youtube_video(url: str, document_id: str, model=None) -> List[Dict]:
    """
    Ingests a YouTube video, transcribes it, and produces RAG-compatible blocks.

    Pipeline:
    1. Download Audio (yt-dlp) -> 16kHz WAV
    2. Transcribe & Align (WhisperX large-v3)
    3. Diarize (Pyannote)
    4. Normalize (Merge words into semantic blocks)
    5. Output (List of strict schema dicts)

    Args:
        url: The full YouTube video URL.
        document_id: A unique ID for the document (e.g., UUID or 'yt_<id>').
        model: Optional pre-loaded WhisperX model.

    Returns:
        List[Dict]: A list of block dictionaries compatible with the PDF ingestor.
    
    Raises:
        RuntimeError: If critical stages (download, transcription) fail.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="ingest_yt_"))
    audio_path = None
    
    try:
        logger.info(f"Starting ingestion for {document_id} from {url}")
        
        # 1. Download
        logger.info("Phase 1: Downloading audio...")
        audio_path = download_audio_from_youtube(url, temp_dir)
        
        # 2. Transcribe
        logger.info("Phase 2: Transcribing & Diarizing...")
        raw_result = transcribe_with_whisperx(audio_path, model=model)
        
        # 3. Normalize
        logger.info("Phase 3: Normalizing segments...")
        flat_words = flatten_words(raw_result)
        normalized_segments = normalize_segments(flat_words)
        
        # 4. Generate Blocks
        logger.info("Phase 4: Generating blocks...")
        blocks = create_blocks(normalized_segments, document_id)
        
        if not blocks:
            logger.warning(f"No valid content extracted from {url}")
            return []
            
        logger.info(f"Ingestion complete. Generated {len(blocks)} blocks.")
        return blocks
        
    except Exception as e:
        logger.error(f"Ingestion failed for {document_id}: {e}", exc_info=True)
        # Ensure we return a list, even if empty/error (though raising might be preferred depending on caller)
        # However, prompt says "Never crash ingestion for a single bad segment" and "Always return a list".
        # If the WHOLE process failed, we probably should return empty or raise. 
        # Given "Never crash ingestion", returning empty list with error logged is safest.
        return []
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

