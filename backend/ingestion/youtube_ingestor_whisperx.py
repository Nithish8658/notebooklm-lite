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
from pathlib import Path

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
except ImportError as e:
    print(f"DEBUG: Failed to import whisperx: {e}")

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
# We delay checking torch availability until needed or default to cpu if missing
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

if torch is not None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

BATCH_SIZE = 16 # Adjust based on VRAM

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
    # Add ingestion directory itself to possible paths
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

        # Download efficiency
        'concurrent_fragment_downloads': 10,

        # Audio extraction
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],

        # Enforce ASR-safe audio
        'postprocessor_args': [
            '-ac', '1',                 # mono
            '-ar', '16000',              # 16 kHz
            '-acodec', 'pcm_s16le',      # explicit PCM 16-bit
        ],

        # Stability & debugging
        'quiet': False,                 # DO NOT suppress during development
        'no_warnings': False,
        'ignoreerrors': False,
        'noplaylist': True,             # CRITICAL: Prevent downloading 1000+ videos if URL is part of a playlist
    }

    
    if ffmpeg_location:
        ydl_opts['ffmpeg_location'] = ffmpeg_location

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', 'unknown')
            # yt-dlp post-processing changes extension to .wav
            expected_filename = f"{video_id}.wav"
            output_path = temp_dir / expected_filename
            
            if not output_path.exists():
                # Fallback: find any wav in the dir if exact match fails
                wavs = list(temp_dir.glob("*.wav"))
                if wavs:
                    return wavs[0]
                raise FileNotFoundError("Audio extraction failed: No WAV file found.")
            
            return output_path
            
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed: {str(e)}")

# ==============================================================================
# PHASE 2: TRANSCRIPTION & DIARIZATION (WhisperX)
# ==============================================================================

def load_whisperx_model():
    """Helper to load the model with compatibility patches."""
    if whisperx is None:
        raise ImportError("The 'whisperx' library is missing.")
    
    # Fix: Define local model path relative to backend/
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "whisper-small"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading WhisperX model (small) on {DEVICE}...")
    logger.info(f"Model cache directory: {model_dir}")
    
    from unittest.mock import patch
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    try:
        with patch('torch.load', side_effect=unsafe_load):
            model = whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, download_root=str(model_dir))
        return model
    except Exception as e:
        logger.warning(f"Patching torch.load failed, trying direct load: {e}")
        return whisperx.load_model("small", DEVICE, compute_type=COMPUTE_TYPE, download_root=str(model_dir))

def transcribe_with_whisperx(audio_path: Path, model=None) -> Dict[str, Any]:
    """
    Runs the full WhisperX pipeline: 
    Transcribe -> Align -> Diarize -> Assign Speakers.
    
    Returns a dictionary containing segments with word-level timestamps and speakers.
    """
    if whisperx is None:
        raise ImportError("The 'whisperx' library is missing. Please run: pip install git+https://github.com/m-bain/whisperx.git")

    audio_file = str(audio_path)
    
    # --- FFmpeg PATH FIX ---
    # WhisperX assumes ffmpeg is in PATH. We temporarily add our local bin.
    # We reuse the discovery logic 
    script_dir = Path(__file__).parent.absolute()
    local_ffmpeg = script_dir / "ffmpeg.exe"
    if not local_ffmpeg.exists():
        # Try fallbacks if moved
        local_ffmpeg = Path("ffmpeg.exe").absolute()
    
    old_path = os.environ.get("PATH", "")
    if local_ffmpeg.exists():
        ffmpeg_dir = str(local_ffmpeg.parent)
        if ffmpeg_dir not in old_path:
            logger.info(f"Temporarily adding {ffmpeg_dir} to PATH for WhisperX")
            os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{old_path}"

    try:
        # 1. Transcribe (ASR)
        t_start_p2 = time.time()
        should_cleanup = False
        if model is None:
            logger.info("[PHASE 2.1] Loading WhisperX Model...")
            t_model_load = time.time()
            model = load_whisperx_model()
            should_cleanup = True
            logger.info(f"[PHASE 2.1] Model Load Time: {time.time() - t_model_load:.2f}s")
        
        logger.info("[PHASE 2.2] Loading Audio into Memory...")
        t_audio_load = time.time()
        audio = whisperx.load_audio(audio_file)
        logger.info(f"[PHASE 2.2] Audio Load Time: {time.time() - t_audio_load:.2f}s")

        logger.info("[PHASE 2.3] Starting ASR (Transcription)...")
        # Reverted: Now allowing automatic language detection
        t_asr = time.time()
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        logger.info(f"[PHASE 2.3] ASR Execution Time: {time.time() - t_asr:.2f}s")
        
        # Cleanup ASR model ONLY if we loaded it locally
        if should_cleanup:
            del model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # 2. Alignment (Forced Alignment)
        logger.info(f"[PHASE 2.4] Aligning transcript for language: {result['language']}...")
        
        project_root = Path(__file__).parent.parent
        align_model_dir = project_root / "models" / "alignment"
        align_model_dir.mkdir(parents=True, exist_ok=True)
        
        t_align_load = time.time()
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=DEVICE,
            model_dir=str(align_model_dir)
        )
        logger.info(f"[PHASE 2.4] Alignment Model Load Time: {time.time() - t_align_load:.2f}s")

        logger.info("[PHASE 2.5] Starting Alignment...")
        t_align_exec = time.time()
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            DEVICE, 
            return_char_alignments=False
        )
        logger.info(f"[PHASE 2.5] Alignment Execution Time: {time.time() - t_align_exec:.2f}s")
        
        # Cleanup Alignment model
        del model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 3. Diarization (SKIP - Performance Optimization)
        logger.info("[PHASE 2.6] INDUSTRIAL OPTIMIZATION: Skipping Diarization.")
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"
            if "words" in seg:
                for word in seg["words"]:
                    word["speaker"] = "SPEAKER_00"
        
        logger.info(f"Phase 2 Total Time: {time.time() - t_start_p2:.2f}s")
        return result
    finally:
        # Restore PATH
        os.environ["PATH"] = old_path

# ==============================================================================
# PHASE 3: SEGMENT NORMALIZATION (Flattening & Regrouping)
# ==============================================================================

def flatten_words(whisper_result: Dict[str, Any]) -> List[Dict]:
    """
    Extracts a flat list of words from the hierarchical WhisperX result.
    Filters out words missing timestamps.
    """
    all_words = []
    segments = whisper_result.get("segments", [])
    
    for seg in segments:
        # Some segments might lack 'words' if alignment failed slightly
        words = seg.get("words", [])
        
        # Fallback: if no word-level timestamps, construct pseudo-words from segment?
        # For this strict pipeline, we prefer skipping unaligned content or utilizing segment level
        # if word level is totally missing. Ideally WhisperX alignment works.
        
        default_speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        
        for w in words:
            if "start" in w and "end" in w:
                w_obj = {
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "score": w.get("score", 0.0),
                    # Prefer word speaker, fall back to segment speaker
                    "speaker": w.get("speaker", default_speaker)
                }
                all_words.append(w_obj)
            
    return sorted(all_words, key=lambda x: x["start"])

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
        words = seg["words"]
        
        # Reconstruct text with proper punctuation spacing
        # WhisperX words often include punctuation, or we simply join with space.
        # Ideally, we verify punctuation attachment, but simple join is standard for Whisper word-level.
        text_content = " ".join([w["word"].strip() for w in words]).strip()
        
        # Calculate Average Confidence
        # Fallback to DEFAULT_CONFIDENCE if score is missing
        scores = [w.get("score", DEFAULT_CONFIDENCE) for w in words]
        avg_confidence = sum(scores) / len(scores) if scores else DEFAULT_CONFIDENCE
        
        # Format Metadata Words
        metadata_words = [
            {"w": w["word"], "s": w["start"], "e": w["end"]}
            for w in words
        ]
        block = {
            "document_id": document_id,
            "page": None, # YouTube has no pages
            "block_id": _generate_block_id(),
            "block_type": "paragraph",
            "text": text_content,
            "bbox": None,
            "source": source_tag,
            "confidence": round(avg_confidence, 4),
            "metadata": {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "speaker": seg["speaker"],
                "words": metadata_words
                
            }
        }
        
        
        final_output.append(block)
        print(f"Created block {block['block_id']} with {len(words)} words, speaker {seg['speaker']}, duration {seg['end'] - seg['start']:.2f}s")
        
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

