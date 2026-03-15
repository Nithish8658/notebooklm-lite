import json
import uuid
import os
import asyncio
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Callable, Dict, Awaitable
from pydantic import BaseModel, Field

from services.retrieval import retrieve_candidates
from retrieval.query_rewriter import rewrite_query_ensemble
from services.prosody_planner import ProsodyPlanner
from services.audio_processor import audio_master
from json_utils import safe_json_load
from services.llm import call_gemini_async
from retrieval.bm25_index import BM25ChunkIndex

# ============================================
# Configuration
# ============================================

AUDIO_OUTPUT_DIR = Path("uploads/podcasts")
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEGMENT_WORDS = 220
MIN_SEGMENTS_REQUIRED = 6
TTS_CONCURRENCY_LIMIT = 5


# ============================================
# Models
# ============================================

class PodcastSegment(BaseModel):
    speaker: str
    text: str
    emotion: Optional[str] = None
    intensity: Optional[float] = None
    interrupt: Optional[bool] = False


class Podcast(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    script: List[PodcastSegment]
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0

class PodcastRefusalError(Exception):
    """Exception raised when the LLM refuses to generate a podcast due to lack of context."""
    pass


# ============================================
# Episode State Engine
# ============================================

class EpisodeState:
    def __init__(self, total_segments: int):
        self.total = total_segments
        self.index = 0
        self.momentum = 0.0
        self.tension = 0.0
        self.prev_emotion = None

    def phase(self) -> str:
        progress = self.index / max(self.total - 1, 1)

        if progress < 0.15:
            return "HOOK"
        elif progress < 0.40:
            return "BUILD"
        elif progress < 0.65:
            return "FRICTION"
        elif progress < 0.85:
            return "CLIMAX"
        else:
            return "RESOLUTION"

    def advance(self):
        self.index += 1


# ============================================
# Utilities
# ============================================

def validate_script(data: dict) -> List[PodcastSegment]:
    if "script" not in data:
        raise ValueError("Invalid structure: 'script' key missing")

    segments = []
    raw_count = len(data["script"])

    for seg in data["script"]:
        if not isinstance(seg, dict):
            continue

        speaker = str(seg.get("speaker", ""))
        text = seg.get("text")

        # AC-48: Flexible Speaker Mapping
        # Gemini often uses "Host A", "Speaker 1", etc. Map them to our internal IDs.
        mapped_speaker = None
        if "1" in speaker or "A" in speaker or "First" in speaker:
            mapped_speaker = "Host 1"
        elif "2" in speaker or "B" in speaker or "Second" in speaker:
            mapped_speaker = "Host 2"
        elif speaker in ["Host 1", "Host 2"]:
            mapped_speaker = speaker

        if not mapped_speaker:
            continue
        if not text:
            continue

        words = text.split()
        if len(words) > MAX_SEGMENT_WORDS:
            text = " ".join(words[:MAX_SEGMENT_WORDS])

        segments.append(
            PodcastSegment(
                speaker=mapped_speaker,
                text=text.strip(),
                emotion=seg.get("emotion"),
                intensity=seg.get("intensity"),
                interrupt=seg.get("interrupt", False)
            )
        )

    print(f"PODCAST VALIDATION: Found {len(segments)} valid segments out of {raw_count} raw blocks.")
    
    if len(segments) < MIN_SEGMENTS_REQUIRED:
        if len(segments) > 0:
            print(f"DEBUG: First segment speaker: {segments[0].speaker}")
        raise ValueError(f"Script too short ({len(segments)} < {MIN_SEGMENTS_REQUIRED})")

    return segments


def apply_tension_model(state: EpisodeState, segment: PodcastSegment):

    phase = state.phase()

    # Phase-based emotional shaping
    if phase == "HOOK":
        segment.intensity = 0.6
    elif phase == "BUILD":
        segment.intensity = 0.75
    elif phase == "FRICTION":
        segment.intensity = 0.85
        segment.emotion = segment.emotion or "SERIOUS"
    elif phase == "CLIMAX":
        segment.intensity = 1.0
        segment.emotion = segment.emotion or "EXCITED"
    elif phase == "RESOLUTION":
        segment.intensity = 0.5
        segment.emotion = segment.emotion or "REFLECTIVE"

    # Momentum smoothing
    if state.prev_emotion == segment.emotion:
        segment.intensity *= 0.9

    state.prev_emotion = segment.emotion
    state.advance()


# ============================================
# Audio Generation
# ============================================

async def generate_podcast_audio(script: List[PodcastSegment], output_path: str) -> bool:
    import edge_tts

    if not shutil.which("ffmpeg"):
        return False

    planner = ProsodyPlanner()  # isolate per episode

    VOICES = {
        "Host 1": "en-US-AndrewNeural",
        "Host 2": "en-US-AvaNeural"
    }

    semaphore = asyncio.Semaphore(TTS_CONCURRENCY_LIMIT)
    temp_files: List[Path] = []

    async def synth_segment(i: int, seg: PodcastSegment):
        async with semaphore:
            enriched = planner.plan_segment(seg.speaker, seg.text)
            conf = enriched.config.copy()

            # intensity scaling
            if seg.intensity:
                for k in conf:
                    if "%" in conf[k]:
                        val = int(conf[k].replace("%", "").replace("+", ""))
                        conf[k] = f"{int(val * seg.intensity):+d}%"
                    if "Hz" in conf[k]:
                        val = int(conf[k].replace("Hz", "").replace("+", ""))
                        conf[k] = f"{int(val * seg.intensity):+d}Hz"

            temp = AUDIO_OUTPUT_DIR / f"temp_{i}_{uuid.uuid4().hex}.wav"

            communicate = edge_tts.Communicate(
                text=seg.text,
                voice=VOICES[seg.speaker],
                rate=conf.get("rate"),
                pitch=conf.get("pitch"),
                volume=conf.get("volume")
            )

            await communicate.save(str(temp))
            return temp

    try:
        tasks = [synth_segment(i, s) for i, s in enumerate(script)]
        temp_files = await asyncio.gather(*tasks)

        concat_file = AUDIO_OUTPUT_DIR / f"concat_{uuid.uuid4().hex}.txt"
        with open(concat_file, "w") as f:
            for tf in temp_files:
                f.write(f"file '{tf.resolve()}'\n")

        merged = output_path.replace(".mp3", "_merged.wav")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "pcm_s16le",
            merged
        ], check=True)

        os.remove(concat_file)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", merged,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ], check=True)

        os.remove(merged)

        mastered = output_path.replace(".mp3", "_mastered.mp3")
        if audio_master.master_podcast(output_path, mastered):
            os.remove(output_path)
            os.rename(mastered, output_path)

        return True

    except Exception:
        return False

    finally:
        for tf in temp_files:
            if tf.exists():
                os.remove(tf)


# ============================================
# Script Generation
# ============================================

async def generate_podcast_script(
    topic: str,
    bm25: Optional[BM25ChunkIndex],
    graph: Dict,
    chunk_fetcher: Callable,
    dense_retrieve_fn: Callable,
    call_gemini_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    batch_id: str = "default_batch",
    complexity: str = "Undergrad"
) -> Optional[List[PodcastSegment]]:

    llm_fn = call_gemini_fn or call_gemini_async
    
    COMPLEXITY_MAP = {
        "5-Year-Old": "Explain like I'm five. Use very simple language, analogies, and keep it very brief.",
        "High School": "Explain at a high school level. Use clear, accessible language and avoid overly technical jargon unless explained.",
        "Undergrad": "Explain at a college undergraduate level. Provide a balanced, detailed, overview with standard academic terminology.",
        "PhD Expert": "Provide a highly technical, rigorous, and nuanced analysis suitable for a PhD expert. Use advanced terminology and address subtle complexities."
    }

    complexity_instr = COMPLEXITY_MAP.get(complexity, COMPLEXITY_MAP["Undergrad"])

    try:
        rewrite = await rewrite_query_ensemble(topic, llm_fn, 3, 2, use_llm=True)
        rewrites = rewrite.get("rewrites", [{"query": topic, "weight": 1.0}])

        candidates = await retrieve_candidates(
            query=topic,
            rewrites=rewrites,
            bm25=bm25,
            graph=graph,
            chunk_fetcher=chunk_fetcher,
            dense_fn=dense_retrieve_fn,
            batch_id=batch_id,
            dense_top_k=15,
            max_candidates=6,
            min_dense_score=0.30, # AC-35: Enforce Chat-level precision
            correlation_id=f"podcast_{topic[:10]}"
        )


        if not candidates:
            raise PodcastRefusalError("Not enough resource to generate podcast")

        context = "\n\n".join(
            f"<SOURCE>\n{c['text']}\n</SOURCE>" for c in candidates
        )

        prompt = f"""
Generate a high-quality, emotionally dynamic podcast conversation .

Complexity Level: {complexity}
Tone Instructions: {complexity_instr}

RULES
1. Topic Validation
If the CHUNKS do not clearly discuss the topic "{topic}", return: NO_RELEVANT_CONTEXT
2. Context Restriction
Use only information from the CHUNKS. No external knowledge or assumptions.
3. Confidence Gate
If the topic cannot be supported from the CHUNKS with ≥0.8 confidence, return: NO_RELEVANT_CONTEXT
4. Chunk Filtering
Ignore vague or unrelated chunks. If none remain, return:NO_RELEVANT_CONTEXT
5. Generation (only if above rules pass)
Create exactly 10-14 podcast segments with emotional escalation, a conflict moment, natural interruptions, and a clear narrative arc.
All sentences must be grounded in the CHUNKS.
STRICT JSON:
{{
 "script": [
   {{
     "speaker": "Host 1",
     "text": "...",
     "emotion": "EXCITED",
     "intensity": 0.8,
     "interrupt": false
   }}
 ]
}}

TOPIC: {topic}
SOURCE:
{context}
"""

        response = await llm_fn(prompt)
        
        if "NO_RELEVANT_CONTEXT" in response:
            print(f"PODCAST ALERT: LLM refused generation due to lack of context for topic '{topic}'")
            raise PodcastRefusalError("Not enough resource to generate podcast")

        try:
            data = safe_json_load(response)
        except Exception as pe:
            print(f"PODCAST ERROR: JSON Parse failed. Raw response snippet: {response[:200]}...")
            raise RuntimeError(f"Failed to parse podcast JSON: {pe}")

        try:
            segments = validate_script(data)
        except Exception as ve:
            print(f"PODCAST ERROR: Script validation failed: {ve}")
            raise RuntimeError(f"Invalid podcast script structure: {ve}")

        # Apply tension model
        episode_state = EpisodeState(len(segments))
        for seg in segments:
            apply_tension_model(episode_state, seg)

        return segments

    except Exception as e:
        if isinstance(e, (RuntimeError, PodcastRefusalError)):
            raise
        print(f"PODCAST CRITICAL: Script generation crashed: {e}")
        return None


# ============================================
# Main Entry
# ============================================

async def generate_podcast(
    topic: str,
    bm25: Optional[BM25ChunkIndex],
    graph: Dict,
    chunk_fetcher: Callable,
    dense_retrieve_fn: Callable,
    call_gemini_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    batch_id: str = "default_batch",
    complexity: str = "Undergrad"
) -> Optional[Podcast]:

    segments = await generate_podcast_script(
        topic, bm25, graph, chunk_fetcher, dense_retrieve_fn, call_gemini_fn, batch_id=batch_id, complexity=complexity
    )

    if not segments:
        return None

    podcast_id = str(uuid.uuid4())
    filename = f"{podcast_id}.mp3"
    full_path = str(AUDIO_OUTPUT_DIR / filename)

    success = await generate_podcast_audio(segments, full_path)

    total_words = sum(len(s.text.split()) for s in segments)
    duration = (total_words / 150) * 60

    return Podcast(
        id=podcast_id,
        topic=topic,
        script=segments,
        audio_path=filename if success else None,
        duration_seconds=duration
    )
