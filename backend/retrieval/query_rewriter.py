import re
import json
import logging
import nltk
import asyncio
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Callable, Dict, List, Literal, TypedDict, Awaitable, Optional
from fastapi import HTTPException
from json_utils import safe_json_load
from services.embeddings import get_embedding_model, compute_cosine_similarity
from services.embedding_runtime import EmbeddingRequestContext, get_embedding_runtime

# Ensure NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

_LOG = logging.getLogger("query_rewrite_ensemble")

RewriteType = Literal["original", "semantic", "keyword", "decomposition"]

# ----------------------------
# DATA STRUCTURES
# ----------------------------

class Rewrite(TypedDict):
    query: str
    type: RewriteType
    weight: float


class RewriteResult(TypedDict):
    original: str
    rewrites: List[Rewrite]


# ----------------------------
# SEMANTIC DRIFT GUARD
# ----------------------------

async def _too_much_drift(original: str, rewrite: str, threshold: float = 0.65) -> bool:
    """
    Professional-grade drift detection using semantic embeddings (Offloaded).
    """
    if not original or not rewrite:
        return True
    
    try:
        model = get_embedding_model()
        
        def compute_drift():
            v1 = model.encode(original, is_query=True)
            v2 = model.encode(rewrite, is_query=True)
            return compute_cosine_similarity(v1, v2)
            
        sim = await asyncio.to_thread(compute_drift)
        return sim < threshold
    except Exception as e:
        _LOG.warning("Drift detection failed, failing closed: %s", e)
        return True


# ----------------------------
# NORMALIZATION
# ----------------------------

def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def _dedupe(queries: List[str]) -> List[str]:
    seen = set()
    out = []
    for q in queries:
        qn = q.lower()
        if qn not in seen:
            seen.add(qn)
            out.append(q)
    return out


# ----------------------------
# UNIFIED GENERATION STRATEGY
# ----------------------------

async def _generate_llm_ensemble_rewrites(
    query: str,
    call_llm: Callable[[str], Awaitable[str]],
    max_semantic: int,
    max_decomposition: int,
) -> Dict[str, List[str]]:
    """
    AC-55: Single-Pass Query Expansion (Unified Intelligence).
    Combines semantic and decomposition instructions into one LLM call
    to reduce API latency by ~50%.
    """
    prompt = f"""
Analyze the following user query for document retrieval optimization.

Tasks:
1. Generate up to {max_semantic} semantic variations (different wording, same intent).
2. Decompose into up to {max_decomposition} atomic sub-queries (smaller searchable parts).

Rules:
- Output valid JSON only.
- Preserve technical terms exactly.
- Do NOT answer the question.
- Format: {{ "semantic": ["..."], "decomposition": ["..."] }}

Query: {query}
""".strip()

    try:
        raw = await call_llm(prompt)
        data = safe_json_load(raw)
        
        return {
            "semantic": data.get("semantic", []) if isinstance(data.get("semantic"), list) else [],
            "decomposition": data.get("decomposition", []) if isinstance(data.get("decomposition"), list) else []
        }
    except Exception as e:
        _LOG.warning("Unified LLM rewrite failed: %s", e)
        return {"semantic": [], "decomposition": []}


def _keyword_rewrite(query: str) -> List[str]:
    """
    Professional BM25-optimized keyword extraction.
    """
    try:
        stop_words = set(stopwords.words("english"))
    except:
        stop_words = set()
    
    stemmer = PorterStemmer()
    
    raw_tokens = re.split(r"[^\w]+", query.lower())
    
    processed = []
    seen = set()
    
    for t in raw_tokens:
        if len(t) < 2 or t in stop_words:
            continue
        
        stemmed = stemmer.stem(t)
        
        if stemmed not in seen:
            processed.append(stemmed)
            seen.add(stemmed)
    
    result = " ".join(processed)
    
    if not result or result == _normalize(query).lower():
        return []
        
    return [result]


# ----------------------------
# MAIN ENSEMBLE ENTRYPOINT
# ----------------------------

async def rewrite_query_ensemble(
    query: str,
    call_llm_fn: Callable[[str], Awaitable[str]],
    max_semantic: int = 3,
    max_decomposition: int = 2,
    use_llm: bool = True,
    embedding_context: Optional[EmbeddingRequestContext] = None,
) -> RewriteResult:
    """
    Optimized query rewrite ensemble.
    - Uses Single-Pass LLM inference for low latency.
    - Batches semantic drift guard for sub-second performance.
    """

    query = _normalize(query)
    if not query:
        return {"original": "", "rewrites": []}

    seen_queries = {query.lower()}
    rewrites: List[Rewrite] = [{"query": query, "type": "original", "weight": 1.0}]

    if use_llm:
        # --- 1. Single-Pass LLM Generation ---
        llm_data = await _generate_llm_ensemble_rewrites(
            query, 
            call_llm_fn, 
            max_semantic, 
            max_decomposition
        )
        
        sem_raw = llm_data["semantic"]
        dec_raw = llm_data["decomposition"]
        
        # Clean and prepare for batch embedding
        sem_cleaned = [_normalize(s) for s in sem_raw if _normalize(s)]
        dec_cleaned = [_normalize(s) for s in dec_raw if _normalize(s)]
        
        # Build unique candidate list for embedding (preserving order)
        candidates_to_embed = []
        for q in sem_cleaned + dec_cleaned:
            if q.lower() not in seen_queries and q not in candidates_to_embed:
                candidates_to_embed.append(q)
        
        if candidates_to_embed:
            try:
                all_texts = [query] + candidates_to_embed
                embeddings = await get_embedding_runtime().get_embeddings(
                    all_texts,
                    request_context=embedding_context,
                    priority="online",
                )
                
                orig_vec = embeddings[0]
                cand_vecs = embeddings[1:]
                
                # Create map for fast lookup
                score_map = {
                    cand: compute_cosine_similarity(orig_vec, cand_vecs[i])
                    for i, cand in enumerate(candidates_to_embed)
                }

                # --- Process Semantic ---
                for cand in sem_cleaned:
                    if cand.lower() in seen_queries: continue
                    sim = score_map.get(cand, 0.0)
                    if sim >= 0.65:
                        rewrites.append({"query": cand, "type": "semantic", "weight": 0.8})
                        seen_queries.add(cand.lower())

                # --- Process Decomposition ---
                for cand in dec_cleaned:
                    if cand.lower() in seen_queries: continue
                    sim = score_map.get(cand, 0.0)
                    if sim >= 0.55:
                        rewrites.append({"query": cand, "type": "decomposition", "weight": 0.7})
                        seen_queries.add(cand.lower())
                        
            except Exception as e:
                _LOG.error("Batch drift guard failed: %s", e)
                # Fallback
                for cand in sem_cleaned[:max_semantic]:
                    if cand.lower() not in seen_queries:
                        rewrites.append({"query": cand, "type": "semantic", "weight": 0.8})
                        seen_queries.add(cand.lower())

    # --- 2. Keyword Extraction (Sync) ---
    keyword_raw = _keyword_rewrite(query)
    for q in keyword_raw:
        qn = _normalize(q)
        if qn and qn.lower() not in seen_queries:
            rewrites.append({"query": qn, "type": "keyword", "weight": 0.5})
            seen_queries.add(qn.lower())

    _LOG.info(f"rewrite_ensemble for: {query} -> {len(rewrites)} rewrites")

    # --- DEVELOPMENT TERMINAL LOGGING ---
    print("\n" + "="*50)
    print(f"🔍 QUERY REWRITE ENSEMBLE [Topic: {query}]")
    print("-" * 50)
    for i, r in enumerate(rewrites):
        type_label = f"[{r['type'].upper()}]".ljust(15)
        print(f"{i+1}. {type_label} -> {r['query']} (w={r['weight']})")
    print("="*50 + "\n")

    return {
        "original": query,
        "rewrites": rewrites,
    }
