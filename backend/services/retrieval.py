import time
import uuid
import asyncio
from typing import Dict, List, Optional, Callable, Awaitable
from retrieval.bm25_index import BM25ChunkIndex
from retrieval.hybrid_retriever import hybrid_retrieve, normalize
from graph.traversal import multi_hop_expand
from rerank import rerank_with_cross_encoder
from metrics import log_metric
import logging

_LOG = logging.getLogger("retrieval_service")

async def retrieve_candidates(
    *,
    query: str,
    rewrites: List[dict],
    bm25: Optional[BM25ChunkIndex],
    graph: Dict,
    chunk_fetcher: Callable[[List[str]], Awaitable[List[Dict]]], # Callback to fetch chunk details from Qdrant
    dense_fn,
    batch_id: str,
    sparse_top_k: int = 40,
    dense_top_k: int = 60,
    alpha: float = 0.45,
    min_score: float = 0.10,
    min_dense_score: float = 0.30,
    max_candidates: int = 30,
    correlation_id: str = None,
    limit: int = 6,
    t_inference_start: float = None
) -> List[dict]:
    """
    Stateless Retrieval pipeline.
    """
    
    # Generate a request ID for correlation if not provided
    req_id = correlation_id or str(uuid.uuid4())
    t_start = t_inference_start or time.time()

    if not rewrites:
        rewrites = [{"query": query, "type": "original", "weight": 1.0}]

    agg_dense: Dict[str, float] = {}
    agg_sparse: Dict[str, float] = {}
    semantic_impacts: Dict[str, Dict[str, float]] = {}
    origins_map: Dict[str, set] = {}

    # ---------- RETRIEVAL PER REWRITE (PARALLEL) ----------
    t_retrieval_start = time.time()
    print(f"      [RETRIEVAL] Executing Pipeline for {len(rewrites)} queries...")

    async def process_single_rewrite(r, idx):
        q = r.get("query")
        r_type = r.get("type")
        w = float(r.get("weight", 1.0))

        if not q:
            return None

        # ---- Dense (Async) ----
        try:
            raw_dense = await dense_fn(q, batch_id=batch_id, top_k=dense_top_k)
        except Exception as e:
            _LOG.error("dense_fn failed for %s: %s", q, e)
            raw_dense = {}

        # In stateless mode, we don't have STATE.chunk_lookup here.
        # We'll validate IDs during materialization.
        dense_scores = {
            str(cid): float(s) 
            for cid, s in raw_dense.items() 
            if float(s) >= min_dense_score
        }
        
        # ---- Sparse (Sync In-Memory per Request) ----
        # CPU-Bound: Offload to thread
        sparse_scores = await asyncio.to_thread(
            bm25.search, q, batch_id=batch_id, top_k=sparse_top_k
        ) if bm25 else {}

        return {
            "query": q,
            "type": r_type,
            "weight": w,
            "dense_scores": dense_scores,
            "sparse_scores": sparse_scores
        }

    # Parallelize searches
    rewrite_tasks = [process_single_rewrite(r, i) for i, r in enumerate(rewrites)]
    rewrite_results = await asyncio.gather(*rewrite_tasks)

    for res in rewrite_results:
        if not res: continue
        q, r_type, w = res["query"], res["type"], res["weight"]
        dense_scores, sparse_scores = res["dense_scores"], res["sparse_scores"]

        # Track Origins
        for cid in dense_scores:
            if cid not in origins_map: origins_map[cid] = set()
            origins_map[cid].add("dense")
        for cid in sparse_scores:
            if cid not in origins_map: origins_map[cid] = set()
            origins_map[cid].add("sparse")

        # Impact Tracking
        if r_type == "semantic":
            if q not in semantic_impacts:
                semantic_impacts[q] = {}
            for cid, s in dense_scores.items():
                semantic_impacts[q][cid] = semantic_impacts[q].get(cid, 0.0) + (s * w * alpha)
            for cid, s in sparse_scores.items():
                semantic_impacts[q][cid] = semantic_impacts[q].get(cid, 0.0) + (s * w * (1.0 - alpha))

        # Aggregation (Max + Consensus Bonus)
        for cid, score in dense_scores.items():
            weighted = score * w
            if cid not in agg_dense:
                agg_dense[cid] = weighted
            else:
                current = agg_dense[cid]
                agg_dense[cid] = max(current, weighted) + (min(current, weighted) * 0.1)

        for cid, score in sparse_scores.items():
            weighted = score * w
            if cid not in agg_sparse:
                agg_sparse[cid] = weighted
            else:
                current = agg_sparse[cid]
                agg_sparse[cid] = max(current, weighted) + (min(current, weighted) * 0.1)

    t_search = time.time()
    print(f"      -> Search (Dense+Sparse) Completed in {t_search - t_retrieval_start:.2f}s (Total: {t_search - t_start:.2f}s)")

    if not agg_dense and not agg_sparse:
        return []

    # ---------- NORMALIZE AGGREGATIONS ----------
    agg_dense = normalize(agg_dense)
    agg_sparse = normalize(agg_sparse)

    # ---------- HYBRID FUSION ----------
    hybrid_scores = hybrid_retrieve(
        dense_scores=agg_dense,
        sparse_scores=agg_sparse,
        alpha=alpha,
    )
    t_fusion = time.time()
    print(f"      -> Hybrid Fusion: {len(hybrid_scores)} candidates fused in {(t_fusion - t_search)*1000:.2f}ms (Total: {t_fusion - t_start:.2f}s)")

    if not hybrid_scores:
        print("      -> [RETRIEVAL STOP] No candidates found during hybrid search. Skipping Graph Expansion and Materialization.")
        return []

    # ---------- GRAPH EXPANSION ----------
    # In stateless mode, multi_hop_expand needs a way to check batch_id.
    # However, since we'll fetch full chunks later, we can temporarily assume the graph 
    # itself was built per-batch or check after materialization.
    # To keep it truly secure, we'll fetch all candidate metadata from Qdrant.
    
    # CPU-Bound: Offload to thread
    expanded = await asyncio.to_thread(
        multi_hop_expand,
        seed_scores=hybrid_scores,
        graph=graph,
        chunk_lookup={}, # We don't have lookup in memory anymore
        batch_id=batch_id,
        max_hops=2,
        max_total=60,
        use_cache_only=True # Tell traversal to not rely on lookup for filtering if unavailable
    )
    
    for cid in expanded:
        if cid not in origins_map: origins_map[cid] = set()
        origins_map[cid].add("graph")

    t_graph = time.time()
    print(f"      -> Graph Expansion: Expanded to {len(expanded)} candidates in {(t_graph - t_fusion)*1000:.2f}ms (Total: {t_graph - t_start:.2f}s)")

    # ---------- NORMALIZE + FILTER ----------
    if not expanded:
        return []
        
    normalized_expanded = normalize(expanded)

    ranked_ids = [
        cid for cid, score in
        sorted(normalized_expanded.items(), key=lambda x: (-x[1], x[0]))
        if score >= min_score
    ][:max_candidates]

    # ---------- MATERIALIZE CANDIDATES (STATELESS) ----------
    # Fetch chunk details from Qdrant/Storage
    chunk_details = await chunk_fetcher(ranked_ids)
    chunk_lookup = {c["chunk_id"]: c for c in chunk_details}

    candidates = []
    for cid in ranked_ids:
        chunk = chunk_lookup.get(cid)
        if not chunk:
            continue

        # CRITICAL SECURITY GATE: Ensure batch matches
        chunk_batch = chunk.get("batch_id") or chunk.get("metadata", {}).get("batch_id")
        if str(chunk_batch) != str(batch_id):
            continue

        candidates.append({
            "doc_id": cid,
            "text": chunk["text"],
            "metadata": {
                "section_title": chunk.get("section_title"),
                "pages": chunk.get("pages"),
                "origins": list(origins_map.get(cid, ["unknown"]))
            },
            "score": expanded[cid],
        })
    t_materialize = time.time()
    print(f"      -> Materialization Completed in {t_materialize - t_graph:.2f}s (Total: {t_materialize - t_start:.2f}s)")

    # ---------- MAX-SIM RERANKING (OFFLOADED) ----------
    rewrite_performance: Dict[str, float] = {}
    for q_text, impacts in semantic_impacts.items():
        rewrite_performance[q_text] = sum(impacts.get(cid, 0.0) for cid in ranked_ids)
    
    best_semantic = []
    if rewrite_performance:
        top_q = sorted(rewrite_performance.items(), key=lambda x: (-x[1], x[0]))[0][0]
        if top_q != query:
            best_semantic = [top_q]

    t4 = time.time()
    # Offload CPU-bound reranking to a thread to keep the event loop free
    results = await asyncio.to_thread(
        rerank_with_cross_encoder,
        query=query,
        candidates=candidates,
        alternative_queries=best_semantic,
        top_k=limit
    )
    t5 = time.time()
    print(f"      -> Reranking Completed in {t5 - t4:.2f}s (Total: {t5 - t_start:.2f}s)")

    
    log_metric({
        "category": "retrieval",
        "operation": "reranking",
        "model_name": "ms-marco-MiniLM-L-6-v2",
        "duration_ms": round((t5 - t4) * 1000, 2),
        "metadata": {"candidates_count": len(candidates), "correlation_id": req_id}
    })

    return results
