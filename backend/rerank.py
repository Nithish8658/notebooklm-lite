import os
import sys
import logging
from typing import List, Dict, Optional
from pathlib import Path
from collections import OrderedDict
import multiprocessing

# AC-100: Critical DLL Path Injection for OpenVINO on Windows
if sys.platform == "win32":
    try:
        import openvino
        ov_libs = Path(openvino.__file__).parent / "libs"
        if ov_libs.exists():
            os.add_dll_directory(str(ov_libs.absolute()))
    except Exception:
        pass

import torch
from transformers import AutoTokenizer
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
try:
    from optimum.intel.openvino import OVModelForSequenceClassification
    _HAS_OV = True
except ImportError:
    _HAS_OV = False

_LOG = logging.getLogger("rerank")
_RERANKER_MODEL = None
_DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-4-v2"

# ---- FUSION CONFIG ----
# These weights ensure structural graph context and deep semantic relevance are balanced.
GRAPH_WEIGHT = 0.35  
RERANK_WEIGHT = 0.65 
CALIBRATION_TEMP = 1.0 


# ---------- ONNX MODEL WRAPPER ----------
class OnnxCrossEncoder:
    def __init__(self, model_id: str = _DEFAULT_MODEL_NAME):
        # Path: backend/rerank.py -> backend/models/reranker-onnx
        project_root = Path(__file__).resolve().parent
        onnx_dir = project_root / "models" / "reranker-onnx"
        
        # AC-52: Smart path resolution for Xenova/Optimum structure
        model_filename = "model_quantized.onnx"
        subfolder = None
        
        # Check for quantized version first
        if (onnx_dir / "onnx" / "model_quantized.onnx").exists():
            subfolder = "onnx"
        elif (onnx_dir / "model_quantized.onnx").exists():
            subfolder = None
        else:
            # Fallback to standard model.onnx
            model_filename = "model.onnx"
            if (onnx_dir / "onnx" / "model.onnx").exists():
                subfolder = "onnx"
            elif (onnx_dir / "model.onnx").exists():
                subfolder = None
            else:
                # CRITICAL: Prevent fallback to HF Hub
                raise FileNotFoundError(f"No ONNX model found in {onnx_dir}. Ensure the main process has finished downloading models.")

        _LOG.info("Loading Optimized Reranker (File: %s, Subfolder: %s) from %s", 
                 model_filename, subfolder or "root", onnx_dir)
        
        # 1. Thread Management: Hardware-Aware and Scale-Safe
        env_threads = os.getenv("RERANKER_THREADS")
        if env_threads:
            num_threads = int(env_threads)
        else:
            total_cores = multiprocessing.cpu_count()
            # Optimization: 4 threads is often the sweet spot for MiniLM on consumer CPUs
            num_threads = min(4, max(1, total_cores // 2))
            
        _LOG.info("CPU Optimization: Using %d threads for reranking inference.", num_threads)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        provider = "CPUExecutionProvider"
        provider_options = None

        if "OpenVINOExecutionProvider" in available_providers:
            provider = "OpenVINOExecutionProvider"
            # Optimization: num_streams=1 reduces latency for single-user queries
            provider_options = {
                "device_type": "CPU",
                "num_streams": "1"
            }
            _LOG.info("OpenVINO detected. Enabling high-speed Intel inference (Latency Mode).")
        
        # Use .as_posix() for Windows path compatibility
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir.as_posix(), local_files_only=True)
        
        # AC-99: Verify model size to confirm quantization is active
        model_path = onnx_dir / (subfolder or "") / model_filename
        if model_path.exists():
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            _LOG.info("Model Verification: Loaded %s (Size: %.2f MB)", model_filename, size_mb)
            if size_mb > 40: # L-4 quantized is ~15MB, L-6 is ~22MB. 
                _LOG.warning("PERFORMANCE ALERT: Model size (%.2f MB) suggests this is NOT the quantized version. Expect high latency.", size_mb)

        self.model = ORTModelForSequenceClassification.from_pretrained(
            onnx_dir.as_posix(),
            file_name=model_filename,
            subfolder=subfolder,
            provider=provider,
            provider_options=provider_options,
            session_options=sess_options,
            local_files_only=True
        )
        
        # Log which provider was actually loaded
        _LOG.info("Active Inference Provider: %s", self.model.providers)

    def predict(self, pairs: List[tuple], batch_size: int = 16) -> List[float]:
        """
        Predict relevance scores for (query, document) pairs.
        Returns sigmoid-normalized scores in [0, 1].
        """
        import time
        t_predict_start = time.perf_counter()
        cpu_start = time.process_time()
        
        scores_out = []
        total_batches = (len(pairs) + batch_size - 1) // batch_size
        
        t_tokenization = 0.0
        t_forward = 0.0
        t_cleanup = 0.0
        total_tokens = 0

        for i in range(0, len(pairs), batch_size):
            batch_num = (i // batch_size) + 1
            batch = pairs[i : i + batch_size]
            
            # 1. Tokenization
            t0 = time.perf_counter()
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            t_tokenization += time.perf_counter() - t0
            
            # Track batch stats
            batch_seq_len = inputs["input_ids"].shape[1]
            total_tokens += (len(batch) * batch_seq_len)

            # 2. Forward Pass (Raw ONNX Execution)
            t1 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(**inputs)
            t_forward += time.perf_counter() - t1

            # 3. Cleanup & Calibration (Logits -> Sigmoid -> NumPy)
            t2 = time.perf_counter()
            logits = outputs.logits
            if logits.shape[1] == 1:
                scores = logits.squeeze(-1)
            else:
                scores = logits[:, 1]

            # Calibration: apply temperature before sigmoid
            scores = torch.sigmoid(scores / CALIBRATION_TEMP).cpu().numpy()
            scores_out.extend(scores.tolist())
            t_cleanup += time.perf_counter() - t2
            
            if batch_num % 1 == 0 or batch_num == total_batches:
                _LOG.debug(f"      [BATCH {batch_num}/{total_batches}] seq_len: {batch_seq_len} | forward: {t1-t0:.4f}s")

        wall_time = time.perf_counter() - t_predict_start
        cpu_time = time.process_time() - cpu_start
        parallelism = cpu_time / wall_time if wall_time > 0 else 0
        
        print(f"\n   -> RERANK DEEP-DIVE (N={len(pairs)} pairs):")
        print(f"      - Avg Seq Length: {total_tokens/len(pairs):.1f} tokens")
        print(f"      - Tokenization:   {t_tokenization:.4f}s")
        print(f"      - Forward Pass:   {t_forward:.4f}s (Raw Math)")
        print(f"      - Result Cleanup: {t_cleanup:.4f}s")
        print(f"      - Parallelism:    {parallelism:.1f}x Cores")
        print(f"      - Wall Clock:     {wall_time:.4f}s")

        return scores_out


# ---------- SIMPLE LRU SCORE CACHE ----------
class _ScoreCache:
    def __init__(self, maxsize: int = 10000):
        self._maxsize = maxsize
        self._data = OrderedDict()

    def get(self, key):
        try:
            val = self._data.pop(key)
            self._data[key] = val
            return val
        except KeyError:
            return None

    def set(self, key, value):
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self._maxsize:
            self._data.popitem(last=False)
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val):
        self.set(key, val)


_cache_size = int(os.getenv("SCORE_CACHE_SIZE", "50000"))
_SCORE_CACHE = _ScoreCache(maxsize=_cache_size)


# ---------- SINGLETON ----------
def load_reranker():
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = OnnxCrossEncoder()
    return _RERANKER_MODEL

def get_reranker():
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _LOG.warning("RERANKER ALERT: Model was not pre-loaded at startup. Loading on-demand (this will cause a one-time delay)...")
        return load_reranker()
    return _RERANKER_MODEL


# ---------- MAIN API ----------
def rerank_with_cross_encoder(
    query: str,
    candidates: list[dict],
    alternative_queries: list[str]|None = None,
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    device: str | None = None,
    top_k: int | None = None,
    max_rewrites: int | None = None,
    use_fusion: bool = True,
) -> list[dict]:
    """
    Production-grade Cross-Encoder Reranking with Context-Awareness and Score Fusion.

    Strategy:
    - Context-Aware Injection: Prepends section titles to text to provide structural grounding.
    - Max-Sim Aggregation: Scores against original query and semantic rewrites, taking the maximum.
    - Weighted Fusion: Harmonizes semantic scores with structural graph resonance scores.
    - Optimized Execution: Uses ONNX Runtime for CPU performance and LRU caching for latency.
    """

    if not candidates:
        return []

    # ---- Config ----
    rerank_top_k = int(os.getenv("RERANKER_TOP_K", "50"))
    output_k = top_k or int(os.getenv("OUTPUT_K", "5"))

    top_candidates = list(candidates)[: min(len(candidates), rerank_top_k)]

    # ---- Build query list ----
    import time
    t_start = time.perf_counter()
    
    # PART 2.1: Expansion & Context
    t_exp_start = time.perf_counter()
    queries = [query]
    if alternative_queries:
        for q in alternative_queries:
            if q and q != query:
                queries.append(q)
                if max_rewrites is not None and len(queries) - 1 >= max_rewrites:
                    break
    seen_q = set()
    queries = [q for q in queries if not (q in seen_q or seen_q.add(q))]

    for c in top_candidates:
        section = c.get("metadata", {}).get("section_title", "")
        text = c.get("text", "")
        contextual_text = f"Section: {section}\n{text}" if section else text
        c["_contextual_text"] = contextual_text
    t_exp_end = time.perf_counter()

    # PART 2.2: Cache Reconciliation
    t_cache_start = time.perf_counter()
    cached_scores: dict[tuple, float] = {}
    to_score_pairs: list[tuple] = []
    to_score_keys: list[tuple] = []
    request_unique_keys = set()

    for c in top_candidates:
        text = c["_contextual_text"]
        for q in queries:
            key = (q, text)
            if key in _SCORE_CACHE:
                cached_scores[key] = _SCORE_CACHE[key]
            elif key not in request_unique_keys:
                to_score_pairs.append((q, text))
                to_score_keys.append(key)
                request_unique_keys.add(key)

    total_pairs = len(top_candidates) * len(queries)
    actual_hits = total_pairs - len(to_score_pairs)
    t_cache_end = time.perf_counter()
    
    _LOG.info("[RERANK] Part 2.1 (Expansion): %.4fs | Part 2.2 (Cache): %.4fs (Hits: %d/%d)", 
             t_exp_end - t_exp_start, t_cache_end - t_cache_start, actual_hits, total_pairs)

    try:
        # PART 2.3: Model Inference
        model = get_reranker()
        if to_score_pairs:
            t_inf_start = time.perf_counter()
            scores = model.predict(to_score_pairs, batch_size=batch_size)
            t_inf_end = time.perf_counter()
            _LOG.info("[RERANK] Part 2.3 (Inference): %.4fs for %d pairs (%.2fms/pair)", 
                     t_inf_end - t_inf_start, len(to_score_pairs), ((t_inf_end - t_inf_start)/len(to_score_pairs))*1000)
            
            for key, sc in zip(to_score_keys, scores):
                _SCORE_CACHE[key] = sc
                cached_scores[key] = sc

        # ---- PHASE 3: AGGREGATE, FUSE & FILTER ----
        # PART 3.1: Fusion & Max-Sim
        t_fusion_start = time.perf_counter()
        for c in top_candidates:
            text = c["_contextual_text"]
            c_scores = [cached_scores.get((q, text), 0.0) for q in queries]
            max_rerank = max(c_scores) if c_scores else 0.0
            c["rerank_score"] = max_rerank

            if use_fusion:
                incoming_score = c.get("score", 0.0)
                c["final_score"] = (GRAPH_WEIGHT * incoming_score) + (RERANK_WEIGHT * max_rerank)
            else:
                c["final_score"] = max_rerank
        t_fusion_end = time.perf_counter()

        # PART 3.2: Dynamic Filtering
        t_filter_start = time.perf_counter()
        SCORE_THRESHOLD = 0.3 
        filtered = [c for c in top_candidates if c.get("final_score", 0) >= SCORE_THRESHOLD]
        
        GAP_THRESHOLD = 0.25  
        final_candidates = []
        if filtered:
            ranked_filtered = sorted(filtered, key=lambda x: -x["final_score"])
            final_candidates.append(ranked_filtered[0])
            for i in range(1, len(ranked_filtered)):
                gap = ranked_filtered[i-1]["final_score"] - ranked_filtered[i]["final_score"]
                if gap > GAP_THRESHOLD:
                    _LOG.info("[RERANK] Part 3.2: Gap Triggered at rank %d (Gap: %.4f)", i, gap)
                    break
                final_candidates.append(ranked_filtered[i])
        t_filter_end = time.perf_counter()
        
        _LOG.info("[RERANK] Part 3.1 (Fusion): %.4fs | Part 3.2 (Filtering): %.4fs (Survivors: %d/%d)", 
                 t_fusion_end - t_fusion_start, t_filter_end - t_filter_start, len(final_candidates), len(top_candidates))
        
        # PART 3.3: Audit & Cleanup
        t_audit_start = time.perf_counter()
        from collections import Counter
        accepted_origins = Counter()
        rejected_origins = Counter()
        
        # We consider 'final_candidates' as accepted, and top_candidates not in final_candidates as rejected
        final_ids = {c["doc_id"] for c in final_candidates}
        
        for c in top_candidates:
            origins = c.get("metadata", {}).get("origins", ["unknown"])
            if c["doc_id"] in final_ids:
                for o in origins: accepted_origins[o] += 1
            else:
                for o in origins: rejected_origins[o] += 1

        # Return only the best ones, capped by output_k (limit)
        final_results = final_candidates[: min(output_k, len(final_candidates))]
        t_audit_end = time.perf_counter()

        # Log survival count
        _LOG.info(
            "Reranked %d candidates using %d query variations (%d cache hits, %d scored). Found %d high-relevance, returning %d.",
            len(top_candidates),
            len(queries),
            actual_hits,
            len(to_score_pairs),
            len(final_candidates),
            len(final_results)
        )
        print(f"\n--- RETRIEVAL AUDIT REPORT ---")
        print(f"Accepted High-Relevance ({len(final_candidates)}): {dict(accepted_origins)}")
        print(f"Rejected Noise ({len(top_candidates) - len(final_candidates)}): {dict(rejected_origins)}")
        
        # Calculate 'Culprit' - which source has the highest rejection rate?
        print("Source Efficiency (Accepted / Total):")
        for src in set(list(accepted_origins.keys()) + list(rejected_origins.keys())):
            acc = accepted_origins[src]
            rej = rejected_origins[src]
            total = acc + rej
            print(f"  - {src.upper()}: {acc}/{total} ({acc/total:.1%})")
        print(f"------------------------------\n")

        print(f"INFO: Reranker found {len(final_candidates)} high-relevance candidates. Returning top {len(final_results)} to user.")
        _LOG.info("[RERANK] Part 3.3 (Audit): %.4fs | TOTAL TIME: %.4fs", t_audit_end - t_audit_start, time.perf_counter() - t_start)

        return final_results

    except Exception as e:
        _LOG.error("Reranking execution error: %s", e, exc_info=True)
        # Proceed with primary retrieval scores to maintain system availability
        for c in top_candidates:
            c["final_score"] = c.get("score", 0.0)
            
        ranked = sorted(
            top_candidates,
            key=lambda x: (-x["final_score"], x.get("doc_id", "")),
        )
        return ranked[: min(output_k, len(ranked))]
