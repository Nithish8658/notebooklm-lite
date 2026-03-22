import multiprocessing
import os
from dataclasses import dataclass
from functools import lru_cache


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default

    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Phase2RuntimeConfig:
    embedding_threads: int
    embedding_executor_workers: int
    embedding_batch_size: int
    embedding_batch_window_ms: int
    embedding_encode_batch_size: int
    tenant_online_global_limit: int
    tenant_online_per_tenant_limit: int
    tenant_background_global_limit: int
    tenant_background_per_tenant_limit: int
    tenant_admission_timeout_ms: int
    tenant_state_ttl_sec: int
    metadata_build_workers: int
    metadata_build_parallelism: int
    metadata_scroll_page_size: int
    retrieval_cpu_workers: int
    retrieval_cpu_parallelism: int
    graph_expand_max_hops: int
    graph_expand_max_total: int
    graph_skip_top_score: float
    graph_skip_candidate_count: int
    graph_skip_min_top_gap: float
    graph_skip_min_consensus_count: int
    candidate_max_per_document: int
    candidate_max_per_section: int
    enable_qdrant_payload_indexes: bool
    rerank_pool_size: int
    rerank_threads_per_session: int
    rerank_executor_workers: int
    rerank_batch_size: int
    rerank_batch_window_ms: int
    rerank_openvino_streams: int
    rerank_score_cache_size: int


@lru_cache(maxsize=1)
def get_phase2_config() -> Phase2RuntimeConfig:
    total_cores = max(1, multiprocessing.cpu_count())
    rerank_pool_default = max(1, min(3, total_cores // 4 or 1))
    retrieval_workers_default = max(2, min(8, total_cores // 2 or 1))
    session_threads_default = max(1, min(2, total_cores // max(1, rerank_pool_default * 2)))
    embedding_threads_default = max(1, min(2, total_cores // 4 or 1))
    metadata_workers_default = 1
    online_global_default = max(2, min(8, total_cores // 2 or 1))
    background_global_default = max(1, min(2, total_cores // 6 or 1))

    rerank_pool_size = _env_int("RERANK_POOL_SIZE", rerank_pool_default, minimum=1, maximum=max(1, total_cores))
    rerank_threads_per_session = _env_int(
        "RERANK_SESSION_THREADS",
        session_threads_default,
        minimum=1,
        maximum=max(1, total_cores),
    )
    retrieval_cpu_workers = _env_int(
        "PHASE2_CPU_WORKERS",
        retrieval_workers_default,
        minimum=1,
        maximum=max(1, total_cores),
    )

    return Phase2RuntimeConfig(
        embedding_threads=_env_int(
            "EMBEDDING_THREADS",
            embedding_threads_default,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        embedding_executor_workers=_env_int(
            "EMBEDDING_EXECUTOR_WORKERS",
            1,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 16, minimum=1, maximum=128),
        embedding_batch_window_ms=_env_int("EMBEDDING_BATCH_WINDOW_MS", 10, minimum=1, maximum=250),
        embedding_encode_batch_size=_env_int("EMBEDDING_ENCODE_BATCH_SIZE", 32, minimum=1, maximum=256),
        tenant_online_global_limit=_env_int(
            "ONLINE_REQUEST_GLOBAL_LIMIT",
            online_global_default,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        tenant_online_per_tenant_limit=_env_int(
            "ONLINE_REQUEST_PER_TENANT_LIMIT",
            2,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        tenant_background_global_limit=_env_int(
            "BACKGROUND_REQUEST_GLOBAL_LIMIT",
            background_global_default,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        tenant_background_per_tenant_limit=_env_int(
            "BACKGROUND_REQUEST_PER_TENANT_LIMIT",
            1,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        tenant_admission_timeout_ms=_env_int(
            "REQUEST_ADMISSION_TIMEOUT_MS",
            0,
            minimum=0,
            maximum=300000,
        ),
        tenant_state_ttl_sec=_env_int(
            "TENANT_STATE_TTL_SEC",
            600,
            minimum=60,
            maximum=86400,
        ),
        metadata_build_workers=_env_int(
            "METADATA_BUILD_WORKERS",
            metadata_workers_default,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        metadata_build_parallelism=_env_int(
            "METADATA_BUILD_PARALLELISM",
            metadata_workers_default,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        metadata_scroll_page_size=_env_int("METADATA_SCROLL_PAGE_SIZE", 512, minimum=64, maximum=4096),
        retrieval_cpu_workers=retrieval_cpu_workers,
        retrieval_cpu_parallelism=_env_int(
            "PHASE2_CPU_PARALLELISM",
            retrieval_cpu_workers,
            minimum=1,
            maximum=max(1, retrieval_cpu_workers),
        ),
        graph_expand_max_hops=_env_int("GRAPH_EXPAND_MAX_HOPS", 2, minimum=1, maximum=4),
        graph_expand_max_total=_env_int("GRAPH_EXPAND_MAX_TOTAL", 60, minimum=10, maximum=200),
        graph_skip_top_score=_env_float("GRAPH_SKIP_TOP_SCORE", 0.92, minimum=0.1, maximum=1.0),
        graph_skip_candidate_count=_env_int("GRAPH_SKIP_CANDIDATE_COUNT", 8, minimum=1, maximum=50),
        graph_skip_min_top_gap=_env_float("GRAPH_SKIP_MIN_TOP_GAP", 0.08, minimum=0.0, maximum=1.0),
        graph_skip_min_consensus_count=_env_int("GRAPH_SKIP_MIN_CONSENSUS_COUNT", 3, minimum=1, maximum=20),
        candidate_max_per_document=_env_int("PHASE2_MAX_PER_DOCUMENT", 3, minimum=1, maximum=20),
        candidate_max_per_section=_env_int("PHASE2_MAX_PER_SECTION", 2, minimum=1, maximum=20),
        enable_qdrant_payload_indexes=os.getenv("ENABLE_QDRANT_PAYLOAD_INDEXES", "1").lower() not in {"0", "false", "no"},
        rerank_pool_size=rerank_pool_size,
        rerank_threads_per_session=rerank_threads_per_session,
        rerank_executor_workers=_env_int(
            "RERANK_EXECUTOR_WORKERS",
            rerank_pool_size,
            minimum=1,
            maximum=max(1, total_cores),
        ),
        rerank_batch_size=_env_int("RERANK_BATCH_SIZE", 16, minimum=1, maximum=128),
        rerank_batch_window_ms=_env_int("RERANK_BATCH_WINDOW_MS", 15, minimum=1, maximum=250),
        rerank_openvino_streams=_env_int("RERANK_OPENVINO_STREAMS", 1, minimum=1, maximum=8),
        rerank_score_cache_size=_env_int("SCORE_CACHE_SIZE", 50000, minimum=1000, maximum=500000),
    )
