from typing import Dict


def _clamp_alpha(alpha: float) -> float:
    """
    Clamp alpha to [0.0, 1.0].
    """
    try:
        alpha = float(alpha)
    except Exception:
        return 0.5

    if alpha < 0.0:
        return 0.0
    if alpha > 1.0:
        return 1.0
    return alpha


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Robust Min-Max normalization to [0, 1].
    Prevents outlier dominance and preserves relative distribution.
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)
    
    # Handle edge case: all scores are identical
    if max_v == min_v:
        return {k: 1.0 if max_v > 0 else 0.0 for k in scores}

    denom = max_v - min_v
    return {k: (v - min_v) / denom for k, v in scores.items()}


def hybrid_retrieve(
    dense_scores: Dict[str, float],
    sparse_scores: Dict[str, float],
    alpha: float = 0.6, # In RRF, this can be used to weight one list over another
    k: int = 20,        # RRF constant (tuned for chunk-level retrieval)
) -> Dict[str, float]:
    """
    Combine dense and sparse retrieval scores using Reciprocal Rank Fusion (RRF).
    
    Formula: score = sum( 1.0 / (rank + k) )
    
    This is more robust than score normalization because it doesn't care about 
    the different scales of BM25 (unbounded) vs Dense (cosine/dot).
    """
    alpha = _clamp_alpha(alpha)

    # Short-circuit
    if not dense_scores and not sparse_scores:
        return {}

    # 1. Sort both to get ranks (using .items() for efficiency)
    ranked_dense = [
        item[0] for item in 
        sorted(dense_scores.items(), key=lambda x: (-x[1], x[0]))
    ]
    ranked_sparse = [
        item[0] for item in 
        sorted(sparse_scores.items(), key=lambda x: (-x[1], x[0]))
    ]

    hybrid_scores = {}

    # 2. Apply RRF for Dense
    # We can multiply by alpha to keep the user's preference for dense/sparse
    for rank, cid in enumerate(ranked_dense):
        score = alpha * (1.0 / (rank + 1 + k))
        hybrid_scores[cid] = hybrid_scores.get(cid, 0.0) + score

    # 3. Apply RRF for Sparse
    for rank, cid in enumerate(ranked_sparse):
        score = (1.0 - alpha) * (1.0 / (rank + 1 + k))
        hybrid_scores[cid] = hybrid_scores.get(cid, 0.0) + score

    # 4. Normalize final scores to [0, 1] for the rest of the pipeline
    if not hybrid_scores:
        return {}
        
    max_h = max(hybrid_scores.values())
    for cid in hybrid_scores:
        hybrid_scores[cid] /= max_h

    return dict(
        sorted(
            hybrid_scores.items(),
            key=lambda x: (-x[1], x[0]), # Sort by score (desc), then ID (asc)
        )
    )