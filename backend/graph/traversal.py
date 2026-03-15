from typing import Dict, List, Set, Optional
from collections import defaultdict

# ---- EDGE CONDUCTANCE CONFIG ----
# Higher values mean the score flows more easily through these edges.
EDGE_WEIGHTS = {
    "same_section": 0.92,      # Very high: contextually identical
    "thematic_related": 0.85,  # High: semantic connection
    "parent_section": 0.75,    # Medium-High: structural context
    "child_section": 0.75,     # Medium-High: structural detail
    "sibling_section": 0.65,   # Medium: related topics in same parent
    "next_section": 0.55,      # Medium-Low: narrative flow
    "previous_section": 0.45,  # Low: backwards context
}

DEFAULT_HOP_DECAY = 0.80
MIN_EXPANSION_SCORE = 0.15 # Drift Guard: don't expand very weak nodes


def multi_hop_expand(
    seed_scores: Dict[str, float],
    graph: Dict[str, Dict[str, List[str]]],
    chunk_lookup: Dict[str, dict], 
    batch_id: str,               
    *,
    max_hops: int = 3,
    max_total: int = 40,
    resonance_factor: float = 0.3, 
    use_cache_only: bool = False, # Added for stateless mode
) -> Dict[str, float]:
    """
    Upgraded Multi-Hop Expansion with Path Resonance and Priority Flow.
    Now with strict Batch Firewall to reduce reranker load.
    """

    if not seed_scores:
        return {}

    # Final scores map
    expanded: Dict[str, float] = dict(seed_scores)
    
    # We use a set to track which nodes we have already EXPANED (not just visited)
    expanded_set: Set[str] = set()

    # Current frontier: (chunk_id, current_path_score)
    current_frontier = sorted(seed_scores.items(), key=lambda x: -x[1])

    for hop in range(1, max_hops + 1):
        next_frontier_map = defaultdict(float)
        
        for cid, base_score in current_frontier:
            if cid in expanded_set or base_score < MIN_EXPANSION_SCORE:
                continue
            
            expanded_set.add(cid)
            neighbors = graph.get(cid, {})
            if not neighbors:
                continue

            for edge_type, weight in EDGE_WEIGHTS.items():
                target_nodes = neighbors.get(edge_type, [])
                if not target_nodes:
                    continue
                
                # Conductance = path_score * edge_weight * hop_decay
                incoming_score = base_score * weight * DEFAULT_HOP_DECAY
                
                for nbr in target_nodes:
                    # BATCH FIREWALL: Verify neighbor belongs to same batch before adding
                    if not use_cache_only:
                        nbr_chunk = chunk_lookup.get(nbr)
                        if not nbr_chunk:
                            continue
                        
                        nbr_batch = nbr_chunk.get("batch_id") or nbr_chunk.get("metadata", {}).get("batch_id")
                        if str(nbr_batch) != str(batch_id):
                            continue

                    # Path Resonance
                    if nbr in expanded:
                        old_score = expanded[nbr]
                        boost = (1.0 - old_score) * incoming_score * resonance_factor
                        expanded[nbr] = old_score + boost
                    else:
                        expanded[nbr] = incoming_score
                    
                    next_frontier_map[nbr] = max(next_frontier_map[nbr], incoming_score)

        # Prepare for next hop: Sort by score and filter
        if not next_frontier_map:
            break
            
        current_frontier = sorted(
            next_frontier_map.items(), 
            key=lambda x: -x[1]
        )

    # ---- FINAL TRUNCATION & STABILIZATION ----
    seeds = set(seed_scores.keys())
    
    ranked = sorted(
        expanded.items(),
        key=lambda x: (-x[1], x[0]),
    )

    final_results = {cid: expanded[cid] for cid in seed_scores}
    
    for cid, score in ranked:
        if len(final_results) >= max_total:
            break
        if cid not in final_results:
            final_results[cid] = score
            
    return final_results
