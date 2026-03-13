import bisect
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK data is available for keyword extraction
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def _extract_keywords(text: str) -> Set[str]:
    """
    Simplified keyword extraction for graph linking.
    """
    try:
        stop_words = set(stopwords.words("english"))
    except:
        stop_words = set()
    
    stemmer = PorterStemmer()
    
    # Tokenize and normalize
    raw_tokens = re.split(r"[^\w]+", text.lower())
    
    processed = set()
    for t in raw_tokens:
        if len(t) < 3 or t in stop_words:
            continue
        processed.add(stemmer.stem(t))
    return processed


def build_chunk_graph(chunks: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build an upgraded, high-potential chunk graph.
    
    Upgrades:
    - Linear Section Flow (Linearizes doc flow instead of N x M bloat)
    - Thematic Keyword Linking (Cross-document discovery)
    - Hierarchical Breadth (Sibling edges)
    - Safe Section Navigation
    """

    graph = defaultdict(lambda: defaultdict(list))
    if not chunks:
        return graph

    # ---- Indexing & Pre-processing ----
    by_doc = defaultdict(list)
    keyword_index = defaultdict(list)
    
    def _safe_page(c: Dict) -> int:
        pgs = c.get("pages", [])
        if not pgs: return 0
        val = pgs[0]
        return val if val is not None else 0

    for c in chunks:
        cid = c["chunk_id"]
        doc_id = c["document_id"]
        c["_page"] = _safe_page(c)
        by_doc[doc_id].append(c)
        
        # Keyword extraction for thematic linking
        keywords = _extract_keywords(c.get("text", ""))
        for kw in keywords:
            keyword_index[kw].append(cid)

    # ---- 1. THEMATIC KEYWORD EDGES (Cross-Doc) ----
    # Link chunks that share multiple keywords
    chunk_to_keywords = {c["chunk_id"]: _extract_keywords(c.get("text", "")) for c in chunks}
    
    # To keep the graph sparse and meaningful, we only link if overlap > threshold
    # and we limit the number of thematic edges per chunk.
    for cid, kws in chunk_to_keywords.items():
        if not kws: continue
        
        candidates = defaultdict(int)
        for kw in kws:
            for neighbor_id in keyword_index[kw]:
                if neighbor_id != cid:
                    candidates[neighbor_id] += 1
        
        # Take top 3 most similar chunks by keyword overlap
        top_related = sorted(candidates.items(), key=lambda x: -x[1])[:3]
        for neighbor_id, overlap in top_related:
            if overlap >= 2: # At least 2 shared stems
                graph[cid]["thematic_related"].append(neighbor_id)

    # ---- 2. DOCUMENT STRUCTURAL EDGES ----
    for doc_id, doc_chunks in by_doc.items():
        # Sort chunks globally for the document to identify linear flow
        doc_ordered = sorted(doc_chunks, key=lambda c: (c["_page"], c["chunk_id"]))
        
        # Organize by sections
        sections = defaultdict(list)
        section_metadata = {} # section_id -> {level, title, first_chunk_id}

        # Identifying sections and their linear order
        current_section_key = None
        ordered_section_keys = []
        
        for c in doc_ordered:
            sec_key = (c.get("section_level", 1), c.get("section_title", "General"))
            if sec_key not in sections:
                ordered_section_keys.append(sec_key)
            sections[sec_key].append(c)

        # ---- Linear Same-Section Edges ----
        for sec_key, sec_chunks in sections.items():
            sec_ordered = sorted(sec_chunks, key=lambda x: (x["_page"], x["chunk_id"]))
            for i in range(len(sec_ordered) - 1):
                a, b = sec_ordered[i]["chunk_id"], sec_ordered[i+1]["chunk_id"]
                graph[a]["same_section"].append(b)
                graph[b]["same_section"].append(a)
            
            # Store first chunk for hierarchy
            section_metadata[sec_key] = {
                "level": sec_key[0],
                "first_chunk_id": sec_ordered[0]["chunk_id"],
                "last_chunk_id": sec_ordered[-1]["chunk_id"]
            }

        # ---- Linear Section Transitions (The 1-to-1 Fix) ----
        for i in range(len(ordered_section_keys) - 1):
            curr_sec = ordered_section_keys[i]
            next_sec = ordered_section_keys[i+1]
            
            last_of_curr = section_metadata[curr_sec]["last_chunk_id"]
            first_of_next = section_metadata[next_sec]["first_chunk_id"]
            
            graph[last_of_curr]["next_section"].append(first_of_next)
            graph[first_of_next]["previous_section"].append(last_of_curr)

        # ---- Hierarchical (Parent/Sibling) Edges ----
        # Parent lookup via page-aware level nesting
        for i, sec_key in enumerate(ordered_section_keys):
            level = sec_key[0]
            first_cid = section_metadata[sec_key]["first_chunk_id"]
            
            # Find Parent: Nearest preceding section with level < current level
            parent_key = None
            for j in range(i - 1, -1, -1):
                prev_sec = ordered_section_keys[j]
                if prev_sec[0] < level:
                    parent_key = prev_sec
                    break
            
            if parent_key:
                parent_first_cid = section_metadata[parent_key]["first_chunk_id"]
                graph[first_cid]["parent_section"].append(parent_first_cid)
                graph[parent_first_cid]["child_section"].append(first_cid)

            # Find Siblings: Other sections with same parent and same level
            # (Simplification: just link to previous section if same level)
            if i > 0:
                prev_sec = ordered_section_keys[i-1]
                if prev_sec[0] == level:
                    prev_first_cid = section_metadata[prev_sec]["first_chunk_id"]
                    graph[first_cid]["sibling_section"].append(prev_first_cid)
                    graph[prev_first_cid]["sibling_section"].append(first_cid)

    return graph
