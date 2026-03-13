from typing import List, Dict
import math
import hashlib


# ---------- CONFIG ----------
# MAX is a HARD limit (forces chunk split)
# MIN is ONLY for embedding eligibility (not chunk creation)
TARGET_MAX_TOKENS = 512
TARGET_MIN_TOKENS = 50
AVG_CHARS_PER_TOKEN = 4  # conservative estimate

# Font-size thresholds (relative, heuristic but locked)
FONT_LARGE = 14
FONT_MEDIUM = 12


# ---------- HELPERS ----------
def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / AVG_CHARS_PER_TOKEN)


def _stable_hash(value: str) -> str:
    """Short deterministic hash for IDs"""
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:8]


def infer_heading_level(block: Dict) -> int:
    """
    Deterministic heading level inference.

    Priority:
    1. metadata['heading_path'] (for web/structured data)
    2. Numeric headings (1, 1.1, 1.1.1)
    3. Font size from bbox (if available)
    4. Locked fallback (level 2)
    """

    metadata = block.get("metadata", {})
    if "heading_path" in metadata and metadata["heading_path"]:
        return len(metadata["heading_path"])

    text = block.get("text", "").strip()
    if not text:
        return 2

    # ---- Numeric headings ----
    if text[0].isdigit():
        return text.count(".") + 1

    # ---- Font-size heuristic ----
    bbox = block.get("bbox")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        # bbox: (x0, y0, x1, y1)
        height = abs(bbox[3] - bbox[1])
        if height >= FONT_LARGE:
            return 1
        if height >= FONT_MEDIUM:
            return 2
        return 3

    # ---- Locked fallback ----
    return 2


# ---------- CORE CHUNKER ----------
def semantic_chunk_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Convert layout-aware blocks into semantic, section-based chunks.

    Guarantees:
    - Deterministic chunk IDs
    - Hard max token enforcement
    - Stable section indexing per document
    """

    chunks: List[Dict] = []

    current_section = None
    current_blocks: List[Dict] = []

    section_index = -1
    chunk_index = 0

    def flush_section():
        nonlocal current_blocks, chunk_index

        if not current_section or not current_blocks:
            return

        text_parts = []
        pages = set()
        source_blocks = []

        for b in current_blocks:
            text_parts.append(b["text"])
            pages.add(b["page"])
            source_blocks.append(b["block_id"])

        merged_text = "\n".join(text_parts)
        token_estimate = estimate_tokens(merged_text)

        # deterministic ID seed
        id_seed = (
            f'{current_section["document_id"]}|'
            f'{section_index}|'
            f'{chunk_index}|'
            f'{min(pages)}'
        )

        chunks.append({
            "chunk_id": (
                f'{current_section["document_id"]}:'
                f'S{section_index}:'
                f'C{chunk_index}:'
                f'{_stable_hash(id_seed)}'
            ),
            "document_id": current_section["document_id"],
            "section_title": current_section["section_title"],
            "section_level": current_section["section_level"],
            "pages": sorted(pages),
            "chunk_type": "section_body",
            "text": merged_text,
            "source_blocks": source_blocks,
            "token_estimate": token_estimate,
            # MIN is an embedding gate, not a chunking rule
            "embedding_eligible": token_estimate >= TARGET_MIN_TOKENS
        })

        current_blocks = []
        chunk_index += 1

    # ---------- MAIN PASS ----------
    for block in blocks:
        block_type = block.get("block_type")

        # ---- New section ----
        if block_type == "heading":
            flush_section()

            section_index += 1
            chunk_index = 0

            current_section = {
                "document_id": block["document_id"],
                "section_title": block["text"],
                "section_level": infer_heading_level(block)
            }

            current_blocks = [block]
            continue

        # ---- Preamble ----
        if current_section is None:
            section_index += 1
            chunk_index = 0
            current_section = {
                "document_id": block["document_id"],
                "section_title": "Document Introduction",
                "section_level": 1
            }

        current_blocks.append(block)

        # ---- HARD size enforcement ----
        merged_preview = "\n".join(b["text"] for b in current_blocks)
        if estimate_tokens(merged_preview) >= TARGET_MAX_TOKENS:
            flush_section()

    # Flush tail
    flush_section()

    return chunks