import os
import tempfile
from pathlib import Path
from typing import List, Dict
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from rapidocr_onnxruntime import RapidOCR

# Initialize RapidOCR
try:
    ocr_engine = RapidOCR()
except Exception:
    print("Warning: RapidOCR failed to load. OCR will be unavailable.")
    ocr_engine = None

def ocr_image(image_bytes: bytes) -> str:
    """Fallback OCR for images."""
    if not ocr_engine:
        return "OCR Unavailable"
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_name = tmp.name
    
    try:
        result, _ = ocr_engine(tmp_name)
        if result:
            return "\n".join([line[1] for line in result])
    except Exception as e:
        print(f"[PPTX-OCR] Failed: {e}")
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except PermissionError:
                pass
    return ""

def ingest_pptx_structure_aware(file_path: Path, document_id: str) -> List[Dict]:
    """
    STRICT PPTX Ingestor:
    1. Primary parser: python-pptx
    2. Native structure-first
    3. OCR as fallback ONLY
    4. Preserves titles, text, tables, notes, images, charts.
    """
    if not file_path.exists():
        return []

    prs = Presentation(file_path)
    blocks = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        # 1. Slide Title (if exists)
        title_shape = slide.shapes.title
        if title_shape:
            blocks.append({
                "document_id": document_id,
                "page": slide_idx,
                "block_id": f"s{slide_idx}_title",
                "block_type": "heading",
                "text": title_shape.text.strip(),
                "bbox": None,
                "source": "pptx_native",
                "confidence": 1.0
            })

        # 2. Sort Shapes for Reading Order (Top -> Bottom, Left -> Right)
        # Filter out the title since we already processed it
        content_shapes = [s for s in slide.shapes if s != title_shape]
        
        # Sort key: primary=top (y), secondary=left (x)
        # using a small threshold for 'top' could help with slight misalignments, but exact sort is a good start.
        content_shapes.sort(key=lambda s: (s.top if hasattr(s, 'top') else 0, s.left if hasattr(s, 'left') else 0))

        # 3. Iterate Sorted Shapes
        for shape_idx, shape in enumerate(content_shapes):

            # Text Boxes / Placeholders
            if shape.has_text_frame:
                text_content = []
                paragraph_metadata = []
                
                for paragraph in shape.text_frame.paragraphs:
                    # Preserve bullet levels/hierarchy
                    text_content.append(paragraph.text)
                    paragraph_metadata.append({
                        "level": paragraph.level,
                        "text": paragraph.text
                    })
                
                full_text = "\n".join([p['text'] for p in paragraph_metadata]).strip()
                if full_text:
                    blocks.append({
                        "document_id": document_id,
                        "page": slide_idx,
                        "block_id": f"s{slide_idx}_sh{shape_idx}",
                        "block_type": "paragraph" if len(paragraph_metadata) == 1 else "list",
                        "text": full_text,
                        "metadata": {
                            "paragraphs": paragraph_metadata,
                            "is_list": len(paragraph_metadata) > 1
                        },
                        "bbox": (shape.left, shape.top, shape.width, shape.height),
                        "source": "pptx_native",
                        "confidence": 1.0
                    })

            # Tables (Exact row/column preservation)
            elif shape.has_table:
                raw_grid = []
                table_lines = []
                
                # --- HEADER DETECTION HEURISTIC ---
                header_score = 0
                first_row = shape.table.rows[0]
                
                # Signal 1: UI Flag (+1)
                if shape.table.first_row:
                    header_score += 1
                
                # Analyze First Row Style
                row0_bold_count = 0
                row0_fonts = []
                row0_fills = 0
                
                total_cells = len(first_row.cells)
                
                for cell in first_row.cells:
                    # Check Bold (+2 if ALL cells bold)
                    is_bold = False
                    # Check Fill (+1 if ANY cell has solid fill)
                    if cell.fill.type == 1: # msoFillSolid
                         row0_fills += 1
                         
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            if run.font.bold:
                                is_bold = True
                            if run.font.size:
                                row0_fonts.append(run.font.size.pt)
                    
                    if is_bold:
                        row0_bold_count += 1

                # Apply Scoring Rules
                if total_cells > 0:
                    # +2 if all cells are bold
                    if row0_bold_count == total_cells:
                        header_score += 2
                    
                    # +1 if background fill present (heuristic: > 50% cells filled)
                    if row0_fills > total_cells / 2:
                        header_score += 1

                    # +2 if font size > median (requires scanning other rows, simplified here to "is large?")
                    # For now, we omit relative font comparison to keep it fast, or we could scan row 1.
                    # Let's verify against Row 1 if it exists
                    if len(shape.table.rows) > 1:
                        row1_fonts = []
                        for cell in shape.table.rows[1].cells:
                            for p in cell.text_frame.paragraphs:
                                for r in p.runs:
                                    if r.font.size: row1_fonts.append(r.font.size.pt)
                        
                        if row0_fonts and row1_fonts:
                            avg0 = sum(row0_fonts)/len(row0_fonts)
                            avg1 = sum(row1_fonts)/len(row1_fonts)
                            if avg0 > avg1:
                                header_score += 2

                header_inferred = header_score >= 3
                # ----------------------------------

                for row in shape.table.rows:
                    row_cells = [cell.text_frame.text.strip() for cell in row.cells]
                    raw_grid.append(row_cells)
                    
                    # GENERATE TEXT REPRESENTATION
                    # If header is inferred, we treat it as a proper table (Pipes).
                    # If NOT inferred, we treat it as a list/grid of text (Tabs/Spaces), 
                    # avoiding strong Markdown table bias in embeddings.
                    if header_inferred:
                         table_lines.append("| " + " | ".join(row_cells) + " |")
                    else:
                        # Fallback: Tab-separated or separated by "  " to preserve some visual spacing 
                        # without implying a strict schema.
                        table_lines.append("  ".join(row_cells))
                
                if raw_grid:
                    blocks.append({
                        "document_id": document_id,
                        "page": slide_idx,
                        "block_id": f"s{slide_idx}_tbl{shape_idx}",
                        "block_type": "table",
                        "text": "\n".join(table_lines),
                        "metadata": {
                            "table_rows": len(shape.table.rows),
                            "table_cols": len(shape.table.columns),
                            "header_inferred": header_inferred,
                            "header_score": header_score,
                            "raw_grid": raw_grid
                        },
                        "bbox": (shape.left, shape.top, shape.width, shape.height),
                        "source": "pptx_native",
                        "confidence": 1.0
                    })

            # Pictures / Images (OCR Fallback)
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                ocr_text = ocr_image(shape.image.blob)
                if ocr_text.strip():
                    blocks.append({
                        "document_id": document_id,
                        "page": slide_idx,
                        "block_id": f"s{slide_idx}_img{shape_idx}",
                        "block_type": "paragraph",
                        "text": f"[Image OCR]: {ocr_text.strip()}",
                        "bbox": (shape.left, shape.top, shape.width, shape.height),
                        "source": "pptx_ocr",
                        "confidence": 0.8
                    })

            # Charts / SmartArt (Visual Blocks - Unparsed or Placeholder)
            elif shape.shape_type in (MSO_SHAPE_TYPE.CHART, MSO_SHAPE_TYPE.GROUP):
                 blocks.append({
                    "document_id": document_id,
                    "page": slide_idx,
                    "block_id": f"s{slide_idx}_sh{shape_idx}_vis",
                    "block_type": "unparsed",
                    "text": f"Visual content: {shape.name} ({shape.shape_type})",
                    "reason": "Complex visual object (Chart/SmartArt/Group) requiring vision-based parsing for full extraction.",
                    "bbox": (shape.left, shape.top, shape.width, shape.height),
                    "source": "pptx_native",
                    "confidence": 0.5
                })

        # 3. Speaker Notes
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                blocks.append({
                    "document_id": document_id,
                    "page": slide_idx,
                    "block_id": f"s{slide_idx}_notes",
                    "block_type": "paragraph",
                    "text": f"Speaker Notes: {notes_text}",
                    "bbox": None,
                    "source": "pptx_native_notes",
                    "confidence": 1.0
                })

    return blocks
