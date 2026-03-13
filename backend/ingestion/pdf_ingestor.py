import fitz  # PyMuPDF
# import pytesseract  <-- Removed
from rapidocr_onnxruntime import RapidOCR
import tempfile
import os
import threading
from pathlib import Path
from typing import List, Dict
from PIL import Image
import dask
from dask import delayed
from fastapi import HTTPException
import traceback

# ---------------- CONFIG ----------------
HEADING_FONT_THRESHOLD = 1.25
LIST_MARKERS = ("-", "•", "*", "–", "—")

# Global lock for thread-unsafe table extraction
table_extraction_lock = threading.Lock()

# OCR CONFIG
# Initialize RapidOCR once (it uses ONNXRuntime, very efficient)
try:
    ocr_engine = RapidOCR()
except Exception:
    print("Warning: RapidOCR failed to load. OCR will be unavailable.")
    ocr_engine = None

from img2table.document import PDF
from img2table.ocr import TesseractOCR

# ---------------------------------------
# PAGE ANALYZER
# ---------------------------------------
def analyze_page(page: fitz.Page) -> Dict:
    text = page.get_text().strip()
    images = page.get_images(full=True)

    return {
        "has_text_layer": bool(text),
        "text_length": len(text),
        "image_count": len(images),
        "is_scanned": len(text) < 30 and len(images) > 0,
    }

# ---------------------------------------
# TABLE EXTRACTION (Robust)
# ---------------------------------------
def extract_tables_safe(file_path: str, page_index: int, document_id: str) -> List[Dict]:
    """
    Extract tables from a specific page using img2table.
    Returns blocks of type 'table' with Markdown content.
    """
    # Check if table extraction is globally disabled via ENV
    if os.getenv("DISABLE_TABLE_EXTRACTION", "false").lower() == "true":
        print("[TABLE] Table extraction is disabled via environment variable.")
        return []

    # Use lock to prevent concurrent file handle collisions in PDFium
    with table_extraction_lock:
        print(f"[TABLE] Scanning Page {page_index} for tables...")
        blocks = []
        
        try:
            # img2table works on the whole file, but we filter by pages argument to be efficient
            # pages is 0-indexed list
            pdf = PDF(src=file_path, pages=[page_index - 1]) 
            
            # Extract tables (Digital only for speed, or add ocr=... for scanned)
            # Using implicit_rows=False prevents false positives on grid-like text
            extracted_tables = pdf.extract_tables(ocr=None, implicit_rows=False, borderless_tables=False, min_confidence=50)
            
            # Extracted_tables is {page_idx: [Table objects]}
            page_tables = extracted_tables.get(page_index - 1, [])
            
            if not page_tables:
                print(f"[TABLE] No tables found on Page {page_index}.")
                return []

            print(f"[TABLE] SUCCESS: Found {len(page_tables)} tables on Page {page_index}.")
            
            for i, table in enumerate(page_tables):
                # ... [rest of the logic remains inside the try block] ...
                raw_grid = []
                # table.content is {row_idx: {col_idx: Cell object}}
                # We need to reconstruct the grid 
                
                if hasattr(table, "content") and table.content:
                    # Handle both dictionary {row: {col: cell}} and list [[cell]] formats
                    if isinstance(table.content, dict):
                        max_row = max(table.content.keys())
                        max_col = max(c for r in table.content.values() for c in r.keys())
                        
                        for r in range(max_row + 1):
                            row_data = []
                            for c in range(max_col + 1):
                                cell = table.content.get(r, {}).get(c)
                                cell_text = cell.value.strip() if cell and hasattr(cell, "value") and cell.value else ""
                                row_data.append(cell_text)
                            raw_grid.append(row_data)
                    elif isinstance(table.content, list):
                        for row in table.content:
                            row_data = []
                            for cell in row:
                                cell_text = cell.value.strip() if hasattr(cell, "value") and cell.value else str(cell)
                                row_data.append(cell_text)
                            raw_grid.append(row_data)
                else:
                    # Fallback if content not accessible: parse markdown (brittle but functional)
                    # skipping for now to rely on content.
                    continue

                # --- HEADER DETECTION HEURISTIC (Text-Based) ---
                header_score = 0
                header_inferred = False
                
                if len(raw_grid) > 1:
                    row0 = raw_grid[0]
                    row1 = raw_grid[1]
                    
                    # Metric 1: Data Type Variance (+2)
                    # If Row 0 is all text and Row 1 is mostly numbers -> Strong signal
                    r0_nums = sum(1 for c in row0 if c.replace('.', '', 1).isdigit())
                    r1_nums = sum(1 for c in row1 if c.replace('.', '', 1).isdigit())
                    
                    if r0_nums == 0 and r1_nums > len(row1) / 2:
                        header_score += 2
                    
                    # Metric 2: Length Variance (+1)
                    # Headers are usually shorter labels
                    avg_len0 = sum(len(c) for c in row0) / len(row0) if row0 else 0
                    avg_len1 = sum(len(c) for c in row1) / len(row1) if row1 else 0
                    
                    if avg_len0 < avg_len1 * 0.8: # 20% shorter
                        header_score += 1
                
                # Conservative Threshold
                header_inferred = header_score >= 2
                
                # Force False if UI flag says so (not available here) or if empty
                if not raw_grid:
                    header_inferred = False

                # Create visual representation
                # If header is inferred, use Pipes (Table).
                # If NOT, use simple spacing (List/Text).
                table_lines = []
                title_text = table.title if table.title else ""
                if title_text:
                    table_lines.append(title_text)
                
                for row in raw_grid:
                    if header_inferred:
                        table_lines.append("| " + " | ".join(row) + " |")
                    else:
                        table_lines.append("  ".join(row))

                blocks.append({
                    "document_id": document_id,
                    "page": page_index,
                    "block_id": f"p{page_index}_tbl{i}",
                    "block_type": "table",
                    "text": "\n".join(table_lines).strip(),
                    "metadata": {
                        "header_inferred": header_inferred,
                        "header_score": header_score,
                        "raw_grid": raw_grid,
                        "rows": len(raw_grid)
                    },
                    "bbox": (table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2) if table.bbox else None,
                    "source": "table_extractor",
                    "confidence": 0.95,
                })
                
            return blocks

        except Exception as e:
            print(f"[TABLE] FAILED on Page {page_index}: {e}")
            return []

# ---------------------------------------
# IMAGE → TEXT DESCRIPTION (GIT VLM)
# ---------------------------------------
def describe_image(image_path: str) -> str:
    """
    Stub: replace with Gemini / BLIP / LLaVA call.
    Must return plain descriptive text.
    """
    return "Image depicting a diagram or figure relevant to the document content."

# ---------------------------------------

# OCR PROCESSOR

# ---------------------------------------

def ocr_page(page: fitz.Page, document_id: str, page_index: int) -> List[Dict]:

    print(f"[OCR] Processing Page {page_index}...")

    pix = page.get_pixmap(dpi=300)

    

    # Create a temp file but close it immediately so other libs can use the path

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:

        tmp_name = tmp.name



    try:

        pix.save(tmp_name)

        

        # Use RapidOCR instead of Tesseract

        text = ""

        if ocr_engine:

            result, _ = ocr_engine(tmp_name)

            if result:

                # result format: [[[[x1,y1],...], "text", confidence], ...]

                text = "\n".join([line[1] for line in result])

        

        if text.strip():

            print(f"[OCR] SUCCESS: Extracted {len(text)} chars from Page {page_index}.")

        else:

            print(f"[OCR] WARNING: No text found on Page {page_index}.")



    except Exception as e:

        print(f"[OCR] FAILED on Page {page_index}: {e}")

        return []



    finally:

        if os.path.exists(tmp_name):

            try:

                os.unlink(tmp_name)

            except PermissionError:

                pass  # Sometimes Windows holds on a bit too long



    if not text.strip():

        return []



    return [{

        "document_id": document_id,

        "page": page_index,

        "block_id": f"p{page_index}_ocr",

        "block_type": "paragraph",

        "text": text.strip(),

        "bbox": None,

        "source": "ocr",

        "confidence": 0.80,

    }]



# ---------------------------------------

# TEXT BLOCK EXTRACTION

# ---------------------------------------

def extract_text_blocks(page_dict, document_id, page_index) -> List[Dict]:

    print(f"[TEXT] Extracting blocks from Page {page_index}...")

    font_sizes = []

    for blk in page_dict.get("blocks", []):

        for line in blk.get("lines", []):

            for span in line.get("spans", []):

                if span.get("size"):

                    font_sizes.append(span["size"])



    median_font = sorted(font_sizes)[len(font_sizes)//2] if font_sizes else 0.0

    blocks = []



    for block_index, block in enumerate(page_dict.get("blocks", [])):

        if "lines" not in block:

            continue



        lines, sizes, list_count = [], [], 0

        for line in block["lines"]:

            spans = line.get("spans", [])

            if not spans:

                continue



            text_parts, max_size = [], 0

            for span in spans:

                t = span.get("text", "").strip()

                if t:

                    text_parts.append(t)

                    max_size = max(max_size, span.get("size", 0))



            if not text_parts:

                continue



            line_text = " ".join(text_parts)

            lines.append(line_text)

            sizes.append(max_size)

            if line_text.lstrip().startswith(LIST_MARKERS):

                list_count += 1



        if not lines:

            continue



        max_font = max(sizes) if sizes else 0

        is_heading = median_font > 0 and max_font >= median_font * HEADING_FONT_THRESHOLD

        is_list = list_count >= len(lines) / 2



        block_type = "heading" if is_heading else "list" if is_list else "paragraph"



        blocks.append({

            "document_id": document_id,

            "page": page_index,

            "block_id": f"p{page_index}_b{block_index}",

            "block_type": block_type,

            "text": "\n".join(lines),

            "bbox": block.get("bbox"),

            "source": "text",

            "confidence": 1.0,

        })



    print(f"[TEXT] SUCCESS: Found {len(blocks)} blocks on Page {page_index}.")

    return blocks



# ---------------------------------------

# IMAGE BLOCK EXTRACTION

# ---------------------------------------

def extract_images(page: fitz.Page, document_id: str, page_index: int) -> List[Dict]:

    images = page.get_images(full=True)

    if not images:

        return []

    

    print(f"[IMAGE] Found {len(images)} images on Page {page_index}. Processing...")

    blocks = []

    for i, img in enumerate(images):

        xref = img[0]

        pix = fitz.Pixmap(page.parent, xref)



        # Create a temp file but close it immediately

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:

            tmp_name = tmp.name

        

        try:

            pix.save(tmp_name)

            description = describe_image(tmp_name)

            print(f"[IMAGE] SUCCESS: Processed Image {i} on Page {page_index}.")

        except Exception as e:

            print(f"[IMAGE] FAILED for Image {i} on Page {page_index}: {e}")

            description = "Error processing image."

        finally:

            if os.path.exists(tmp_name):

                try:

                    os.unlink(tmp_name)

                except PermissionError:

                    pass



        blocks.append({

            "document_id": document_id,

            "page": page_index,

            "block_id": f"p{page_index}_img{i}",

            "block_type": "image",

            "text": description,

            "bbox": None,

            "source": "vision",

            "confidence": 0.85,

        })

    return blocks

# ---------------------------------------
# PAGE WORKER
# ---------------------------------------
def _process_page(file_path: str, page_index: int, document_id: str) -> List[Dict]:
    blocks: List[Dict] = []

    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_index - 1)
        analysis = analyze_page(page)

        # 1. TEXT or OCR
        if analysis["has_text_layer"]:
            page_dict = page.get_text("dict")
            blocks.extend(
                extract_text_blocks(page_dict, document_id, page_index)
            )
        else:
            blocks.extend(
                ocr_page(page, document_id, page_index)
            )

        # 2. TABLES (safe, independent)
        blocks.extend(
            extract_tables_safe(file_path, page_index, document_id)
        )

        # 3. IMAGES (always attempt)
        blocks.extend(
            extract_images(page, document_id, page_index)
        )

        return blocks

    except Exception as e:
        print(f"[ERROR] Page {page_index} failed: {e}")
        traceback.print_exc()
        return []

    finally:
        try:
            doc.close()
        except Exception:
            pass
# ---------------------------------------
# MAIN INGEST FUNCTION
# ---------------------------------------
def ingest_pdf_layout_aware(file_path: Path, document_id: str) -> List[Dict]:
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(415, "Only PDF files are supported")

    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
    except Exception as e:
        raise HTTPException(422, f"Invalid PDF: {e}")
    finally:
        if 'doc' in locals():
            doc.close()

    if total_pages == 0:
        raise HTTPException(422, "PDF contains zero pages")

    tasks = [
        delayed(_process_page)(str(file_path), i, document_id)
        for i in range(1, total_pages + 1)
    ]
    
    # Use 'threads' for parallel page processing
    results = dask.compute(*tasks, scheduler="threads")

    all_blocks = [b for page in results for b in page]

    if not all_blocks:
        raise HTTPException(
            status_code=422,
            detail="Please upload a valid PDF."
        )

    return all_blocks