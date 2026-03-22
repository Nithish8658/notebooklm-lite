"""
Deterministic Streaming Excel Ingestor
Slices Excel sheets into exact 1,000-character CSV segments.
"""

import pandas as pd
import io
from pathlib import Path
from typing import List, Dict, Any
import uuid

def ingest_excel(file_path: Path, document_id: str) -> List[Dict[str, Any]]:
    """
    Treats each sheet in an Excel file as a continuous character stream.
    Converts to CSV and slices into fixed-size 1,000 character chunks.
    """
    all_segments = []
    
    try:
        # Load workbook
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        print(f"CRITICAL: Failed to open Excel file {file_path}: {e}")
        return []

    for sheet_index, sheet_name in enumerate(xls.sheet_names, start=1):
        try:
            # 1. Convert Sheet to Raw CSV String
            # header=None ensures we treat the first row (headers) as data 
            # so it is captured in the first chunk.
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str).fillna("")
            
            output = io.StringIO()
            # Use lineterminator='\n' to ensure 1-character newline count as per requirement
            df.to_csv(output, index=False, header=False, lineterminator='\n')
            csv_content = output.getvalue()
            
            if not csv_content:
                continue

            # 2. Sequential Slicing (The "Hard Cut" Logic)
            chunk_size = 1000
            total_chars = len(csv_content)
            
            for i in range(0, total_chars, chunk_size):
                chunk_text = csv_content[i : i + chunk_size]
                seq_index = (i // chunk_size) + 1 # 1-based indexing
                
                # Unique ID for this specific segment
                chunk_id = f"chunk_{document_id}_{uuid.uuid4().hex[:8]}"
                
                all_segments.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": chunk_text,
                    "block_type": "excel_deterministic_chunk",
                    "section_title": sheet_name or f"Sheet {sheet_index}",
                    "section_level": 1,
                    "metadata": {
                        "filename": file_path.name,
                        "sheet_index": sheet_index,
                        "sheet_name": sheet_name,
                        "sequential_index": seq_index,
                        "is_last_chunk": (i + chunk_size >= total_chars),
                        "total_chars_in_sheet": total_chars
                    },
                    "is_prechunked": True, # Flag to signal bypass of semantic chunker
                    "embedding_eligible": True # CRITICAL: Ensure it is indexed
                })
                
            print(f"SUCCESS: Sliced sheet '{sheet_name}' into {len(all_segments)} chunks.")

        except Exception as e:
            print(f"ERROR: Failed to process sheet '{sheet_name}': {e}")
            continue
            
    return all_segments
