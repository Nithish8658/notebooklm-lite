import json
import os
from pathlib import Path
from datetime import datetime

METRICS_FILE = Path("metrics.json")

def load_metrics():
    if not METRICS_FILE.exists():
        return []
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def log_metric(data: dict):
    """
    Logs a performance metric.
    Expected keys in data:
    - category: "ingestion" | "retrieval"
    - operation: e.g., "transcription", "embedding", "search", "reranking"
    - model_name: e.g., "WhisperX", "bge-base", "BM25"
    - duration_ms: float (time taken in milliseconds)
    - timestamp: ISO string (optional, auto-added)
    - metadata: dict (optional, extra info like file_size, query_length)
    """
    metrics = load_metrics()
    
    # Add timestamp if not present
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
        
    metrics.append(data)
    
    # Limit to last 1000 metrics to avoid infinite growth
    if len(metrics) > 1000:
        metrics = metrics[-1000:]
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def get_metrics():
    return load_metrics()
