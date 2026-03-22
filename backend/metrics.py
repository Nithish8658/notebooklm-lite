import atexit
import json
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List
from uuid import uuid4

METRICS_FILE = Path("metrics.json")
_MAX_METRICS = 1000
_FLUSH_INTERVAL_SECONDS = 2.0


class MetricsRuntime:
    def __init__(self, path: Path, *, max_items: int = _MAX_METRICS, flush_interval_s: float = _FLUSH_INTERVAL_SECONDS):
        self._path = path
        self._max_items = max_items
        self._flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._dirty = False
        self._metrics: Deque[dict] = deque(
            self._read_from_disk(),
            maxlen=self._max_items,
        )
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="metrics-writer",
            daemon=True,
        )
        self._writer.start()

    def load_metrics(self) -> List[dict]:
        disk_metrics = self._read_from_disk()
        with self._lock:
            memory_metrics = list(self._metrics)
        return self._merge_metrics(disk_metrics, memory_metrics)

    def log_metric(self, data: dict):
        metric = dict(data or {})
        if "timestamp" not in metric:
            metric["timestamp"] = datetime.now().isoformat()
        metric.setdefault("_metric_id", uuid4().hex)

        with self._lock:
            self._metrics.append(metric)
            self._dirty = True

        self._wake_event.set()

    def shutdown(self):
        self._stop_event.set()
        self._wake_event.set()
        if self._writer.is_alive():
            self._writer.join(timeout=max(1.0, self._flush_interval_s + 1.0))
        self._flush_to_disk(force=True)

    def _writer_loop(self):
        while not self._stop_event.is_set():
            self._wake_event.wait(self._flush_interval_s)
            self._wake_event.clear()
            self._flush_to_disk()

        self._flush_to_disk(force=True)

    def _flush_to_disk(self, *, force: bool = False):
        with self._lock:
            if not self._dirty and not force:
                return
            memory_snapshot = list(self._metrics)
            self._dirty = False

        merged_metrics = self._merge_metrics(self._read_from_disk(), memory_snapshot)
        self._write_to_disk(merged_metrics)

        with self._lock:
            current_snapshot = list(self._metrics)
            if current_snapshot != memory_snapshot:
                self._metrics = deque(
                    self._merge_metrics(merged_metrics, current_snapshot),
                    maxlen=self._max_items,
                )
                self._dirty = True
            else:
                self._metrics = deque(merged_metrics, maxlen=self._max_items)

    def _read_from_disk(self) -> List[dict]:
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload[-self._max_items:]
        except (OSError, json.JSONDecodeError):
            return []
        return []

    def _write_to_disk(self, metrics: List[dict]):
        tmp_path = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            os.replace(tmp_path, self._path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _merge_metrics(self, disk_metrics: List[dict], memory_metrics: List[dict]) -> List[dict]:
        merged: List[dict] = []
        seen_ids = set()

        for metric in [*disk_metrics, *memory_metrics]:
            metric_id = metric.get("_metric_id")
            if metric_id is None:
                metric_id = f"{metric.get('timestamp', '')}:{metric.get('category', '')}:{metric.get('operation', '')}"
                metric["_metric_id"] = metric_id
            if metric_id in seen_ids:
                continue
            seen_ids.add(metric_id)
            merged.append(metric)

        return merged[-self._max_items:]


_METRICS_RUNTIME = MetricsRuntime(METRICS_FILE)


def load_metrics():
    return _METRICS_RUNTIME.load_metrics()


def log_metric(data: dict):
    """
    Logs a performance metric.
    Expected keys in data:
    - category: "ingestion" | "retrieval" | "llm" | "runtime"
    - operation: e.g., "transcription", "embedding", "search", "reranking"
    - model_name: e.g., "WhisperX", "bge-base", "BM25"
    - duration_ms: float (time taken in milliseconds)
    - timestamp: ISO string (optional, auto-added)
    - metadata: dict (optional, extra info like file_size, query_length)
    """
    _METRICS_RUNTIME.log_metric(data)


def get_metrics():
    return _METRICS_RUNTIME.load_metrics()


def shutdown_metrics():
    _METRICS_RUNTIME.shutdown()


atexit.register(shutdown_metrics)
