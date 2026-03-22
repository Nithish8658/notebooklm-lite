import os
import asyncio
import time
import json
import multiprocessing
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForFeatureExtraction
import logging

_LOG = logging.getLogger("embeddings")

# ---------- CONFIG ----------
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this question for searching relevant passages: "
_embedding_model = None

# ---------- ONNX MODEL WRAPPER ----------
class OnnxEmbeddingModel:
    def __init__(self, model_id: str, *, num_threads: int | None = None):
        # Path: backend/services/embeddings.py -> backend/models/bge-onnx
        self.project_root = Path(__file__).resolve().parent.parent 
        onnx_path = self.project_root / "models" / "bge-onnx"
        
        # 1. AC-52: Dynamic Config Detection (Architectural Improvement)
        config_path = onnx_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {onnx_path}")
            
        with open(config_path, "r") as f:
            self.config_data = json.load(f)
        
        # Expose dimensions for infrastructure synchronization
        self.hidden_size = self.config_data.get("hidden_size", 768)
        _LOG.info("Detected model dimensions: %d", self.hidden_size)

        # 2. Optimized File Selection (INT8 First)
        model_filename = "model_quantized.onnx"
        subfolder = None
        
        if (onnx_path / "onnx" / "model_quantized.onnx").exists():
            subfolder = "onnx"
        elif (onnx_path / "model_quantized.onnx").exists():
            subfolder = None
        else:
            # Fallback to standard model.onnx
            model_filename = "model.onnx"
            if (onnx_path / "onnx" / "model.onnx").exists():
                subfolder = "onnx"
            elif (onnx_path / "model.onnx").exists():
                subfolder = None
            else:
                raise FileNotFoundError(f"No valid ONNX model found in {onnx_path}. Run setup_optimized_model.py first.")

        _LOG.info("Loading INT8 optimized ONNX embedding model (File: %s, Subfolder: %s) from %s", 
                 model_filename, subfolder or "root", onnx_path)

        if num_threads is None:
            env_threads = os.getenv("EMBEDDING_THREADS")
            if env_threads:
                num_threads = int(env_threads)
            else:
                total_cores = multiprocessing.cpu_count()
                num_threads = min(4, max(1, total_cores // 2))
        else:
            total_cores = multiprocessing.cpu_count()
            num_threads = max(1, min(int(num_threads), total_cores))

        _LOG.info("CPU Optimization: Using %d threads for embedding inference.", num_threads)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 3. AC-52: Production-Grade Optimizations
        # Enable Memory Mapping (mmap) for Zero-Copy Sharing across workers
        sess_options.add_session_config_entry("session.use_mmap", "1")
        # Optimization for INT8 Quantized models: Prefer sequential execution
        sess_options.add_session_config_entry("session.inter_op_num_threads", "1")
        
        # Use .as_posix() to ensure Windows paths are handled correctly
        # Use use_fast=True for high-speed Rust-based tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(
            onnx_path.as_posix(), 
            local_files_only=True,
            use_fast=True
        )
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path.as_posix(), 
            file_name=model_filename,
            subfolder=subfolder,
            provider="CPUExecutionProvider",
            session_options=sess_options,
            local_files_only=True
        )
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=32, is_query: bool = False, **kwargs):
        is_single = isinstance(sentences, str)
        if is_single: 
            sentences = [sentences]
        
        if is_query:
            sentences = [f"{BGE_QUERY_PREFIX}{s}" for s in sentences]

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            # AC-52: Context capacity increased to 512 for Small-INT8
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.numpy())
        
        if not all_embeddings: 
            return np.array([])
            
        result = np.concatenate(all_embeddings, axis=0)
        return result[0] if is_single else result

def load_embedding_model(*, num_threads: int | None = None):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OnnxEmbeddingModel(EMBEDDING_MODEL_NAME, num_threads=num_threads)
    return _embedding_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        return load_embedding_model()
    return _embedding_model

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2))


# ---------- BATCH MANAGER ----------
class BatchEmbeddingManager:
    """
    Compatibility wrapper for the dedicated embedding runtime.
    """
    def __init__(self, model_getter, batch_size: int = 16, wait_time_ms: int = 10):
        self.model_getter = model_getter
        self.batch_size = batch_size
        self.wait_time_ms = wait_time_ms / 1000.0

    async def get_embedding(self, text: str, *, request_context=None, priority: str = "online") -> List[float]:
        from services.embedding_runtime import get_embedding_runtime

        return await get_embedding_runtime().get_embedding(
            text,
            request_context=request_context,
            priority=priority,
        )

    async def get_embeddings(self, texts: List[str], *, request_context=None, priority: str = "online") -> List[List[float]]:
        from services.embedding_runtime import get_embedding_runtime

        return await get_embedding_runtime().get_embeddings(
            texts,
            request_context=request_context,
            priority=priority,
        )
