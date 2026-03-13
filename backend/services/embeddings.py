import os
import asyncio
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import logging

_LOG = logging.getLogger("embeddings")

# ---------- CONFIG ----------
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this question for searching relevant passages: "
_embedding_model = None

# ---------- ONNX MODEL WRAPPER ----------
class OnnxEmbeddingModel:
    def __init__(self, model_id: str):
        # Path: backend/services/embeddings.py -> backend/models/bge-onnx
        self.project_root = Path(__file__).resolve().parent.parent 
        onnx_path = self.project_root / "models" / "bge-onnx"
        
        # AC-52: Smart path resolution for Xenova/Optimum structure
        model_filename = "model_quantized.onnx"
        subfolder = None
        
        # Check for quantized version first
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
                # CRITICAL: If no ONNX file is found, we must not let it fall back to HF Hub
                raise FileNotFoundError(f"No ONNX model found in {onnx_path}. Ensure the main process has finished downloading models.")

        _LOG.info("Loading optimized ONNX embedding model (File: %s, Subfolder: %s) from %s", 
                 model_filename, subfolder or "root", onnx_path)
        
        # Standardize loading: always load from verified local directory
        # Use .as_posix() to ensure Windows paths are handled correctly by HF libraries
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_path.as_posix(), local_files_only=True)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path.as_posix(), 
            file_name=model_filename,
            subfolder=subfolder,
            provider="CPUExecutionProvider", 
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
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=384, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.numpy())
        
        if not all_embeddings: 
            return np.array([])
            
        result = np.concatenate(all_embeddings, axis=0)
        return result[0] if is_single else result

def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OnnxEmbeddingModel(EMBEDDING_MODEL_NAME)
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
    AC-22: Smart Batching for Embeddings (Thread/Loop Safe version).
    Uses Loop-Local Storage to prevent 'different event loop' crashes 
    when running concurrent Celery tasks.
    """
    def __init__(self, model_getter, batch_size: int = 16, wait_time_ms: int = 10):
        self.model_getter = model_getter
        self.batch_size = batch_size
        self.wait_time_ms = wait_time_ms / 1000.0
        
        # INDUSTRIAL FIX: State is stored per-loop to avoid cross-contamination
        # Dictionary of {loop_id: {"queue": Queue, "task": Task}}
        self._loop_states = {}
        self._lock = asyncio.Lock()

    def _get_loop_state(self):
        """Returns the queue and background task for the current loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        
        if loop_id not in self._loop_states:
            _LOG.info("Initializing unique Embedding Batcher for Loop: %s", loop_id)
            queue = asyncio.Queue()
            task = loop.create_task(self._batch_loop(queue))
            self._loop_states[loop_id] = {"queue": queue, "task": task}
            
        # Cleanup: Remove dead loops from registry to prevent memory leak
        # (Simplified: in production we would use a WeakKeyDictionary)
        
        return self._loop_states[loop_id]

    async def get_embedding(self, text: str) -> List[float]:
        """
        Public API to request an embedding. 
        Isolated per event loop for maximum stability.
        """
        state = self._get_loop_state()
        
        # Ensure the background task is still alive
        if state["task"].done():
            _LOG.info("Restarting dead background task for loop.")
            state["task"] = asyncio.get_running_loop().create_task(self._batch_loop(state["queue"]))

        future = asyncio.get_running_loop().create_future()
        await state["queue"].put((text, future))
        return await future

    async def _batch_loop(self, queue: asyncio.Queue):
        _LOG.info("Embedding Batch Loop Started.")
        while True:
            try:
                text, future = await queue.get()
                batch = [(text, future)]

                start_wait = time.time()
                while len(batch) < self.batch_size and (time.time() - start_wait) < self.wait_time_ms:
                    try:
                        text, future = queue.get_nowait()
                        batch.append((text, future))
                    except asyncio.QueueEmpty:
                        break
                
                try:
                    texts = [item[0] for item in batch]
                    futures = [item[1] for item in batch]
                    
                    t_start = time.perf_counter()
                    model = self.model_getter()
                    
                    # Offload model inference to a thread to keep the loop responsive
                    embeddings = await asyncio.to_thread(
                        model.encode,
                        texts,
                        is_query=True
                    )
                    
                    if len(texts) == 1 and embeddings.ndim == 1:
                        embeddings = np.expand_dims(embeddings, axis=0)

                    duration = time.perf_counter() - t_start
                    if len(batch) > 1:
                        _LOG.info("BATCH ENCODE: Processed %d texts in %.4fs (%.4fs per text)", 
                                len(batch), duration, duration/len(batch))

                    for i, emb in enumerate(embeddings):
                        if not futures[i].done():
                            futures[i].set_result(emb.tolist())

                except Exception as e:
                    _LOG.error("BATCH ENCODE ERROR: %s", e)
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                finally:
                    for _ in range(len(batch)): 
                        queue.task_done()
            except Exception as loop_err:
                _LOG.critical("BATCH LOOP FATAL ERROR: %s", loop_err)
                await asyncio.sleep(1) # Prevent tight loop on crash
