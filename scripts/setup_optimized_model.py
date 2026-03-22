import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def setup_model():
    # We use the Xenova version as it provides pre-converted ONNX and quantized weights
    onnx_model_id = "Xenova/bge-small-en-v1.5"
    
    # Path: backend/models/bge-onnx
    target_dir = Path("backend/models/bge-onnx")
    
    # Clean old base model to free up space
    if target_dir.exists():
        print(f"[*] Cleaning old model files from {target_dir}...")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Downloading optimized BGE-Small (INT8) from {onnx_model_id} to {target_dir}...")
    
    # Download essential configuration and the quantized ONNX model
    try:
        snapshot_download(
            repo_id=onnx_model_id,
            local_dir=target_dir.as_posix(),
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "onnx/model_quantized.onnx"
            ],
            local_files_only=False
        )
        print("[+] Model acquisition complete.")
    except Exception as e:
        print(f"[-] Model download failed: {e}")
        # Re-raise to stop execution if download fails
        raise

if __name__ == "__main__":
    setup_model()
