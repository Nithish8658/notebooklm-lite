import os
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

_LOG = logging.getLogger("bootstrap")
# Mapping of local folder name to Hugging Face Repo ID
MODEL_MAP = {
    "bge-onnx": {
        "repo_id": "Xenova/bge-base-en-v1.5",
        # Only download the base ONNX and tokenizer configs
        "allow_patterns": ["*.json", "*.txt", "onnx/model.onnx"] 
    },
    "reranker-onnx": {
        "repo_id": "Xenova/ms-marco-MiniLM-L-4-v2",
        # Only download the quantized version (3x faster) and configs
        "allow_patterns": ["*.json", "*.txt", "onnx/model_quantized.onnx"]
    }
}

def verify_all_models():
    """
    Ensures all required ML models are present on disk.
    If missing, downloads them from Hugging Face Hub.
    """
    base_models_dir = Path(__file__).parent.parent / "models"
    base_models_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("Starting model verification...")

    for folder_name, config in MODEL_MAP.items():
        target_dir = base_models_dir / folder_name
        repo_id = config["repo_id"]

        # Check if folder is truly empty or missing crucial ONNX file
        is_missing = not target_dir.exists() or not any(target_dir.rglob("*.onnx"))

        if is_missing:
            _LOG.info(f"MODEL MISSING: '{folder_name}' not found. Downloading surgical assets from {repo_id}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    allow_patterns=config.get("allow_patterns"),
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.pt"], 
                    local_dir_use_symlinks=False
                )
                _LOG.info(f"SUCCESS: '{folder_name}' downloaded.")
            except Exception as e:
                _LOG.error(f"CRITICAL: Failed to download model {repo_id}: {e}")
        else:
            _LOG.info(f"MODEL READY: '{folder_name}' verified on disk.")

    # AC-53: Pre-download WhisperX Alignment Model (English)
    # This avoids the runtime download during the first ingestion.
    align_dir = base_models_dir / "alignment"
    expected_pth = align_dir / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    
    if not expected_pth.exists():
        _LOG.info("MODEL MISSING: WhisperX English Alignment not found. Downloading to %s...", align_dir)
        try:
            import whisperx
            # This triggers the specific torchaudio download that whisperx expects for 'en'
            # We load it on CPU just to trigger the download/verification.
            whisperx.load_align_model(language_code="en", device="cpu", model_dir=str(align_dir))
            _LOG.info("SUCCESS: WhisperX English Alignment verified.")
        except Exception as e:
            _LOG.error(f"ERROR: Failed to pre-download WhisperX Alignment: {e}")
    else:
        _LOG.info("MODEL READY: WhisperX English Alignment verified on disk.")


if __name__ == "__main__":
    # Allow running standalone for manual prep
    logging.basicConfig(level=logging.INFO)
    verify_all_models()
