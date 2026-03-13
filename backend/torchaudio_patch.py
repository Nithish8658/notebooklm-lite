import os
import torchaudio
import torch
from dataclasses import dataclass
from typing import BinaryIO, Optional, Tuple, Union

@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str

def list_audio_backends():
    """Mock list_audio_backends for compatibility."""
    # Since this version uses torchcodec/ffmpeg under the hood
    return ["ffmpeg", "torchcodec"]

def info(uri: Union[BinaryIO, str, os.PathLike], backend: Optional[str] = None) -> AudioMetaData:
    """Mock info function using torchcodec."""
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(uri)
        metadata = decoder.metadata
        
        # Determine num_frames if possible
        num_frames = 0
        if metadata.duration is not None and metadata.sample_rate is not None:
            num_frames = int(metadata.duration * metadata.sample_rate)
            
        return AudioMetaData(
            sample_rate=metadata.sample_rate or 16000,
            num_frames=num_frames,
            num_channels=metadata.num_channels or 1,
            bits_per_sample=16, # Assume 16-bit for PCM
            encoding="PCM_S" # Mock encoding
        )
    except Exception as e:
        # Fallback if torchcodec fails
        print(f"Warning: Mock info failed: {e}")
        return AudioMetaData(16000, 0, 1, 16, "PCM_S")

def patch_torchaudio():
    """Applies the patches to the torchaudio module."""
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    
    if not hasattr(torchaudio, "info"):
        torchaudio.info = info
        
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = list_audio_backends
    
    # Add to __all__ if missing
    if hasattr(torchaudio, "__all__"):
        if "AudioMetaData" not in torchaudio.__all__:
            torchaudio.__all__.append("AudioMetaData")
        if "info" not in torchaudio.__all__:
            torchaudio.__all__.append("info")
        if "list_audio_backends" not in torchaudio.__all__:
            torchaudio.__all__.append("list_audio_backends")

# Apply immediately upon import
patch_torchaudio()
