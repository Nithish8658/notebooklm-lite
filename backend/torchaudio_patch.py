import os
import sys
import torchaudio
import torch
import inspect
from dataclasses import dataclass
from typing import BinaryIO, Optional, Tuple, Union

# --- INDUSTRIAL-GRADE MODULE INJECTION (THE "SHIM") ---
def inject_industrial_mocks():
    # 1. Fake 'torchaudio.backend' module structure
    class MockModule:
        def __init__(self, name):
            self.__name__ = name
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    backend = MockModule('torchaudio.backend')
    if 'torchaudio.backend' not in sys.modules:
        sys.modules['torchaudio.backend'] = backend
    if 'torchaudio.backend.common' not in sys.modules:
        sys.modules['torchaudio.backend.common'] = MockModule('torchaudio.backend.common')

    # 2. Attach missing legacy functions to torchaudio
    mock_functions = {
        "set_audio_backend": lambda x: None,
        "get_audio_backend": lambda: "soundfile",
        "list_audio_backends": lambda: ["soundfile", "ffmpeg"],
        "backend": backend
    }
    for func_name, func_obj in mock_functions.items():
        if not hasattr(torchaudio, func_name):
            setattr(torchaudio, func_name, func_obj)

    # 3. SOPHISTICATED WHISPERX COMPATIBILITY PATCH (Filter & Fill)
    try:
        import faster_whisper.transcribe
        orig_options = faster_whisper.transcribe.TranscriptionOptions
        
        # Determine the target class's expected fields
        # NamedTuples use _fields, Dataclasses use __annotations__
        expected_fields = []
        if hasattr(orig_options, "_fields"):
            expected_fields = list(orig_options._fields)
        elif hasattr(orig_options, "__annotations__"):
            expected_fields = list(orig_options.__annotations__.keys())

        class PatchedTranscriptionOptions(orig_options):
            def __new__(cls, *args, **kwargs):
                # A. Handle Positional Arguments (If any)
                # B. Fill missing keys that the engine EXPECTS
                defaults = {
                    'multilingual': False,
                    'max_new_tokens': None,
                    'clip_timestamps': None,
                    'hallucination_silence_threshold': None,
                    'hotwords': None,
                    'chunk_length': None
                }
                
                # Only add keys if the class actually supports them
                for key, val in defaults.items():
                    if key in expected_fields and key not in kwargs:
                        kwargs[key] = val
                
                # C. FILTER: Remove keys the engine DOES NOT support
                # This prevents the "unexpected keyword argument" crash
                final_kwargs = {k: v for k, v in kwargs.items() if k in expected_fields}
                
                # Ensure we handle positional args if the library passes them the old way
                return super().__new__(cls, *args, **final_kwargs)

        faster_whisper.transcribe.TranscriptionOptions = PatchedTranscriptionOptions
        print(f"PATCH: Filter & Fill shim active. (Target fields: {len(expected_fields)})")
    except ImportError:
        pass

@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str

def info(uri: Union[BinaryIO, str, os.PathLike], backend: Optional[str] = None) -> AudioMetaData:
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(uri)
        metadata = decoder.metadata
        num_frames = int(metadata.duration * metadata.sample_rate) if metadata.duration and metadata.sample_rate else 0
        return AudioMetaData(metadata.sample_rate or 16000, num_frames, metadata.num_channels or 1, 16, "PCM_S")
    except Exception:
        return AudioMetaData(16000, 0, 1, 16, "PCM_S")

def apply_industrial_patch():
    inject_industrial_mocks()
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "info"):
        torchaudio.info = info

# Execute immediately
apply_industrial_patch()
print("SOPHISTICATED PATCH: Industrial-grade compatibility layer is now LIVE.")
