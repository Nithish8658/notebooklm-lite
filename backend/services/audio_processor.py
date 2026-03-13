import subprocess
import os
import shutil
from pathlib import Path


class AudioProcessor:
    """
    Phase-Aware Adaptive Mastering Engine.
    Responds to episode energy and tension automatically.
    """

    def __init__(self, music_dir: str = "backend/uploads/music"):
        self.music_dir = Path(music_dir)
        self.music_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Public Entry
    # ============================================

    def master_podcast(self, voice_path: str, output_path: str, music_track: str = "ambient_lofi.mp3") -> bool:

        if not shutil.which("ffmpeg"):
            return False

        if not os.path.exists(voice_path):
            return False

        energy_profile = self._analyze_energy(voice_path)

        music_path = self.music_dir / music_track

        if not music_path.exists():
            return self._normalize_only(voice_path, output_path, energy_profile)

        return self._master_with_music(voice_path, str(music_path), output_path, energy_profile)

    # ============================================
    # Energy Analysis
    # ============================================

    def _analyze_energy(self, voice_path: str) -> dict:

        cmd = [
            "ffmpeg", "-i", voice_path,
            "-af", "volumedetect",
            "-f", "null", "-"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        import re
        mean_match = re.search(r"mean_volume:\s*(-?\d+\.?\d*)", result.stderr)

        mean_volume = float(mean_match.group(1)) if mean_match else -25.0

        if mean_volume > -18:
            level = "HIGH"
        elif mean_volume > -23:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"mean_volume": mean_volume, "level": level}

    # ============================================
    # Master With Music
    # ============================================

    def _master_with_music(self, voice_path: str, music_path: str, output_path: str, energy: dict) -> bool:

        temp_mix = output_path.replace(".mp3", "_premaster.wav")

        if energy["level"] == "HIGH":
            voice_comp = "compand=0.03|0.03:0.2|0.2:-70/-60/-20/-15:5:0:-16:0.2,"
            stereo_width = "stereotools=mlev=1:slev=1.3,"
            music_vol = "volume=0.3,"
        elif energy["level"] == "MEDIUM":
            voice_comp = "compand=0.03|0.03:0.2|0.2:-70/-60/-20/-15:4:0:-18:0.2,"
            stereo_width = "stereotools=mlev=1:slev=1.2,"
            music_vol = "volume=0.35,"
        else:
            voice_comp = "compand=0.04|0.04:0.3|0.3:-70/-60/-20/-15:3:0:-20:0.3,"
            stereo_width = "stereotools=mlev=1:slev=1.1,"
            music_vol = "volume=0.4,"

        filter_complex = (
            "[0:a]"
            "highpass=f=80,"
            f"{voice_comp}"
            "equalizer=f=3000:t=q:w=1:g=2,"
            "aecho=0.6:0.3:15:0.04,"
            "[voice];"

            "[1:a]"
            "highpass=f=80,"
            "equalizer=f=3000:t=q:w=1:g=-6,"
            f"{stereo_width}"
            f"{music_vol}"
            "aloop=loop=-1:size=0"
            "[bg];"

            "[bg][voice]"
            "sidechaincompress=threshold=0.02:ratio=6:attack=15:release=700"
            "[bgduck];"

            "[voice][bgduck]"
            "amix=inputs=2:duration=shortest,"
            "softclip"
            "[mix]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "[mix]",
            "-ar", "44100",
            "-c:a", "pcm_s16le",
            temp_mix
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return self._normalize_only(voice_path, output_path, energy)

        return self._two_pass_loudnorm(temp_mix, output_path)

    # ============================================
    # Two Pass Loudnorm
    # ============================================

    def _two_pass_loudnorm(self, input_wav: str, output_mp3: str) -> bool:

        try:
            cmd_analysis = [
                "ffmpeg", "-y",
                "-i", input_wav,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json",
                "-f", "null", "-"
            ]

            result = subprocess.run(cmd_analysis, check=True, capture_output=True, text=True)

            import json
            analysis = json.loads(self._extract_loudnorm_json(result.stderr))

            loudnorm = (
                f"loudnorm=I=-16:TP=-1.5:LRA=7:"
                f"measured_I={analysis['input_i']}:"
                f"measured_LRA={analysis['input_lra']}:"
                f"measured_TP={analysis['input_tp']}:"
                f"measured_thresh={analysis['input_thresh']}:"
                f"offset={analysis['target_offset']}:"
                "linear=true"
            )

            cmd_apply = [
                "ffmpeg", "-y",
                "-i", input_wav,
                "-af", loudnorm,
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                output_mp3
            ]

            subprocess.run(cmd_apply, check=True, capture_output=True)

            os.remove(input_wav)
            return True

        except Exception:
            return False

    # ============================================
    # Fallback
    # ============================================

    def _normalize_only(self, voice_path: str, output_path: str, energy: dict) -> bool:
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", voice_path,
                "-af",
                "highpass=f=80,"
                "compand=0.04|0.04:0.3|0.3:-70/-60/-20/-15:3:0:-20:0.3,"
                "loudnorm=I=-16:TP=-1.5:LRA=7",
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return True

        except Exception:
            return False

    # ============================================
    # Extract JSON
    # ============================================

    def _extract_loudnorm_json(self, stderr_text: str) -> str:
        import re
        match = re.search(r"\{.*\}", stderr_text, re.DOTALL)
        if not match:
            raise ValueError("Loudnorm JSON not found")
        return match.group(0)


audio_master = AudioProcessor()
