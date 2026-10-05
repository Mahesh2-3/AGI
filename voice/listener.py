import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from config.settings import settings
from tools.base import registry


class JarvisListener:
    """Jarvis Speech-to-Text Engine powered by Groq's high-speed Whisper Large v3
    with Voice Activity Detection (VAD) and dynamic trailing silence cutoff."""

    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.openai_api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.whisper_model = "whisper-large-v3-turbo"

    def record_microphone_vad(
        self,
        max_duration_seconds: float = 8.0,
        silence_cutoff_seconds: float = 0.5,
        max_initial_wait_seconds: float = 3.0,
        sample_rate: int = 16000,
        output_path: Optional[str] = None,
    ) -> str:
        """Dynamically records microphone audio, stopping immediately when speech finishes.
        Uses real-time RMS energy analysis on PCM frames.
        """
        if not output_path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

        # Choose recording backend
        if shutil.which("pw-record"):
            cmd = [
                "pw-record",
                "--channels", "1",
                "--rate", str(sample_rate),
                "--format", "s16",
                "--raw",
                "-",
            ]
        elif shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "pulse",
                "-i", "default",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-f", "s16le",
                "-",
            ]
        else:
            return self.record_microphone_fixed(duration_seconds=5.0, output_path=output_path)

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )

            # Frame calculation: 40ms frame
            frame_ms = 40
            samples_per_frame = int(sample_rate * (frame_ms / 1000))
            bytes_per_frame = samples_per_frame * 2  # 16-bit PCM = 2 bytes/sample

            audio_buffer = bytearray()
            started_speaking = False
            consecutive_silence_s = 0.0
            start_time = time.perf_counter()

            # Baseline noise calibration
            noise_samples = []
            speech_threshold = 2500  # Default initial threshold

            while True:
                elapsed = time.perf_counter() - start_time
                if elapsed >= max_duration_seconds:
                    break

                raw_chunk = proc.stdout.read(bytes_per_frame)
                if not raw_chunk or len(raw_chunk) < bytes_per_frame:
                    break

                audio_buffer.extend(raw_chunk)

                # Compute RMS energy
                samples = struct.unpack(f"<{samples_per_frame}h", raw_chunk)
                sq_sum = sum(s * s for s in samples)
                rms = int(math.isqrt(sq_sum // samples_per_frame))

                # Calibrate noise floor in the first 200ms (5 frames)
                if len(noise_samples) < 5:
                    noise_samples.append(rms)
                    if len(noise_samples) == 5:
                        avg_noise = sum(noise_samples) // 5
                        speech_threshold = max(2200, int(avg_noise * 1.8))
                    continue

                if not started_speaking:
                    if rms >= speech_threshold:
                        started_speaking = True
                    elif elapsed >= max_initial_wait_seconds:
                        # User did not speak within initial window
                        break
                else:
                    if rms < speech_threshold:
                        consecutive_silence_s += (frame_ms / 1000)
                        if consecutive_silence_s >= silence_cutoff_seconds:
                            # Speech ended; cutoff recording
                            break
                    else:
                        consecutive_silence_s = 0.0

            # Gracefully terminate recorder process
            try:
                proc.terminate()
                proc.wait(timeout=0.2)
            except Exception:
                proc.kill()

            # Write accumulated raw PCM frames as standard WAV
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(bytes(audio_buffer))

            return output_path

        except Exception:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            # Fallback to fixed recording if dynamic capture fails
            return self.record_microphone_fixed(duration_seconds=5.0, output_path=output_path)

    def record_microphone_fixed(self, duration_seconds: float = 5.0, output_path: Optional[str] = None) -> str:
        """Records microphone input for the specified fixed duration as a fallback."""
        if not output_path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

        if shutil.which("pw-record"):
            cmd = [
                "timeout",
                str(duration_seconds),
                "pw-record",
                "--channels", "1",
                "--rate", "16000",
                output_path,
            ]
        elif shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "pulse",
                "-i", "default",
                "-t", str(duration_seconds),
                "-ac", "1",
                "-ar", "16000",
                output_path,
            ]
        else:
            raise RuntimeError("No audio recorder found (pw-record or ffmpeg).")

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribes an audio file into text using Groq Whisper Large v3."""
        path = Path(audio_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Check file size: if under 1KB, empty recording
        if path.stat().st_size < 1000:
            return ""

        if self.groq_api_key:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(path.name, f.read()),
                    model=self.whisper_model,
                    language="en",
                    temperature=0.0,
                )
            return transcription.text.strip()

        if self.openai_api_key:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-1",
                )
            return transcription.text.strip()

        raise RuntimeError("No Whisper STT API key configured. Set GROQ_API_KEY in .env.")

    def listen(self, duration_seconds: float = 8.0, use_vad: bool = True) -> str:
        """Records microphone and transcribes user speech in a single call."""
        if use_vad:
            audio_path = self.record_microphone_vad(max_duration_seconds=duration_seconds)
        else:
            audio_path = self.record_microphone_fixed(duration_seconds=duration_seconds)

        try:
            text = self.transcribe(audio_path)
            return text
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


listener = JarvisListener()


@registry.register(description="Listens to the user's microphone using voice activity detection and transcribes speech into text.")
def listen_to_user(duration_seconds: float = 8.0) -> str:
    text = listener.listen(duration_seconds=duration_seconds, use_vad=True)
    return text if text else "No audible speech was detected."
