import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from config.settings import settings
from tools.base import registry


class JarvisListener:
    """Jarvis Speech-to-Text Engine powered by Groq's high-speed Whisper Large v3."""

    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.openai_api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.whisper_model = "whisper-large-v3-turbo"

    def record_microphone_audio(self, duration_seconds: float = 5.0, output_path: Optional[str] = None) -> str:
        """Records microphone input for the specified duration using PipeWire or ffmpeg."""
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
        """Transcribes an audio file into text using Groq Whisper."""
        path = Path(audio_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

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

    def listen(self, duration_seconds: float = 5.0) -> str:
        """Records microphone and transcribes user speech in a single call."""
        audio_path = self.record_microphone_audio(duration_seconds=duration_seconds)
        try:
            text = self.transcribe(audio_path)
            return text
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


listener = JarvisListener()


@registry.register(description="Listens to the user's microphone for a few seconds and transcribes speech into text.")
def listen_to_user(duration_seconds: float = 5.0) -> str:
    text = listener.listen(duration_seconds=duration_seconds)
    return text if text else "No audible speech was detected."
