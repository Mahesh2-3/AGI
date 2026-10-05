import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional
import edge_tts

from config.settings import settings
from tools.base import registry

DEFAULT_VOICE = "en-GB-RyanNeural"  # Crisp British gentleman voice


class JarvisSpeaker:
    """Jarvis Text-to-Speech Engine powered by Edge-TTS with streaming and barge-in support."""

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def stop(self):
        """Immediately interrupts any ongoing speech (Barge-In)."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                    self._current_process.wait(timeout=0.5)
                except Exception:
                    self._current_process.kill()
            self._current_process = None

    async def _generate_audio(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, voice=self.voice)
        await communicate.save(output_path)

    def speak(self, text: str, wait: bool = True) -> str:
        """Synthesizes text and plays audio with interruption support."""
        clean_text = text.strip()
        if not clean_text:
            return ""

        # Interrupt previous speech
        self.stop()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate audio file
            asyncio.run(self._generate_audio(clean_text, tmp_path))

            # Select audio player
            if shutil.which("pw-play"):
                player_cmd = ["pw-play", tmp_path]
            elif shutil.which("ffplay"):
                player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            elif shutil.which("paplay"):
                player_cmd = ["paplay", tmp_path]
            else:
                raise RuntimeError("No supported audio player found (pw-play, ffplay, paplay).")

            with self._lock:
                self._current_process = subprocess.Popen(player_cmd)

            if wait:
                self._current_process.wait()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            else:
                # Cleanup in background thread
                def _wait_and_clean():
                    if self._current_process:
                        self._current_process.wait()
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                threading.Thread(target=_wait_and_clean, daemon=True).start()

            return f"Spoke: '{clean_text[:60]}...'"
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Speech synthesis error: {e}")


speaker = JarvisSpeaker()


@registry.register(description="Speaks a message out loud to the user in Jarvis's authentic British voice.")
def speak_message(message: str) -> str:
    return speaker.speak(message, wait=True)
