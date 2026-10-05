import asyncio
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator, Iterable, List, Optional
import edge_tts

from config.settings import settings
from tools.base import registry

DEFAULT_VOICE = "en-GB-RyanNeural"  # Crisp British gentleman voice
RAM_AUDIO_DIR = Path("/dev/shm/jarvis_audio") if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK) else None


class JarvisSpeaker:
    """Jarvis Text-to-Speech Engine powered by Edge-TTS with sentence pipelining,
    streaming token consumption, and instant barge-in interruption."""

    def __init__(self, voice: str = DEFAULT_VOICE):
        self.voice = voice
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self.audio_dir = RAM_AUDIO_DIR or (settings.WORKSPACE_DIR / "audio_cache")
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def stop(self):
        """Immediately interrupts any ongoing speech synthesis or playback (Barge-In)."""
        self._stop_flag.set()
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                    self._current_process.wait(timeout=0.3)
                except Exception:
                    self._current_process.kill()
            self._current_process = None

    async def _generate_audio_async(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, voice=self.voice)
        await communicate.save(output_path)

    def _play_audio_file(self, file_path: str):
        """Plays a single audio file using the system's preferred audio player."""
        if shutil.which("pw-play"):
            player_cmd = ["pw-play", file_path]
        elif shutil.which("ffplay"):
            player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]
        elif shutil.which("paplay"):
            player_cmd = ["paplay", file_path]
        else:
            raise RuntimeError("No supported audio player found (pw-play, ffplay, paplay).")

        with self._lock:
            if self._stop_flag.is_set():
                return
            self._current_process = subprocess.Popen(player_cmd)

        self._current_process.wait()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits markdown/text into clean, pronounceable sentences."""
        # Strip markdown syntax (bold, italic, code blocks, bullet points)
        cleaned = re.sub(r"```[\s\S]*?```", "", text)
        cleaned = re.sub(r"`[^`]+`", "", cleaned)
        cleaned = re.sub(r"[\*\_#>\-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Split on sentence boundaries (. ! ?)
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [cleaned]

    def speak(self, text: str, wait: bool = True) -> str:
        """Synthesizes text and plays audio with sentence-pipelining for minimal Time-To-First-Audio."""
        clean_text = text.strip()
        if not clean_text:
            return ""

        # Interrupt previous speech
        self.stop()
        self._stop_flag.clear()

        sentences = self._split_into_sentences(clean_text)

        if len(sentences) <= 1:
            # Single short sentence: synthesize and play directly
            return self._speak_single_sentence(sentences[0], wait=wait)
        else:
            # Multiple sentences: pipeline synthesis & playback concurrently
            return self._speak_pipelined_sentences(sentences, wait=wait)

    def _speak_single_sentence(self, sentence: str, wait: bool = True) -> str:
        tmp_path = str(self.audio_dir / f"jarvis_voice_{time.perf_counter_ns()}.mp3")
        try:
            asyncio.run(self._generate_audio_async(sentence, tmp_path))
            if not self._stop_flag.is_set():
                if wait:
                    self._play_audio_file(tmp_path)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                else:
                    def _async_play():
                        try:
                            self._play_audio_file(tmp_path)
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    threading.Thread(target=_async_play, daemon=True).start()
            return f"Spoke: '{sentence[:60]}...'"
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Speech synthesis error: {e}")

    def _speak_pipelined_sentences(self, sentences: List[str], wait: bool = True) -> str:
        """Concurrently synthesizes sentence N+1 while sentence N is playing."""
        audio_queue: queue.Queue = queue.Queue(maxsize=3)
        created_files: List[str] = []

        def _synthesizer_worker():
            for s in sentences:
                if self._stop_flag.is_set():
                    break
                tmp_path = str(self.audio_dir / f"pipeline_{time.perf_counter_ns()}.mp3")
                created_files.append(tmp_path)
                try:
                    asyncio.run(self._generate_audio_async(s, tmp_path))
                    if not self._stop_flag.is_set():
                        audio_queue.put(tmp_path)
                except Exception:
                    pass
            audio_queue.put(None)  # Sentinel to mark completion

        def _playback_worker():
            try:
                while not self._stop_flag.is_set():
                    try:
                        audio_path = audio_queue.get(timeout=5.0)
                    except queue.Empty:
                        break
                    if audio_path is None:
                        break
                    try:
                        self._play_audio_file(audio_path)
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
            finally:
                # Cleanup any leftover files if interrupted
                for f in created_files:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass

        synth_thread = threading.Thread(target=_synthesizer_worker, daemon=True)
        play_thread = threading.Thread(target=_playback_worker, daemon=True)

        synth_thread.start()
        play_thread.start()

        if wait:
            synth_thread.join()
            play_thread.join()
            return f"Spoke {len(sentences)} sentences pipelined."
        else:
            return f"Pipelined speech started ({len(sentences)} sentences)."

    def speak_stream(self, token_stream: Iterable[str], wait: bool = True) -> str:
        """Streams incoming LLM tokens, buffering sentences and speaking each as soon as formed."""
        buffer = ""
        sentences_emitted = 0

        # Interrupt any current speech
        self.stop()
        self._stop_flag.clear()

        audio_queue: queue.Queue = queue.Queue(maxsize=4)
        created_files: List[str] = []

        def _playback_worker():
            try:
                while not self._stop_flag.is_set():
                    try:
                        audio_path = audio_queue.get(timeout=6.0)
                    except queue.Empty:
                        break
                    if audio_path is None:
                        break
                    try:
                        self._play_audio_file(audio_path)
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
            finally:
                for f in created_files:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass

        player_thread = threading.Thread(target=_playback_worker, daemon=True)
        player_thread.start()

        def _synthesize_sentence(sent: str):
            clean_s = sent.strip()
            if not clean_s or self._stop_flag.is_set():
                return
            tmp_path = str(self.audio_dir / f"stream_{time.perf_counter_ns()}.mp3")
            created_files.append(tmp_path)
            try:
                asyncio.run(self._generate_audio_async(clean_s, tmp_path))
                if not self._stop_flag.is_set():
                    audio_queue.put(tmp_path)
            except Exception:
                pass

        for token in token_stream:
            buffer += token
            # Check for sentence end
            if any(punct in token for punct in [".", "!", "?", "\n"]):
                parts = re.split(r"(?<=[.!?\n])\s+", buffer)
                if len(parts) > 1:
                    ready = parts[0]
                    buffer = "".join(parts[1:])
                    _synthesize_sentence(ready)
                    sentences_emitted += 1

        if buffer.strip():
            _synthesize_sentence(buffer.strip())
            sentences_emitted += 1

        audio_queue.put(None)

        if wait:
            player_thread.join()

        return f"Streamed {sentences_emitted} sentences."


speaker = JarvisSpeaker()


@registry.register(description="Speaks a message out loud to the user in Jarvis's authentic British voice.")
def speak_message(message: str) -> str:
    return speaker.speak(message, wait=True)
