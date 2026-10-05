import re
import time
from typing import Callable, Optional, Tuple
from voice.listener import listener
from voice.speaker import speaker


class WakeWordDetector:
    """Detects 'Jarvis' wake-word activation and extracts immediate commands."""

    def __init__(self, wake_words: Optional[list] = None):
        self.wake_words = wake_words or ["jarvis", "hey jarvis", "ok jarvis"]
        self._running = False

    def check_audio_for_wake_word(self, transcribed_text: str) -> Tuple[bool, str]:
        """Checks if transcribed text contains the wake word and isolates the subsequent instruction."""
        clean = transcribed_text.lower().strip()
        # Remove punctuation for matching
        clean_nopunct = re.sub(r"[^\w\s]", "", clean)

        for w in self.wake_words:
            pattern = rf"\b{w}\b"
            match = re.search(pattern, clean_nopunct)
            if match:
                # Find start index of command after wake word
                idx = match.end()
                remainder = clean_nopunct[idx:].strip()
                return True, remainder

        return False, ""

    def listen_for_activation(self, duration_seconds: float = 3.0) -> Tuple[bool, str]:
        """Records a short audio window and evaluates if Jarvis was summoned."""
        try:
            text = listener.listen(duration_seconds=duration_seconds)
            if not text:
                return False, ""
            return self.check_audio_for_wake_word(text)
        except Exception:
            return False, ""

    def run_continuous_listener(
        self,
        on_wake: Callable[[str], None],
        poll_interval: float = 0.5,
    ):
        """Continuously listens for wake-word activation in a background loop."""
        self._running = True
        while self._running:
            triggered, command = self.listen_for_activation(duration_seconds=3.0)
            if triggered:
                on_wake(command)
            time.sleep(poll_interval)

    def stop(self):
        self._running = False


wake_detector = WakeWordDetector()
