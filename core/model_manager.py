import json
import logging
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
from config.settings import settings

logger = logging.getLogger("JARVIS.ModelManager")


class OllamaModelManager:
    """Manages Ollama model residency and VRAM lifecycle with deliberate,
    exclusive loading and configurable idle timeouts for laptop hardware.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.last_used_timestamp: float = time.time()
        self.last_switch_metrics: Dict[str, Any] = {}

    @staticmethod
    def normalize_model_name(name: str) -> str:
        """Normalizes model names for exact tag comparison (e.g. 'qwen3:8b' vs 'qwen3:4b').
        Guarantees that distinct quantizations/tags sharing the same base family
        are recognized as distinct models.
        """
        if not name:
            return ""
        clean = name.strip().lower()
        if ":" not in clean:
            return f"{clean}:latest"
        return clean

    def get_loaded_models(self) -> List[str]:
        """Queries Ollama for all currently loaded models resident in memory/VRAM."""
        url = f"{self.base_url}/api/ps"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Jarvis-ModelManager"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                loaded_names = []
                for m in models:
                    name = m.get("name") or m.get("model")
                    if name:
                        loaded_names.append(name)
                return loaded_names
        except Exception as e:
            logger.warning(f"Could not retrieve loaded Ollama models from {url}: {e}")
            return []

    def unload_model(self, model_name: str) -> bool:
        """Immediately unloads a model from GPU VRAM by setting keep_alive to 0."""
        if not model_name:
            return False
        url = f"{self.base_url}/api/generate"
        payload = {"model": model_name, "keep_alive": 0}
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "Jarvis-ModelManager"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info(f"Successfully unloaded '{model_name}' from VRAM.")
                    return True
                else:
                    logger.warning(f"Unload request for '{model_name}' returned status {resp.status}.")
                    return False
        except Exception as e:
            logger.warning(f"Failed to unload model '{model_name}': {e}")
            return False

    def unload_all_models(self) -> int:
        """Unloads all currently resident models from memory."""
        loaded = self.get_loaded_models()
        unloaded_count = 0
        for m in loaded:
            if self.unload_model(m):
                unloaded_count += 1
        return unloaded_count

    def prepare_for_model(self, target_model: str) -> Dict[str, Any]:
        """Ensures that only the target model will occupy GPU VRAM.

        Accurately differentiates model tags (e.g. qwen3:4b vs qwen3:8b) and tracks
        model-loading / switching durations.
        """
        start_t = time.perf_counter()
        target_norm = self.normalize_model_name(target_model)
        self.last_used_timestamp = time.time()

        if not getattr(settings, "EXCLUSIVE_MODEL_LOADING", True):
            elapsed = time.perf_counter() - start_t
            metrics = {
                "target_model": target_model,
                "was_resident": True,
                "unloaded_models": [],
                "duration_s": round(elapsed, 4),
            }
            self.last_switch_metrics = metrics
            return metrics

        loaded_models = self.get_loaded_models()
        was_resident = any(self.normalize_model_name(m) == target_norm for m in loaded_models)
        unloaded: List[str] = []

        for m in loaded_models:
            m_norm = self.normalize_model_name(m)
            # If a different model is occupying VRAM, unload it
            if m_norm != target_norm:
                logger.info(f"Exclusive VRAM: Unloading resident model '{m}' to free VRAM for '{target_model}'...")
                if self.unload_model(m):
                    unloaded.append(m)

        elapsed = time.perf_counter() - start_t
        metrics = {
            "target_model": target_model,
            "was_resident": was_resident,
            "unloaded_models": unloaded,
            "duration_s": round(elapsed, 4),
        }
        self.last_switch_metrics = metrics
        return metrics

    def mark_active(self) -> None:
        """Updates the last active timestamp to keep models resident during active conversations."""
        self.last_used_timestamp = time.time()

    def unload_on_idle(self, force: bool = False) -> int:
        """Configurable idle unloading.

        - If force is True: immediately purges all resident models from VRAM.
        - If OLLAMA_AUTO_UNLOAD_ON_IDLE is True: unloads immediately (legacy behaviour).
        - Otherwise: keeps models warm unless idle timeout (e.g. 300s) has elapsed.
        """
        if force:
            logger.info("Force unloading all resident models from VRAM.")
            return self.unload_all_models()

        if getattr(settings, "OLLAMA_AUTO_UNLOAD_ON_IDLE", False):
            logger.info("Immediate idle unload triggered by OLLAMA_AUTO_UNLOAD_ON_IDLE.")
            return self.unload_all_models()

        timeout = getattr(settings, "OLLAMA_IDLE_TIMEOUT_SECONDS", 300)
        idle_duration = time.time() - self.last_used_timestamp
        if idle_duration >= timeout:
            logger.info(f"Idle timeout exceeded ({idle_duration:.1f}s >= {timeout}s). Unloading models to cool GPU.")
            return self.unload_all_models()

        return 0

    def manual_unload(self, model_name: Optional[str] = None) -> bool:
        """Explicitly unloads a specific model or all models on demand."""
        if model_name:
            return self.unload_model(model_name)
        return self.unload_all_models() > 0


model_manager = OllamaModelManager()
