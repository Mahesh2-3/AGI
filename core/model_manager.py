import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional
from config.settings import settings

logger = logging.getLogger("JARVIS.ModelManager")


class OllamaModelManager:
    """Manages Ollama model residency and VRAM lifecycle.
    
    Guarantees that heavy reasoning and vision models never compete for GPU VRAM simultaneously,
    preventing out-of-memory errors, thermal throttling, and GPU load spikes on laptop hardware.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

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
            logger.debug(f"Could not retrieve loaded Ollama models from {url}: {e}")
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
            return False
        except Exception as e:
            logger.debug(f"Failed to unload model '{model_name}': {e}")
            return False

    def unload_all_models(self) -> int:
        """Unloads all currently resident models from memory."""
        loaded = self.get_loaded_models()
        unloaded_count = 0
        for m in loaded:
            if self.unload_model(m):
                unloaded_count += 1
        return unloaded_count

    def prepare_for_model(self, target_model: str) -> None:
        """Ensures that only the target model will occupy GPU VRAM.
        
        If other models are loaded, they are unloaded before the target model is invoked.
        """
        if not getattr(settings, "EXCLUSIVE_MODEL_LOADING", True):
            return

        target_clean = target_model.split(":")[0] if ":" in target_model else target_model
        loaded_models = self.get_loaded_models()

        for m in loaded_models:
            m_clean = m.split(":")[0] if ":" in m else m
            # If a different model is occupying VRAM, unload it
            if m != target_model and m_clean != target_clean:
                logger.info(f"Unloading resident model '{m}' to free VRAM for '{target_model}'...")
                self.unload_model(m)

    def unload_on_idle(self) -> None:
        """Unloads resident models when Jarvis finishes responding to keep the laptop cool."""
        if getattr(settings, "OLLAMA_AUTO_UNLOAD_ON_IDLE", True):
            self.unload_all_models()


model_manager = OllamaModelManager()
