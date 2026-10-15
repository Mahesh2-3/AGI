import logging
import time
from typing import Any, Callable, Dict, Generator, List, Optional
from config.settings import settings
from core.model_manager import model_manager

logger = logging.getLogger("JARVIS.LLM")


class LLMClient:
    """High-speed local inference client interfacing directly with local Ollama GPU models.

    Includes per-request latency tracking, Time-To-First-Token (TTFT) instrumentation,
    streaming integration, and coordinated VRAM residency.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        on_fallback: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.provider = (provider or getattr(settings, "DEFAULT_PROVIDER", "ollama")).lower()
        self.model = model or getattr(settings, "DEFAULT_MODEL", "qwen3:8b")
        self.ollama_base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.keep_alive = getattr(settings, "OLLAMA_KEEP_ALIVE", "5m")
        self.on_fallback = on_fallback
        self._client = None

        # Metrics for the last inference request
        self.last_metrics: Dict[str, Any] = {
            "model": self.model,
            "duration_s": 0.0,
            "ttft_s": None,
            "token_count": 0,
            "streamed": False,
        }

        self._init_client()

    def _init_client(self):
        """Initializes the OpenAI-compatible client for local Ollama."""
        from openai import OpenAI
        base_url = f"{self.ollama_base_url.rstrip('/')}/v1"
        self._client = OpenAI(base_url=base_url, api_key="ollama")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
    ) -> Any:
        """Sends chat request to local Ollama with full latency measurement and residency management."""
        if not self._client:
            raise RuntimeError(
                f"Local Ollama client not configured. Ensure Ollama is running at {self.ollama_base_url}."
            )

        # Coordinate VRAM residency before inference
        model_manager.prepare_for_model(self.model)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": {"keep_alive": self.keep_alive},
        }

        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
            duration = time.perf_counter() - start_time
            model_manager.mark_active()

            token_count = 0
            if hasattr(response, "usage") and response.usage:
                token_count = getattr(response.usage, "completion_tokens", 0)

            self.last_metrics = {
                "model": self.model,
                "duration_s": round(duration, 4),
                "ttft_s": None,
                "token_count": token_count,
                "streamed": False,
            }
            logger.debug(f"Inference completed on {self.model} in {duration:.3f}s ({token_count} tokens).")
            return response
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Inference failed on {self.model} after {duration:.3f}s: {e}")
            raise e

    def classify_intent(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Any:
        """Semantically classifies user intent using local Qwen3 without tool calls."""
        return self.chat(messages=messages, tools=None, temperature=temperature)

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
    ) -> Generator[Any, None, None]:
        """Streams chat completion tokens from local Ollama with TTFT measurement."""
        if not self._client:
            raise RuntimeError(
                f"Local Ollama client not configured. Ensure Ollama is running at {self.ollama_base_url}."
            )

        # Coordinate VRAM residency before inference
        model_manager.prepare_for_model(self.model)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "extra_body": {"keep_alive": self.keep_alive},
        }

        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        start_time = time.perf_counter()
        ttft: Optional[float] = None
        chunks_count = 0

        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                chunks_count += 1
                yield chunk

            duration = time.perf_counter() - start_time
            model_manager.mark_active()

            self.last_metrics = {
                "model": self.model,
                "duration_s": round(duration, 4),
                "ttft_s": round(ttft, 4) if ttft is not None else None,
                "token_count": chunks_count,
                "streamed": True,
            }
            logger.debug(
                f"Streamed inference on {self.model}: TTFT={ttft:.3f}s, total={duration:.3f}s ({chunks_count} chunks)."
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Streamed inference failed on {self.model} after {duration:.3f}s: {e}")
            raise e
