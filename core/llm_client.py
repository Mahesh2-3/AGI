import logging
import os
from typing import Any, Callable, Dict, Generator, List, Optional
from config.settings import settings

logger = logging.getLogger("JARVIS.LLM")

# Groq free-tier model cascade hierarchy for zero downtime / rate-limit resilience
GROQ_CASCADE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


class LLMClient:
    """Manages connections to model providers (Groq, OpenAI, etc.) with automatic rate-limit failover cascading."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        on_fallback: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.provider = (provider or settings.DEFAULT_PROVIDER).lower()
        self.model = model or settings.DEFAULT_MODEL
        self.on_fallback = on_fallback
        self._client = None
        self._init_client()

        # Cooldown tracker: model_name -> expiry timestamp
        self._model_cooldowns: Dict[str, float] = {}

        # Build cascade list ensuring primary model is first
        if self.provider == "groq":
            self.model_cascade = [self.model] + [m for m in GROQ_CASCADE_MODELS if m != self.model]
        else:
            self.model_cascade = [self.model]

    def _get_active_cascade(self) -> List[str]:
        """Returns candidate models with cooled-down models prioritized."""
        import time
        now = time.time()
        ready = []
        cooling = []
        for m in self.model_cascade:
            cooldown_until = self._model_cooldowns.get(m, 0.0)
            if now >= cooldown_until:
                ready.append(m)
            else:
                cooling.append(m)
        # Try ready models first, then cooling models as backup
        return ready if ready else cooling

    def _init_client(self):
        if self.provider == "groq":
            api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if api_key:
                from groq import Groq
                self._client = Groq(api_key=api_key)
            else:
                self._client = None
        elif self.provider == "openai":
            api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            else:
                self._client = None
        else:
            api_key = settings.OPENAI_API_KEY or settings.GROQ_API_KEY
            if api_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            else:
                self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def _is_rate_limit_error(self, err: Exception) -> bool:
        """Determines if the exception is due to rate limits or quota."""
        err_str = str(err).lower()
        if "rate_limit" in err_str or "rate limit" in err_str or "429" in err_str:
            return True
        status_code = getattr(err, "status_code", None)
        if status_code == 429:
            return True
        return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
    ) -> Any:
        """Sends chat request to the LLM backend with automatic fallback on rate limits."""
        if not self._client:
            raise RuntimeError(
                f"No API key found for provider '{self.provider}'. "
                "Please configure GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )

        last_exception = None
        import time
        cascade = self._get_active_cascade()

        for idx, candidate_model in enumerate(cascade):
            kwargs: Dict[str, Any] = {
                "model": candidate_model,
                "messages": messages,
                "temperature": temperature,
            }

            if tools and len(tools) > 0:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            try:
                response = self._client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                last_exception = e
                if self._is_rate_limit_error(e) and idx < len(cascade) - 1:
                    self._model_cooldowns[candidate_model] = time.time() + 60.0
                    next_model = cascade[idx + 1]
                    logger.warning(
                        f"Rate limit encountered on {candidate_model}. Cascading failover to {next_model}."
                    )
                    if self.on_fallback:
                        self.on_fallback(candidate_model, next_model, str(e))
                    continue
                # If it's not a rate limit error or we have exhausted candidates, raise
                raise e

        if last_exception:
            raise last_exception

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
    ) -> Generator[Any, None, None]:
        """Streams chat completion tokens with automatic fallback on rate limits."""
        if not self._client:
            raise RuntimeError(
                f"No API key found for provider '{self.provider}'. "
                "Please configure GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )

        last_exception = None
        import time
        cascade = self._get_active_cascade()

        for idx, candidate_model in enumerate(cascade):
            kwargs: Dict[str, Any] = {
                "model": candidate_model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }

            if tools and len(tools) > 0:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            try:
                stream = self._client.chat.completions.create(**kwargs)
                for chunk in stream:
                    yield chunk
                return
            except Exception as e:
                last_exception = e
                if self._is_rate_limit_error(e) and idx < len(cascade) - 1:
                    self._model_cooldowns[candidate_model] = time.time() + 60.0
                    next_model = cascade[idx + 1]
                    logger.warning(
                        f"Rate limit encountered on {candidate_model}. Cascading failover stream to {next_model}."
                    )
                    if self.on_fallback:
                        self.on_fallback(candidate_model, next_model, str(e))
                    continue
                raise e

        if last_exception:
            raise last_exception
