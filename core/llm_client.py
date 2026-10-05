import os
from typing import Any, Dict, List, Optional
from config.settings import settings


class LLMClient:
    """Manages connections to model providers (Groq, OpenAI, etc.)."""

    def __init__(self):
        self.provider = settings.DEFAULT_PROVIDER.lower()
        self.model = settings.DEFAULT_MODEL
        self._client = None
        self._init_client()

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
            # Fallback to OpenAI-compatible
            api_key = settings.OPENAI_API_KEY or settings.GROQ_API_KEY
            if api_key:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            else:
                self._client = None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
    ) -> Any:
        """Sends chat request to the LLM backend with optional tool schemas."""
        if not self._client:
            raise RuntimeError(
                f"No API key found for provider '{self.provider}'. "
                "Please configure GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return response
