import base64
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from config.settings import settings
from tools.base import registry
from vision.capture import capture_engine

logger = logging.getLogger("jarvis.vision")


@dataclass
class VLMModelInfo:
    """Metadata and operational status for an individual Vision-Language Model."""
    model_id: str          # e.g. "groq:qwen/qwen3.8-27b"
    provider: str          # "groq", "google", "openai"
    model_name: str        # API model identifier
    description: str       # Human-readable title
    cooldown_until: float = 0.0
    rate_limit_count: int = 0
    consecutive_errors: int = 0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        rem = self.cooldown_until - time.time()
        return round(max(0.0, rem), 1)


class VLMHealthTracker:
    """Monitors the operational health, rate limits, and availability of all Vision-Language Models (VLMs)."""

    def __init__(self):
        self.models: List[VLMModelInfo] = []
        self._initialize_models()

    def _initialize_models(self):
        self.models.clear()
        groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        google_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        openai_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

        if groq_key:
            self.models.append(
                VLMModelInfo(
                    model_id="groq:qwen/qwen3.8-27b",
                    provider="groq",
                    model_name="qwen/qwen3.8-27b",
                    description="Groq Qwen 2.5 27B Vision",
                )
            )

        if google_key:
            self.models.append(
                VLMModelInfo(
                    model_id="google:gemini-3.5-flash-lite",
                    provider="google",
                    model_name="gemini-3.5-flash-lite",
                    description="Google Gemini 3.5 Flash-Lite",
                )
            )
            self.models.append(
                VLMModelInfo(
                    model_id="google:gemini-3.1-flash-lite",
                    provider="google",
                    model_name="gemini-3.1-flash-lite",
                    description="Google Gemini 3.1 Flash-Lite",
                )
            )
            self.models.append(
                VLMModelInfo(
                    model_id="google:gemini-flash-lite-latest",
                    provider="google",
                    model_name="gemini-flash-lite-latest",
                    description="Google Gemini Flash-Lite Latest",
                )
            )

        if openai_key:
            self.models.append(
                VLMModelInfo(
                    model_id="openai:gpt-4o-mini",
                    provider="openai",
                    model_name="gpt-4o-mini",
                    description="OpenAI GPT-4o Mini Vision",
                )
            )

    def mark_rate_limited(self, model_id: str, cooldown_seconds: float = 60.0):
        for m in self.models:
            if m.model_id == model_id:
                m.cooldown_until = time.time() + cooldown_seconds
                m.rate_limit_count += 1
                logger.warning(
                    f"⚠️ VLM Rate limit encountered on {model_id}. Cooldown for {cooldown_seconds}s. ({self.get_availability_summary()})"
                )
                break

    def mark_success(self, model_id: str):
        for m in self.models:
            if m.model_id == model_id:
                m.consecutive_errors = 0
                break

    def mark_error(self, model_id: str):
        for m in self.models:
            if m.model_id == model_id:
                m.consecutive_errors += 1
                if m.consecutive_errors >= 3:
                    m.cooldown_until = time.time() + 30.0
                break

    def get_available_models(self) -> List[VLMModelInfo]:
        return [m for m in self.models if m.is_available]

    def get_availability_summary(self) -> str:
        total = len(self.models)
        if total == 0:
            return "0/0 VLMs available (No vision API keys configured)"
        available = len(self.get_available_models())
        rate_limited = total - available
        if rate_limited == 0:
            return f"{available}/{total} VLMs available"
        return f"{available}/{total} VLMs available ({rate_limited} cooling down)"

    def get_status_report(self) -> Dict[str, Any]:
        total = len(self.models)
        avail_list = self.get_available_models()
        return {
            "summary": self.get_availability_summary(),
            "total_vlm_count": total,
            "available_vlm_count": len(avail_list),
            "rate_limited_count": total - len(avail_list),
            "models": [
                {
                    "model_id": m.model_id,
                    "provider": m.provider,
                    "model_name": m.model_name,
                    "description": m.description,
                    "status": "AVAILABLE" if m.is_available else "RATE_LIMITED",
                    "cooldown_remaining_sec": m.cooldown_remaining,
                    "rate_limit_count": m.rate_limit_count,
                }
                for m in self.models
            ],
        }


vlm_tracker = VLMHealthTracker()


class VisualGroundingEngine:
    """Uses Vision-Language Models (VLM) with automatic multi-model failover to ground visual elements to screen coordinates."""

    def __init__(self, tracker: Optional[VLMHealthTracker] = None):
        self.tracker = tracker or vlm_tracker
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.google_api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")

    def _prepare_image(self, image_path: Path, max_dimension: int = 1920) -> Tuple[bytes, int, int]:
        """Loads, optionally resizes, and returns JPEG bytes and actual dimensions."""
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                new_w, new_h = w, h

            import io
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            return buf.getvalue(), w, h

    def locate_element(self, query: str, screenshot_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Locates pixel coordinates (x, y) of a UI element on the screen with VLM cascade."""
        if not screenshot_path:
            shot = capture_engine.capture_full_screen()
            image_path = Path(shot["path"])
        else:
            image_path = Path(screenshot_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Screenshot file not found: {image_path}")

        img_bytes, orig_w, orig_h = self._prepare_image(image_path)

        prompt = f"""You are a precise screen coordinate locator for a desktop AI assistant.
Analyze this desktop screenshot (Resolution: {orig_w}x{orig_h}).
Target element to find: "{query}"

Output ONLY a valid JSON object with these exact keys:
{{
  "found": true or false,
  "x": integer pixel coordinate (0 to {orig_w}) at the center of the element,
  "y": integer pixel coordinate (0 to {orig_h}) at the center of the element,
  "confidence": float between 0.0 and 1.0,
  "description": "short description of the detected element"
}}
Do NOT output any markdown backticks or extra text outside the JSON object."""

        available_models = self.tracker.get_available_models()
        if not available_models:
            summary = self.tracker.get_availability_summary()
            return {
                "found": False,
                "x": orig_w // 2,
                "y": orig_h // 2,
                "confidence": 0.0,
                "description": f"All VLMs are currently rate-limited or unconfigured ({summary}). Target query was: '{query}'",
                "screen_width": orig_w,
                "screen_height": orig_h,
                "vlm_availability": summary,
            }

        last_error = None
        for model_info in available_models:
            try:
                raw_text = self._call_vlm(model_info, prompt, img_bytes)
                if raw_text:
                    parsed = self._parse_coordinate_response(raw_text, orig_w, orig_h)
                    self.tracker.mark_success(model_info.model_id)
                    parsed["vlm_model_used"] = model_info.model_id
                    parsed["vlm_availability"] = self.tracker.get_availability_summary()
                    return parsed
            except Exception as e:
                err_str = str(e).lower()
                last_error = str(e)
                if any(k in err_str for k in ["429", "rate limit", "rate_limit", "quota", "resource_exhausted", "503", "unavailable"]):
                    self.tracker.mark_rate_limited(model_info.model_id, cooldown_seconds=60.0)
                else:
                    self.tracker.mark_error(model_info.model_id)
                continue

        summary = self.tracker.get_availability_summary()
        return {
            "found": False,
            "x": orig_w // 2,
            "y": orig_h // 2,
            "confidence": 0.0,
            "description": f"Visual grounding failed across active models: {last_error}",
            "screen_width": orig_w,
            "screen_height": orig_h,
            "vlm_availability": summary,
        }

    def inspect_screen(self, screenshot_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Provides a textual understanding of current desktop state with VLM cascade."""
        if not screenshot_path:
            shot = capture_engine.capture_full_screen()
            image_path = Path(shot["path"])
        else:
            image_path = Path(screenshot_path)

        img_bytes, orig_w, orig_h = self._prepare_image(image_path)

        prompt = "Describe what is currently visible on this desktop screen. List active applications, visible windows, open web pages, and notable UI controls concisely."

        available_models = self.tracker.get_available_models()
        if not available_models:
            summary = self.tracker.get_availability_summary()
            return {
                "description": f"Visual screen inspection unavailable: all VLMs in cooldown ({summary}).",
                "width": orig_w,
                "height": orig_h,
                "vlm_availability": summary,
            }

        last_error = None
        for model_info in available_models:
            try:
                raw_text = self._call_vlm(model_info, prompt, img_bytes, max_tokens=400)
                if raw_text:
                    self.tracker.mark_success(model_info.model_id)
                    return {
                        "description": raw_text.strip(),
                        "width": orig_w,
                        "height": orig_h,
                        "vlm_model_used": model_info.model_id,
                        "vlm_availability": self.tracker.get_availability_summary(),
                    }
            except Exception as e:
                err_str = str(e).lower()
                last_error = str(e)
                if any(k in err_str for k in ["429", "rate limit", "rate_limit", "quota", "resource_exhausted", "503", "unavailable"]):
                    self.tracker.mark_rate_limited(model_info.model_id, cooldown_seconds=60.0)
                else:
                    self.tracker.mark_error(model_info.model_id)
                continue

        summary = self.tracker.get_availability_summary()
        return {
            "description": f"Screen inspection failed across active models: {last_error}",
            "width": orig_w,
            "height": orig_h,
            "vlm_availability": summary,
        }

    def _call_vlm(self, model_info: VLMModelInfo, prompt: str, img_bytes: bytes, max_tokens: int = 250) -> str:
        """Dispatches an inference request to the appropriate VLM provider."""
        if model_info.provider == "groq":
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            response = client.chat.completions.create(
                model=model_info.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        elif model_info.provider == "google":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.google_api_key)
            response = client.models.generate_content(
                model=model_info.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            return response.text or ""

        elif model_info.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            response = client.chat.completions.create(
                model=model_info.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        raise ValueError(f"Unknown provider '{model_info.provider}'")

    def _parse_coordinate_response(self, text: str, max_w: int, max_h: int) -> Dict[str, Any]:
        """Safely parses JSON coordinate block from model output."""
        try:
            cleaned = text.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            data = json.loads(cleaned)
            # Clamp coordinates to screen bounds
            if "x" in data:
                data["x"] = max(0, min(max_w, int(data["x"])))
            if "y" in data:
                data["y"] = max(0, min(max_h, int(data["y"])))
            return data
        except Exception:
            return {
                "found": False,
                "error": f"Failed to parse VLM response: {text[:200]}",
                "screen_width": max_w,
                "screen_height": max_h,
            }


grounding_engine = VisualGroundingEngine()


@registry.register(description="Analyzes the screen and locates the pixel coordinates (x, y) of a target UI button, icon, link, or text.")
def locate_element(element_description: str) -> Dict[str, Any]:
    return grounding_engine.locate_element(query=element_description)


@registry.register(description="Visually inspects the user's screen and describes what is currently visible.")
def inspect_screen() -> Dict[str, Any]:
    return grounding_engine.inspect_screen()


@registry.register(description="Checks the health, cooldown state, and availability count of all configured Vision-Language Models (VLMs) (e.g. '4/4 VLMs available').")
def get_vlm_status() -> Dict[str, Any]:
    return vlm_tracker.get_status_report()
