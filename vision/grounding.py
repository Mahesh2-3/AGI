import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from config.settings import settings
from tools.base import registry
from vision.capture import capture_engine


class VisualGroundingEngine:
    """Uses Vision-Language Models (VLM) to ground visual elements to screen coordinates."""

    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.google_api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.groq_vision_model = "qwen/qwen3.8-27b"

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
        """Locates pixel coordinates (x, y) of a UI element on the screen."""
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

        if self.groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model=self.groq_vision_model,
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
                    max_tokens=250,
                )
                return self._parse_coordinate_response(response.choices[0].message.content, orig_w, orig_h)
            except Exception as e:
                pass

        if self.google_api_key:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.google_api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                )
                return self._parse_coordinate_response(response.text, orig_w, orig_h)
            except Exception as e:
                pass

        if self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            ],
                        }
                    ],
                )
                return self._parse_coordinate_response(response.choices[0].message.content, orig_w, orig_h)
            except Exception as e:
                pass

        # Return simulated mock coordinate if no vision API key is configured yet
        return {
            "found": False,
            "x": orig_w // 2,
            "y": orig_h // 2,
            "confidence": 0.0,
            "description": f"Visual grounding requires GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in .env. Target query was: '{query}'",
            "screen_width": orig_w,
            "screen_height": orig_h,
        }

    def inspect_screen(self, screenshot_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Provides a textual understanding of current desktop state."""
        if not screenshot_path:
            shot = capture_engine.capture_full_screen()
            image_path = Path(shot["path"])
        else:
            image_path = Path(screenshot_path)

        img_bytes, orig_w, orig_h = self._prepare_image(image_path)

        prompt = "Describe what is currently visible on this desktop screen. List active applications, visible windows, open web pages, and notable UI controls concisely."

        if self.groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model=self.groq_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            ],
                        }
                    ],
                    temperature=0.2,
                    max_tokens=400,
                )
                return {"description": response.choices[0].message.content.strip(), "width": orig_w, "height": orig_h}
            except Exception as e:
                pass

        if self.google_api_key:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.google_api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                )
                return {"description": response.text.strip(), "width": orig_w, "height": orig_h}
            except Exception as e:
                pass

        if self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            ],
                        }
                    ],
                )
                return {"description": response.choices[0].message.content.strip(), "width": orig_w, "height": orig_h}
            except Exception as e:
                pass

        return {
            "description": "Visual screen inspection is ready. To enable live visual understanding, set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in .env.",
            "width": orig_w,
            "height": orig_h,
        }

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
