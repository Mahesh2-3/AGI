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
from vision.coordinates import CoordinateTransformer

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
        # Local GPU-accelerated Qwen 2.5-VL via Ollama (Zero cloud dependencies)
        ollama_model = getattr(settings, "OLLAMA_VLM_MODEL", "qwen2.5vl:3b")
        self.models.append(
            VLMModelInfo(
                model_id=f"ollama:{ollama_model}",
                provider="ollama",
                model_name=ollama_model,
                description=f"Local Qwen 2.5 VL ({ollama_model} on GPU)",
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
        self.ollama_base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_vlm_model = getattr(settings, "OLLAMA_VLM_MODEL", "qwen2.5vl:3b")
        self._current_user_prompt: Optional[str] = None

    def set_current_user_prompt(self, prompt: Optional[str]) -> None:
        """Sets the active user prompt/intent to contextualize screen inspections."""
        self._current_user_prompt = prompt

    def get_current_user_prompt(self) -> Optional[str]:
        return self._current_user_prompt

    def _prepare_image(self, image_path: Path, max_dimension: int = 1280) -> Tuple[bytes, int, int, int, int]:
        """Loads, optionally resizes, and returns JPEG bytes, original dimensions, and model dimensions."""
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                new_w, new_h = w, h

            import io
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            return buf.getvalue(), w, h, new_w, new_h

    def locate_by_template(
        self,
        template_path: str | Path,
        screenshot_path: Optional[str | Path] = None,
        confidence: float = 0.8,
    ) -> Dict[str, Any]:
        """Method A: Locates a known button or icon on screen using template matching (PyAutoGUI / OpenCV)."""
        template_p = Path(template_path)
        if not template_p.exists():
            return {
                "found": False,
                "error": f"Template image not found: {template_path}",
            }

        try:
            with Image.open(template_p) as t_img:
                tw, th = t_img.size
        except Exception as e:
            return {
                "found": False,
                "error": f"Failed to open template image '{template_path}': {e}",
            }

        # Guard: Reject full-screen desktop screenshots passed as template images
        if tw >= 800 and th >= 600:
            return {
                "found": False,
                "error": (
                    f"Invalid template image: '{template_p.name}' is {tw}x{th} (full-screen screenshot). "
                    "locate_element_by_image is strictly for small cropped icon or button templates. "
                    "To find or click UI elements visible on the screen, use click_element(element_description) "
                    "or locate_element(element_description)."
                ),
            }

        if not screenshot_path:
            shot = capture_engine.capture_full_screen(persistent=False)
            screenshot_p = Path(shot["path"])
        else:
            screenshot_p = Path(screenshot_path)

        with Image.open(screenshot_p) as screen_img:
            orig_w, orig_h = screen_img.size

        # 1. Try pyscreeze template matching directly (pure Pillow, no Tkinter dependency)
        try:
            import pyscreeze
            box = pyscreeze.locate(str(template_p), str(screenshot_p))
            if box is not None:
                x1, y1 = box.left, box.top
                x2, y2 = box.left + box.width, box.top + box.height
                cx = round((x1 + x2) / 2)
                cy = round((y1 + y2) / 2)
                nx1, ny1 = round(x1 / orig_w, 4), round(y1 / orig_h, 4)
                nx2, ny2 = round(x2 / orig_w, 4), round(y2 / orig_h, 4)
                return {
                    "found": True,
                    "target": template_p.stem,
                    "method": "image_matching",
                    "bbox": [nx1, ny1, nx2, ny2],
                    "bbox_pixels": [x1, y1, x2, y2],
                    "x": cx,
                    "y": cy,
                    "confidence": confidence,
                    "description": f"Image template '{template_p.name}' matched at center ({cx}, {cy})",
                    "screen_width": orig_w,
                    "screen_height": orig_h,
                }
        except (Exception, SystemExit) as e:
            logger.debug(f"Pyscreeze template matching skipped: {e}")

        # 2. Try OpenCV template matching if available
        try:
            import cv2
            import numpy as np
            img_rgb = cv2.imread(str(screenshot_p))
            template = cv2.imread(str(template_p))
            if img_rgb is not None and template is not None:
                h, w = template.shape[:2]
                res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= confidence:
                    x1, y1 = max_loc[0], max_loc[1]
                    x2, y2 = x1 + w, y1 + h
                    cx = round((x1 + x2) / 2)
                    cy = round((y1 + y2) / 2)
                    return {
                        "found": True,
                        "target": template_p.stem,
                        "method": "opencv_template_matching",
                        "bbox": [round(x1 / orig_w, 4), round(y1 / orig_h, 4), round(x2 / orig_w, 4), round(y2 / orig_h, 4)],
                        "bbox_pixels": [x1, y1, x2, y2],
                        "x": cx,
                        "y": cy,
                        "confidence": round(float(max_val), 3),
                        "description": f"OpenCV template matched at center ({cx}, {cy}) with score {round(float(max_val), 3)}",
                        "screen_width": orig_w,
                        "screen_height": orig_h,
                    }
        except Exception as e:
            logger.debug(f"OpenCV template matching skipped: {e}")

        return {
            "found": False,
            "error": f"Template image '{template_p.name}' not found on screen with confidence >= {confidence}.",
            "screen_width": orig_w,
            "screen_height": orig_h,
        }

    def locate_element(self, query: str, screenshot_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Locates pixel coordinates (x, y) of a UI element on the screen using normalized bounding boxes."""
        # Method A check: If query refers to an existing image file on disk, use image matching
        clean_q = query.strip()
        possible_img = clean_q[len("image:"):].strip() if clean_q.startswith("image:") else clean_q
        possible_img = possible_img[len("template:"):].strip() if possible_img.startswith("template:") else possible_img
        if Path(possible_img).is_file() and any(possible_img.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp"]):
            return self.locate_by_template(possible_img, screenshot_path=screenshot_path)

        if not screenshot_path:
            shot = capture_engine.capture_full_screen(persistent=False)
            image_path = Path(shot["path"])
        else:
            image_path = Path(screenshot_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Screenshot file not found: {image_path}")

        prep = self._prepare_image(image_path)
        if len(prep) == 5:
            img_bytes, orig_w, orig_h, model_w, model_h = prep
        else:
            img_bytes, orig_w, orig_h = prep[:3]
            model_w, model_h = orig_w, orig_h

        # Method B: Prompt vision model for bounding box [x1, y1, x2, y2] and bounded click point [x, y]
        prompt = f"""You are the visual grounding component of a desktop AI assistant.
Image dimensions: {model_w}x{model_h} (Original screen resolution: {orig_w}x{orig_h}).
Target element to find: "{query}"

Identify the exact target button, link, icon, tab, or input field on screen.
Return its bounding box and the precise bounded point to click.
Return ONLY valid JSON in this format:
{{
  "found": true,
  "target": "{query}",
  "bbox": [x1, y1, x2, y2],
  "point": [x, y],
  "confidence": 0.9,
  "description": "short description of the detected element"
}}

Rules:
- Coordinates [x1, y1, x2, y2] must be normalized between 0.0 and 1.0 (origin [0, 0] at top-left).
- point [x, y] contains the click coordinates inside the element.
- Locate the actual interactive control (button, input, link, tab), not nearby decoration or background.
- If multiple matching targets exist and the intended one is ambiguous, do not guess.
- If the target is missing, not visible, or obscured, return found=false, bbox=null, and point=null.
- NEVER invent coordinates.
- Confidence is an estimate between 0.0 and 1.0.
Do NOT output any markdown backticks or extra text outside the JSON object."""

        available_models = self.tracker.get_available_models()
        if not available_models:
            summary = self.tracker.get_availability_summary()
            return {
                "found": False,
                "x": None,
                "y": None,
                "point": None,
                "bbox": None,
                "bbox_pixels": None,
                "confidence": 0.0,
                "description": f"All VLMs are currently rate-limited or unconfigured ({summary}). Target query was: '{query}'",
                "screen_width": orig_w,
                "screen_height": orig_h,
                "vlm_availability": summary,
            }

        last_error = None
        for model_info in available_models:
            try:
                raw_text = self._call_vlm(model_info, prompt, img_bytes, max_tokens=250, json_format=True)
                if raw_text:
                    parsed = self._parse_coordinate_response(raw_text, orig_w, orig_h, model_w, model_h)
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
            "x": None,
            "y": None,
            "point": None,
            "bbox": None,
            "bbox_pixels": None,
            "confidence": 0.0,
            "description": f"Visual grounding failed across active models: {last_error}",
            "screen_width": orig_w,
            "screen_height": orig_h,
            "vlm_availability": summary,
        }

    def inspect_screen(
        self,
        screenshot_path: Optional[str | Path] = None,
        user_prompt: Optional[str] = None,
        focus: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Provides a context-aware textual understanding of current desktop state with local VLM."""
        if not screenshot_path:
            shot = capture_engine.capture_full_screen(persistent=False)
            image_path = Path(shot["path"])
        else:
            image_path = Path(screenshot_path)

        prep = self._prepare_image(image_path)
        if len(prep) == 5:
            img_bytes, orig_w, orig_h, model_w, model_h = prep
        else:
            img_bytes, orig_w, orig_h = prep[:3]
            model_w, model_h = orig_w, orig_h

        base_prompt = (
            "Describe what is currently visible on this desktop screen. "
            "List active applications, visible windows, open web pages, and notable UI controls concisely."
        )

        current_intent = self._current_user_prompt.strip() if self._current_user_prompt else None
        raw_target = user_prompt or query or focus
        if raw_target and current_intent:
            # If caller passed a generic screen query from a previous turn but current user prompt is a specific task,
            # prioritize the actual active user intent.
            generic_phrases = ["what do you see", "what are you seeing", "see on the screen", "describe the screen", "inspect screen"]
            raw_str = str(raw_target).lower()
            if any(gp in raw_str for gp in generic_phrases) and not any(gp in current_intent.lower() for gp in generic_phrases):
                clean_intent = current_intent
            else:
                clean_intent = str(raw_target).strip()
        elif raw_target:
            clean_intent = str(raw_target).strip()
        elif current_intent:
            clean_intent = current_intent
        else:
            clean_intent = None

        if clean_intent:
            prompt = (
                f"{base_prompt}\n\n"
                f"USER REQUEST / INTENT: \"{clean_intent}\"\n"
                f"Pay special attention to any UI elements, text, windows, tabs, or details relevant to this request. "
                f"Specifically analyze and answer what the user is looking for based on what is visible on screen."
            )
        else:
            prompt = base_prompt

        available_models = self.tracker.get_available_models()
        if not available_models:
            summary = self.tracker.get_availability_summary()
            return {
                "description": f"Visual screen inspection unavailable: all VLMs in cooldown ({summary}).",
                "user_intent": clean_intent,
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
                        "user_intent": clean_intent,
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
            "user_intent": clean_intent,
            "width": orig_w,
            "height": orig_h,
            "vlm_availability": summary,
        }

    def _call_vlm(
        self,
        model_info: VLMModelInfo,
        prompt: str,
        img_bytes: bytes,
        max_tokens: int = 250,
        json_format: bool = False,
    ) -> str:
        """Dispatches an inference request to the local Ollama VLM with centralized lifecycle management."""
        import urllib.request
        from core.model_manager import model_manager

        # Ensure any resident model is unloaded if needed so VLM has dedicated VRAM
        model_manager.prepare_for_model(model_info.model_name)

        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        url = f"{self.ollama_base_url.rstrip('/')}/api/chat"
        idle_timeout = getattr(settings, "OLLAMA_IDLE_TIMEOUT_SECONDS", 300)
        payload: Dict[str, Any] = {
            "model": model_info.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_img],
                }
            ],
            "stream": False,
            "keep_alive": idle_timeout,
            "options": {
                "temperature": 0.0 if json_format else 0.2,
                "num_predict": max_tokens,
            },
        }
        if json_format:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Jarvis-GroundingEngine"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data.get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            err_detail = ""
            try:
                err_detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"Ollama vision request failed [HTTP {e.code}]: {err_detail or e.reason}") from e
        finally:
            # Centralized lifecycle management: mark active and apply idle policy
            try:
                model_manager.mark_active()
                model_manager.unload_on_idle(force=False)
            except Exception:
                pass

    def _parse_coordinate_response(
        self,
        text: str,
        max_w: int,
        max_h: int,
        model_w: Optional[int] = None,
        model_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Safely parses JSON coordinate block with standardized scaling and validation."""
        try:
            cleaned = text.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            data = json.loads(cleaned)

            transformer = CoordinateTransformer(
                orig_w=max_w,
                orig_h=max_h,
                model_w=model_w or max_w,
                model_h=model_h or max_h,
            )

            found = bool(data.get("found", True))
            raw_conf = data.get("confidence")
            try:
                confidence = max(0.0, min(1.0, float(raw_conf))) if raw_conf is not None else None
            except (ValueError, TypeError):
                confidence = None

            if not found:
                return {
                    "found": False,
                    "target": data.get("target") or "",
                    "bbox": None,
                    "bbox_pixels": None,
                    "point": None,
                    "x": None,
                    "y": None,
                    "confidence": confidence or 0.0,
                    "description": data.get("description") or "Target element not found on screen.",
                    "screen_width": max_w,
                    "screen_height": max_h,
                }

            # Parse bounding box independently
            raw_box = data.get("bbox")
            bbox_info = transformer.transform_bbox(raw_box) if raw_box else None

            # Parse point / coordinates
            raw_point = data.get("point")
            if raw_point is None and "x" in data and "y" in data:
                raw_point = [data["x"], data["y"]]

            resolved_point = None
            if raw_point:
                resolved_point = transformer.resolve_click_point(
                    raw_point,
                    bbox_info["bbox_pixels"] if bbox_info else None,
                )

            # If bbox exists but point was missing, derive click point from box center
            if bbox_info and not resolved_point:
                resolved_point = tuple(bbox_info["center"])

            # Validate detection
            if bbox_info is None and resolved_point is None:
                return {
                    "found": False,
                    "target": data.get("target") or "",
                    "bbox": None,
                    "bbox_pixels": None,
                    "point": None,
                    "x": None,
                    "y": None,
                    "confidence": confidence or 0.0,
                    "description": data.get("description") or "No usable in-bounds coordinates returned.",
                    "screen_width": max_w,
                    "screen_height": max_h,
                }

            pt_list = [resolved_point[0], resolved_point[1]] if resolved_point else None
            bbox_norm = bbox_info["bbox"] if bbox_info else (
                [round(pt_list[0] / max_w, 4), round(pt_list[1] / max_h, 4),
                 round(pt_list[0] / max_w, 4), round(pt_list[1] / max_h, 4)] if pt_list else None
            )
            bbox_px = bbox_info["bbox_pixels"] if bbox_info else (
                [pt_list[0], pt_list[1], pt_list[0], pt_list[1]] if pt_list else None
            )

            target_x = pt_list[0] if pt_list else None
            target_y = pt_list[1] if pt_list else None

            return {
                "found": True,
                "target": data.get("target") or data.get("description", ""),
                "bbox": bbox_norm,
                "bbox_pixels": bbox_px,
                "point": pt_list,
                "x": target_x,
                "y": target_y,
                "confidence": confidence if confidence is not None else 0.85,
                "description": data.get("description") or f"Detected at point ({target_x}, {target_y})",
                "screen_width": max_w,
                "screen_height": max_h,
            }

        except Exception as e:
            return {
                "found": False,
                "error": f"Failed to parse VLM response: {e} (raw: {text[:200]})",
                "bbox": None,
                "bbox_pixels": None,
                "point": None,
                "x": None,
                "y": None,
                "confidence": 0.0,
                "screen_width": max_w,
                "screen_height": max_h,
            }


grounding_engine = VisualGroundingEngine()


@registry.register(description="Analyzes the screen and locates the normalized bounding box and pixel coordinates (x, y) of a target UI button, icon, link, or text.")
def locate_element(element_description: str) -> Dict[str, Any]:
    return grounding_engine.locate_element(query=element_description)


@registry.register(
    description=(
        "[ICON CROP TEMPLATE MATCHING ONLY] Locates a small cropped icon or button template image on screen using OpenCV pixel matching. "
        "DO NOT pass full-screen desktop screenshots here. For finding or clicking UI elements, buttons, or links visible on the screen, "
        "always use click_element(element_description) or locate_element(element_description)."
    )
)
def locate_element_by_image(
    image_path: Optional[str] = None,
    template_path: Optional[str] = None,
    confidence: float = 0.8,
    **kwargs: Any,
) -> Dict[str, Any]:
    target_path = image_path or template_path or kwargs.get("path")
    if not target_path:
        return {
            "found": False,
            "error": "Missing required image_path parameter for template matching.",
        }
    return grounding_engine.locate_by_template(template_path=target_path, confidence=confidence)


@registry.register(
    description=(
        "Visually inspects the user's screen and active desktop windows. "
        "If user_prompt, focus, or query is provided, specifically analyzes and answers that need "
        "in addition to providing general visible desktop state."
    )
)
def inspect_screen(
    user_prompt: Optional[str] = None,
    focus: Optional[str] = None,
    query: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return grounding_engine.inspect_screen(user_prompt=user_prompt, focus=focus, query=query, **kwargs)


@registry.register(description="Checks the health, cooldown state, and availability count of all configured Vision-Language Models (VLMs) (e.g. '4/4 VLMs available').")
def get_vlm_status() -> Dict[str, Any]:
    return vlm_tracker.get_status_report()
