import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image

from config.settings import settings
from vision.coordinates import CoordinateTransformer

logger = logging.getLogger("JARVIS.VisionAnalyzer")


def _encode_image(image_path: Union[str, Path], max_dimension: int = 1280) -> Tuple[str, int, int, int, int]:
    """Validates, reads, and Base64-encodes an image, returning (base64_str, orig_w, orig_h, model_w, model_h)."""
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a regular file: {p}")

    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            orig_w, orig_h = img.size
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            else:
                img = img.copy()
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file ({p}): {e}") from e

    # Resize if larger than max_dimension to keep GPU VRAM footprint minimal
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dimension:
        scale = max_dimension / max(orig_w, orig_h)
        w = max(1, int(orig_w * scale))
        h = max(1, int(orig_h * scale))
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64_str, orig_w, orig_h, w, h


def _parse_structured_response(
    raw_content: str,
    orig_w: int,
    orig_h: int,
    model_w: Optional[int] = None,
    model_h: Optional[int] = None,
    target_query: str = "",
) -> Dict[str, Any]:
    """Extracts bounding boxes and structured coordinates from VLM response using standardized coordinate scaling."""
    text = raw_content.strip()
    # Strip markdown backticks
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            end_idx = -1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[1:end_idx]).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        # Search for first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except Exception:
                pass

    transformer = CoordinateTransformer(
        orig_w=orig_w,
        orig_h=orig_h,
        model_w=model_w or orig_w,
        model_h=model_h or orig_h,
    )

    if isinstance(parsed, dict):
        found = bool(parsed.get("found", True))
        target_desc = parsed.get("target") or target_query
        explanation = parsed.get("explanation") or parsed.get("description") or ""

        # Validate confidence: never assume high confidence if omitted
        raw_conf = parsed.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(raw_conf))) if raw_conf is not None else 0.0
        except (ValueError, TypeError):
            confidence = 0.0

        # Parse bounding box independently
        raw_box = parsed.get("bbox")
        bbox_info = transformer.transform_bbox(raw_box) if raw_box else None

        # Parse point independently
        raw_point = parsed.get("point")
        resolved_point = None
        if raw_point and isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
            resolved_point = transformer.resolve_click_point(
                raw_point,
                bbox_info["bbox_pixels"] if bbox_info else None,
            )

        # If bbox exists but point was missing, derive click point from box center
        if bbox_info and not resolved_point:
            resolved_point = tuple(bbox_info["center"])

        # Validate: target is found ONLY if found=True and at least one valid location was parsed
        if not found or (bbox_info is None and resolved_point is None):
            return {
                "success": True,
                "found": False,
                "target": target_desc,
                "bbox": None,
                "bbox_pixels": None,
                "point": None,
                "confidence": confidence,
                "explanation": explanation or "Target element not located on screen.",
                "image_dimensions": {"width": orig_w, "height": orig_h},
            }

        point_list = [resolved_point[0], resolved_point[1]] if resolved_point else None
        bbox_norm = bbox_info["bbox"] if bbox_info else (
            [round(point_list[0] / orig_w, 4), round(point_list[1] / orig_h, 4),
             round(point_list[0] / orig_w, 4), round(point_list[1] / orig_h, 4)] if point_list else None
        )
        bbox_px = bbox_info["bbox_pixels"] if bbox_info else (
            [point_list[0], point_list[1], point_list[0], point_list[1]] if point_list else None
        )

        return {
            "success": True,
            "found": True,
            "target": target_desc,
            "bbox": bbox_norm,
            "bbox_pixels": bbox_px,
            "point": point_list,
            "confidence": confidence if raw_conf is not None else 0.85,
            "explanation": explanation or f"Located target at ({point_list[0]}, {point_list[1]}).",
            "image_dimensions": {"width": orig_w, "height": orig_h},
        }

    # Could not parse structured JSON but got raw text
    return {
        "success": True,
        "found": False,
        "target": target_query,
        "bbox": None,
        "bbox_pixels": None,
        "point": None,
        "confidence": 0.0,
        "explanation": raw_content.strip() or "No valid coordinates returned.",
        "image_dimensions": {"width": orig_w, "height": orig_h},
    }


def analyze_image(
    image_path: Union[str, Path],
    prompt: str,
    structured: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 45,
) -> Dict[str, Any]:
    """Analyzes an image using local Qwen2.5-VL via Ollama with standardized coordinate tracking.

    Args:
        image_path: Path to image file or captured screenshot.
        prompt: Natural language query (e.g. 'Describe screen', 'Find terminal window').
        structured: If True, returns structured JSON with bounding boxes and coordinates.
        model: VLM model name (defaults to settings.OLLAMA_VLM_MODEL or 'qwen2.5vl:3b').
        base_url: Ollama API URL (defaults to settings.OLLAMA_BASE_URL or 'http://localhost:11434').
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict containing analysis results, coordinates/bbox if requested, and metadata.
    """
    target_model = model or getattr(settings, "OLLAMA_VLM_MODEL", "qwen2.5vl:3b")
    endpoint_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    # 1. Validate and encode image (tracking both screen and model dimensions)
    try:
        b64_img, orig_w, orig_h, model_w, model_h = _encode_image(image_path)
    except FileNotFoundError as e:
        return {"success": False, "error": f"Image not found: {e}", "found": False}
    except ValueError as e:
        return {"success": False, "error": f"Invalid image: {e}", "found": False}
    except Exception as e:
        return {"success": False, "error": f"Failed reading image: {e}", "found": False}

    # 2. Determine if structured coordinate output is needed
    is_structured = structured or any(
        kw in prompt.lower()
        for kw in ["find", "locate", "identify the text and buttons", "where is", "bounding box", "coordinates", "click"]
    )

    formatted_prompt = prompt.strip()
    if is_structured:
        formatted_prompt = (
            f"You are the visual grounding engine of a desktop AI assistant.\n"
            f"User request: {prompt}\n"
            f"Image dimensions: {model_w}x{model_h} (Original screen resolution: {orig_w}x{orig_h}).\n\n"
            f"Identify the exact requested target in the image and return ONLY a JSON object with this format:\n"
            f"{{\n"
            f'  "found": true,\n'
            f'  "target": "description of matched element",\n'
            f'  "bbox": [x1, y1, x2, y2],\n'
            f'  "point": [x, y],\n'
            f'  "confidence": 0.9,\n'
            f'  "explanation": "short visual explanation"\n'
            f"}}\n"
            f"Rules:\n"
            f"- Coordinates [x1, y1, x2, y2] must be normalized between 0.0 and 1.0 (origin [0, 0] at top-left).\n"
            f"- point [x, y] contains the click coordinates inside the element.\n"
            f"- Locate the interactive control or window, not decorative space.\n"
            f"- If the target is missing or obscured, return found=false, bbox=null, and point=null.\n"
            f"- NEVER invent coordinates.\n"
            f"- Output ONLY valid JSON, no markdown backticks or commentary."
        )

    # 3. Model residency management: ensure reasoning models are unloaded to free GPU VRAM
    try:
        from core.model_manager import model_manager
        model_manager.prepare_for_model(target_model)
    except Exception:
        pass

    # 4. Dispatch request to Ollama /api/chat with task-specific token limits and centralized keep_alive
    chat_url = f"{endpoint_url}/api/chat"
    idle_timeout = getattr(settings, "OLLAMA_IDLE_TIMEOUT_SECONDS", 300)
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt,
                "images": [b64_img],
            }
        ],
        "stream": False,
        "keep_alive": idle_timeout,
        "options": {
            "temperature": 0.0 if is_structured else 0.2,
            "num_predict": 200 if is_structured else 400,
        },
    }
    if is_structured:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Jarvis-VisionAnalyzer"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            raw_content = resp_body.get("message", {}).get("content", "")
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_msg = e.read().decode("utf-8")
        except Exception:
            pass
        return {
            "success": False,
            "error": f"Ollama HTTP {e.code} error: {err_msg or e.reason}",
            "found": False,
        }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "error": f"Could not connect to Ollama at {endpoint_url}: {e.reason}",
            "found": False,
        }
    except TimeoutError:
        return {
            "success": False,
            "error": f"Vision model inference timed out after {timeout} seconds.",
            "found": False,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Vision inference failed: {e}",
            "found": False,
        }
    finally:
        # Centralized lifecycle management: mark active and apply idle policy
        try:
            from core.model_manager import model_manager
            model_manager.mark_active()
            model_manager.unload_on_idle(force=False)
        except Exception:
            pass

    # 5. Format and return output
    if is_structured:
        result = _parse_structured_response(raw_content, orig_w, orig_h, model_w, model_h, target_query=prompt)
        result["raw_response"] = raw_content
        return result

    return {
        "success": True,
        "description": raw_content.strip(),
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "model": target_model,
    }
