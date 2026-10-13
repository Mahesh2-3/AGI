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

logger = logging.getLogger("JARVIS.VisionAnalyzer")


def _encode_image(image_path: Union[str, Path], max_dimension: int = 1280) -> Tuple[str, int, int]:
    """Validates, reads, and Base64-encodes an image, returning (base64_str, width, height)."""
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
    return b64_str, orig_w, orig_h


def _parse_structured_response(
    raw_content: str,
    orig_w: int,
    orig_h: int,
    target_query: str = "",
) -> Dict[str, Any]:
    """Extracts bounding boxes and structured coordinates from VLM response."""
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

    if isinstance(parsed, dict):
        found = bool(parsed.get("found", True))
        if not found or parsed.get("bbox") is None:
            return {
                "success": True,
                "found": False,
                "target": parsed.get("target", target_query),
                "bbox": None,
                "bbox_pixels": None,
                "point": None,
                "confidence": float(parsed.get("confidence", 0.0)),
                "explanation": parsed.get("explanation") or parsed.get("description") or "Target element not located on screen.",
                "image_dimensions": {"width": orig_w, "height": orig_h},
            }

        raw_box = parsed.get("bbox")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            b = [float(v) for v in raw_box]
            # Detect coordinate format: normalized 0-1 vs pixel coordinates
            if max(b) > 1.0:
                # Returned as absolute pixels
                px1, py1, px2, py2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x1 = round(px1 / orig_w, 4)
                y1 = round(py1 / orig_h, 4)
                x2 = round(px2 / orig_w, 4)
                y2 = round(py2 / orig_h, 4)
            else:
                # Returned as normalized [x1, y1, x2, y2]
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                px1 = int(round(x1 * orig_w))
                py1 = int(round(y1 * orig_h))
                px2 = int(round(x2 * orig_w))
                py2 = int(round(y2 * orig_h))

            cx = int(round((px1 + px2) / 2))
            cy = int(round((py1 + py2) / 2))

            return {
                "success": True,
                "found": True,
                "target": parsed.get("target", target_query),
                "bbox": [x1, y1, x2, y2],
                "bbox_pixels": [px1, py1, px2, py2],
                "point": [cx, cy],
                "confidence": float(parsed.get("confidence", 0.9)),
                "explanation": parsed.get("explanation") or parsed.get("description") or f"Identified {target_query} at ({cx}, {cy}).",
                "image_dimensions": {"width": orig_w, "height": orig_h},
            }

        # Check for single point
        point = parsed.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            pt = [float(v) for v in point]
            if max(pt) > 1.0:
                cx, cy = int(pt[0]), int(pt[1])
            else:
                cx = int(round(pt[0] * orig_w))
                cy = int(round(pt[1] * orig_h))
            return {
                "success": True,
                "found": True,
                "target": parsed.get("target", target_query),
                "bbox": [round(cx / orig_w, 4), round(cy / orig_h, 4), round(cx / orig_w, 4), round(cy / orig_h, 4)],
                "bbox_pixels": [cx, cy, cx, cy],
                "point": [cx, cy],
                "confidence": float(parsed.get("confidence", 0.85)),
                "explanation": parsed.get("explanation") or f"Located target at ({cx}, {cy}).",
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
    """Analyzes an image using local Qwen2.5-VL via Ollama.

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

    # 1. Validate and encode image
    try:
        b64_img, orig_w, orig_h = _encode_image(image_path)
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
            f"Image dimensions: {orig_w}x{orig_h}.\n\n"
            f"Identify the exact requested target in the image and return ONLY a JSON object with this format:\n"
            f"{{\n"
            f'  "found": true,\n'
            f'  "target": "description of matched element",\n'
            f'  "bbox": [x1, y1, x2, y2],\n'
            f'  "confidence": 0.95,\n'
            f'  "explanation": "short visual explanation"\n'
            f"}}\n"
            f"Rules:\n"
            f"- Coordinates [x1, y1, x2, y2] must be normalized between 0.0 and 1.0 (origin [0, 0] at top-left).\n"
            f"- Locate the interactive control or window, not decorative space.\n"
            f"- If the target is missing or obscured, return found=false and bbox=null.\n"
            f"- NEVER invent coordinates.\n"
            f"- Output ONLY valid JSON, no markdown backticks or commentary."
        )

    # 3. Model residency management: ensure reasoning models are unloaded to free RTX 3050 VRAM
    try:
        from core.model_manager import model_manager
        model_manager.prepare_for_model(target_model)
    except Exception:
        pass

    # 4. Dispatch request to Ollama /api/chat
    chat_url = f"{endpoint_url}/api/chat"
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
        "keep_alive": 0,  # Unload immediately after inference to prevent GPU memory/heating spikes
        "options": {
            "temperature": 0.0 if is_structured else 0.2,
            "num_predict": 400 if is_structured else 600,
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
        # Guarantee vision model is purged from GPU VRAM after call
        try:
            from core.model_manager import model_manager
            model_manager.unload_model(target_model)
        except Exception:
            pass

    # 5. Format and return output
    if is_structured:
        result = _parse_structured_response(raw_content, orig_w, orig_h, target_query=prompt)
        result["raw_response"] = raw_content
        return result

    return {
        "success": True,
        "description": raw_content.strip(),
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "model": target_model,
    }
