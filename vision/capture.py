import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from config.settings import settings
from tools.base import registry

SCREENSHOTS_DIR = settings.WORKSPACE_DIR / "screenshots"


class ScreenCaptureEngine:
    """High-speed screen capture engine optimized for Linux Wayland & Hyprland."""

    def __init__(self):
        self.screenshots_dir = SCREENSHOTS_DIR
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.has_grim = bool(shutil.which("grim"))

    def get_display_geometry(self) -> Dict[str, Any]:
        """Queries monitor resolution and active workspace."""
        if shutil.which("hyprctl"):
            try:
                res = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, check=True)
                monitors = json.loads(res.stdout)
                if monitors:
                    primary = monitors[0]
                    return {
                        "name": primary.get("name"),
                        "width": primary.get("width"),
                        "height": primary.get("height"),
                        "scale": primary.get("scale", 1.0),
                        "refresh_rate": primary.get("refreshRate"),
                        "active_workspace": primary.get("activeWorkspace", {}).get("name"),
                    }
            except Exception:
                pass
        return {"width": 1920, "height": 1080, "scale": 1.0}

    def capture_full_screen(
        self,
        output_path: Optional[str | Path] = None,
        include_cursor: bool = False,
        quality: int = 80,
    ) -> Dict[str, Any]:
        """Captures full desktop screenshot."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            output_path = self.screenshots_dir / f"screen_{timestamp}.jpg"
        else:
            output_path = Path(output_path).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_t = time.perf_counter()

        if self.has_grim:
            cmd = ["grim", "-t", "jpeg", "-q", str(quality)]
            if include_cursor:
                cmd.append("-c")
            cmd.append(str(output_path))
            subprocess.run(cmd, check=True)
        else:
            # Fallback to scrot or Pillow ImageGrab if X11
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(output_path), "JPEG", quality=quality)

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
        with Image.open(output_path) as img:
            width, height = img.size

        return {
            "path": str(output_path),
            "filename": output_path.name,
            "width": width,
            "height": height,
            "file_size_bytes": output_path.stat().st_size,
            "capture_latency_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat(),
        }

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        output_path: Optional[str | Path] = None,
        quality: int = 80,
    ) -> Dict[str, Any]:
        """Captures a specific region of the screen (x, y, w, h)."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            output_path = self.screenshots_dir / f"region_{timestamp}.jpg"
        else:
            output_path = Path(output_path).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_t = time.perf_counter()

        if self.has_grim:
            geometry_str = f"{x},{y} {width}x{height}"
            cmd = ["grim", "-g", geometry_str, "-t", "jpeg", "-q", str(quality), str(output_path)]
            subprocess.run(cmd, check=True)
        else:
            from PIL import ImageGrab
            bbox = (x, y, x + width, y + height)
            img = ImageGrab.grab(bbox=bbox)
            img.save(str(output_path), "JPEG", quality=quality)

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "path": str(output_path),
            "filename": output_path.name,
            "width": width,
            "height": height,
            "file_size_bytes": output_path.stat().st_size,
            "capture_latency_ms": elapsed_ms,
        }

    def capture_window(
        self,
        query: str,
        output_path: Optional[str | Path] = None,
        quality: int = 80,
    ) -> Dict[str, Any]:
        """Locates an open window by title or class and captures only its bounding box."""
        query = query.strip().lower()
        if shutil.which("hyprctl"):
            try:
                res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True)
                clients = json.loads(res.stdout)
                for c in clients:
                    c_title = c.get("title", "").lower()
                    c_class = c.get("class", "").lower()
                    if query in c_title or query in c_class:
                        at = c.get("at", [0, 0])
                        size = c.get("size", [800, 600])
                        result = self.capture_region(
                            x=at[0],
                            y=at[1],
                            width=size[0],
                            height=size[1],
                            output_path=output_path,
                            quality=quality,
                        )
                        result["window_title"] = c.get("title")
                        result["window_class"] = c.get("class")
                        return result
            except Exception as e:
                raise RuntimeError(f"Failed to query window geometry: {e}")

        raise ValueError(f"No active window found matching '{query}'.")


capture_engine = ScreenCaptureEngine()


@registry.register(description="Takes a high-speed screenshot of the user's desktop screen.")
def take_screenshot(include_cursor: bool = False) -> Dict[str, Any]:
    return capture_engine.capture_full_screen(include_cursor=include_cursor)


@registry.register(description="Captures a screenshot of a specific application window by title or name.")
def capture_window_screenshot(window_name: str) -> Dict[str, Any]:
    return capture_engine.capture_window(query=window_name)
