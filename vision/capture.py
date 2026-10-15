import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from config.settings import settings
from core.workspace import workspace_manager
from tools.base import registry

logger = logging.getLogger("jarvis.vision.capture")

# High-speed in-memory RAM disk (/dev/shm) for zero disk I/O during visual perception loops
RAM_DISK_DIR = Path("/dev/shm/jarvis_screen") if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK) else None
FALLBACK_TEMP_DIR = Path("/tmp/jarvis_screen")
PERSISTENT_SCREENSHOTS_DIR = settings.WORKSPACE_DIR / "screenshots"


class ScreenCaptureEngine:
    """High-speed screen capture engine optimized for Linux Wayland & Hyprland."""

    def __init__(self):
        self.persistent_dir = PERSISTENT_SCREENSHOTS_DIR
        self.persistent_dir.mkdir(parents=True, exist_ok=True)
        # Store screenshots directly in workspace screenshots folder
        self.screenshots_dir = self.persistent_dir

        # Temporary destination for transient perception loops (RAM disk or /tmp)
        self.temp_dir = RAM_DISK_DIR if RAM_DISK_DIR is not None else FALLBACK_TEMP_DIR
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.temp_dir = FALLBACK_TEMP_DIR
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.ram_dir = self.temp_dir
        self.has_grim = bool(shutil.which("grim"))

    def _cleanup_old_ram_frames(self, max_frames: int = 20) -> None:
        """Enforces bounded retention on temporary frames to prevent memory/disk bloat."""
        try:
            if not self.temp_dir or not self.temp_dir.exists():
                return
            files = [f for f in self.temp_dir.iterdir() if f.is_file()]
            if len(files) > max_frames:
                files.sort(key=lambda f: f.stat().st_mtime)
                for old_file in files[:-max_frames]:
                    try:
                        old_file.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to cleanup old temporary frames: {e}")

    def _sanitize_output_path(
        self,
        output_path: Optional[str | Path],
        prefix: str = "screen",
        persistent: bool = True,
    ) -> Path:
        """Sanitizes user or LLM supplied output paths, avoiding root/unwritable locations like /workspace."""
        base_dir = self.persistent_dir if persistent else self.temp_dir

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            target = base_dir / f"{prefix}_{timestamp}.jpg"
            if not persistent:
                self._cleanup_old_ram_frames()
            return target

        p = Path(output_path)
        # Prevent permission errors from sandbox paths (e.g. /workspace/...) or non-absolute paths
        if str(p).startswith("/workspace") or not p.is_absolute():
            sanitized = base_dir / p.name
        else:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                sanitized = p.resolve()
            except (PermissionError, OSError):
                sanitized = base_dir / p.name

        try:
            sanitized.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            sanitized = base_dir / p.name
            sanitized.parent.mkdir(parents=True, exist_ok=True)

        if not persistent:
            self._cleanup_old_ram_frames()

        return sanitized

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
                        "active_workspace_id": primary.get("activeWorkspace", {}).get("id", 1),
                    }
                logger.warning("hyprctl monitors returned empty list; falling back to 1920x1080 default.")
            except Exception as e:
                logger.warning(f"Failed to query monitor geometry via hyprctl ({e}); falling back to 1920x1080 default.")
        else:
            logger.warning("hyprctl not found on system; falling back to 1920x1080 default display geometry.")
        return {"width": 1920, "height": 1080, "scale": 1.0, "active_workspace": "1", "active_workspace_id": 1}

    def capture_full_screen(
        self,
        output_path: Optional[str | Path] = None,
        include_cursor: bool = False,
        quality: int = 80,
        persistent: bool = True,
    ) -> Dict[str, Any]:
        """Captures full desktop screenshot and saves to destination (persistent or temporary)."""
        output_path = self._sanitize_output_path(output_path, prefix="screen", persistent=persistent)
        # Ensure Jarvis's working workspace is active on the monitor before capturing
        workspace_manager.ensure_working_workspace_active()
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
        persistent: bool = False,
    ) -> Dict[str, Any]:
        """Captures a specific region of the screen (x, y, w, h)."""
        output_path = self._sanitize_output_path(output_path, prefix="region", persistent=persistent)
        # Ensure Jarvis's working workspace is active on the monitor before capturing
        workspace_manager.ensure_working_workspace_active()
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
        persistent: bool = True,
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
                        # Auto-switch to window workspace if currently on another workspace
                        c_ws = c.get("workspace", {}).get("id")
                        if c_ws is not None:
                            try:
                                active_res = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True)
                                active_id = json.loads(active_res.stdout).get("id")
                                if c_ws != active_id:
                                    subprocess.run(
                                        ["hyprctl", "dispatch", "workspace", str(c_ws)],
                                        capture_output=True,
                                        text=True,
                                    )
                                    # Verify workspace switch succeeded
                                    switched = False
                                    for _ in range(6):
                                        time.sleep(0.05)
                                        v_res = subprocess.run(["hyprctl", "activeworkspace", "-j"], capture_output=True, text=True)
                                        if json.loads(v_res.stdout).get("id") == c_ws:
                                            switched = True
                                            break
                                    if not switched:
                                        logger.warning(f"Workspace switch to {c_ws} may not have completed; re-checking active workspace.")

                                    # Re-query client geometry on the newly focused workspace
                                    updated_res = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True)
                                    for updated_c in json.loads(updated_res.stdout):
                                        if updated_c.get("address") == c.get("address"):
                                            c = updated_c
                                            break
                            except Exception as e:
                                logger.warning(f"Failed verifying workspace switch in capture_window: {e}")

                        at = c.get("at", [0, 0])
                        size = c.get("size", [800, 600])
                        result = self.capture_region(
                            x=at[0],
                            y=at[1],
                            width=size[0],
                            height=size[1],
                            output_path=output_path,
                            quality=quality,
                            persistent=persistent,
                        )
                        result["window_title"] = c.get("title")
                        result["window_class"] = c.get("class")
                        result["workspace_id"] = c_ws
                        return result
            except Exception as e:
                raise RuntimeError(f"Failed to query window geometry: {e}")

        raise ValueError(f"No active window found matching '{query}'.")


capture_engine = ScreenCaptureEngine()


@registry.register(description="Takes a high-speed screenshot of the user's desktop screen and saves it into the workspace screenshots folder.")
def take_screenshot(include_cursor: bool = False, output_path: Optional[str] = None) -> Dict[str, Any]:
    return capture_engine.capture_full_screen(output_path=output_path, include_cursor=include_cursor, persistent=True)


@registry.register(description="Captures a screenshot of a specific application window by title or name.")
def capture_window_screenshot(window_name: str) -> Dict[str, Any]:
    return capture_engine.capture_window(query=window_name, persistent=True)
