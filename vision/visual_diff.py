import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageChops, ImageStat

from tools.base import registry
from vision.capture import capture_engine


class VisualDiffEngine:
    """Detects visual differences and UI updates between desktop screen frames."""

    @staticmethod
    def compare_images(
        image_a_path: str | Path,
        image_b_path: str | Path,
        pixel_threshold: int = 15,
    ) -> Dict[str, Any]:
        """Compares two images and calculates the percentage of changed pixels and bounding box."""
        path_a = Path(image_a_path)
        path_b = Path(image_b_path)

        if not path_a.exists():
            raise FileNotFoundError(f"Base frame not found: {path_a}")
        if not path_b.exists():
            raise FileNotFoundError(f"Target frame not found: {path_b}")

        with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
            # Normalize sizes if differing
            if img_a.size != img_b.size:
                img_b = img_b.resize(img_a.size)

            img_a_rgb = img_a.convert("RGB")
            img_b_rgb = img_b.convert("RGB")

            diff = ImageChops.difference(img_a_rgb, img_b_rgb)
            bbox = diff.getbbox()

            # Calculate Root Mean Square (RMS) difference
            stat = ImageStat.Stat(diff)
            rms = math.sqrt(sum(v ** 2 for v in stat.rms) / len(stat.rms))

            # Bounding box of change
            changed = bbox is not None and rms > 1.0

            change_info = {
                "changed": changed,
                "rms_difference": round(rms, 2),
                "bounding_box": {
                    "left": bbox[0],
                    "top": bbox[1],
                    "right": bbox[2],
                    "bottom": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                } if bbox else None,
                "frame_size": {
                    "width": img_a.width,
                    "height": img_a.height,
                },
            }

            if changed and bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area_ratio = round((w * h) / (img_a.width * img_a.height) * 100, 2)
                change_info["changed_area_percent"] = area_ratio
                change_info["summary"] = f"Visual update observed at region ({bbox[0]}, {bbox[1]}) spanning {w}x{h}px ({area_ratio}% of screen)."
            else:
                change_info["changed_area_percent"] = 0.0
                change_info["summary"] = "No significant visual change detected between frames."

            return change_info

    def wait_for_ui_update(
        self,
        baseline_shot_path: str | Path,
        timeout_seconds: float = 3.0,
        poll_interval: float = 0.25,
    ) -> Dict[str, Any]:
        """Polls screen until a visual change occurs or timeout expires."""
        start_t = time.perf_counter()

        while (time.perf_counter() - start_t) < timeout_seconds:
            time.sleep(poll_interval)
            current_shot = capture_engine.capture_full_screen()
            diff_res = self.compare_images(baseline_shot_path, current_shot["path"])
            if diff_res["changed"]:
                diff_res["elapsed_seconds"] = round(time.perf_counter() - start_t, 2)
                diff_res["final_screenshot_path"] = current_shot["path"]
                return diff_res

        return {
            "changed": False,
            "elapsed_seconds": round(time.perf_counter() - start_t, 2),
            "summary": f"No visual change detected within {timeout_seconds} seconds.",
        }


diff_engine = VisualDiffEngine()


@registry.register(description="Compares two screenshot images to detect visual UI changes, difference score, and bounding region.")
def check_visual_difference(before_image_path: str, after_image_path: str) -> Dict[str, Any]:
    return diff_engine.compare_images(before_image_path, after_image_path)
