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
        """Compares two images and calculates the percentage of changed pixels and bounding box.
        
        Applies pixel_threshold to filter out minor noise/compression artifacts.
        """
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

            # Apply pixel_threshold across channels to isolate meaningful pixel changes
            r, g, b = diff.split()
            mask_r = r.point(lambda p: 255 if p >= pixel_threshold else 0)
            mask_g = g.point(lambda p: 255 if p >= pixel_threshold else 0)
            mask_b = b.point(lambda p: 255 if p >= pixel_threshold else 0)
            mask = ImageChops.lighter(ImageChops.lighter(mask_r, mask_g), mask_b)

            # Calculate actual changed pixel count from thresholded mask
            hist = mask.histogram()
            changed_pixels = sum(hist[1:])
            total_pixels = img_a.width * img_a.height
            changed_pixel_percent = round((changed_pixels / total_pixels) * 100, 2)

            # Bounding box of thresholded change
            bbox = mask.getbbox()

            # Calculate Root Mean Square (RMS) difference across entire image
            stat = ImageStat.Stat(diff)
            rms = math.sqrt(sum(v ** 2 for v in stat.rms) / len(stat.rms))

            changed = bbox is not None and changed_pixels > 0

            change_info: Dict[str, Any] = {
                "changed": changed,
                "rms_difference": round(rms, 2),
                "pixel_threshold": pixel_threshold,
                "changed_pixel_count": changed_pixels,
                "changed_pixel_percent": changed_pixel_percent,
                "changed_area_percent": changed_pixel_percent,
                "bounding_box": {
                    "left": bbox[0],
                    "top": bbox[1],
                    "right": bbox[2],
                    "bottom": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                } if (changed and bbox) else None,
                "frame_size": {
                    "width": img_a.width,
                    "height": img_a.height,
                },
            }

            if changed and bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                bbox_area_ratio = round((w * h) / total_pixels * 100, 2)
                change_info["bounding_box_area_percent"] = bbox_area_ratio
                change_info["summary"] = (
                    f"Visual update observed: {changed_pixels} changed pixels ({changed_pixel_percent}% of screen) "
                    f"within region ({bbox[0]}, {bbox[1]}) spanning {w}x{h}px."
                )
            else:
                change_info["bounding_box_area_percent"] = 0.0
                change_info["summary"] = "No significant visual change detected between frames."

            return change_info

    def wait_for_ui_update(
        self,
        baseline_shot_path: str | Path,
        timeout_seconds: float = 3.0,
        poll_interval: float = 0.25,
        pixel_threshold: int = 15,
    ) -> Dict[str, Any]:
        """Polls screen using temporary frames until a visual change occurs or timeout expires."""
        start_t = time.perf_counter()
        baseline_p = Path(baseline_shot_path)
        if not baseline_p.exists():
            raise FileNotFoundError(f"Baseline frame not found: {baseline_p}")

        while (time.perf_counter() - start_t) < timeout_seconds:
            time.sleep(poll_interval)
            try:
                # Capture frame to temporary storage to avoid polluting disk
                current_shot = capture_engine.capture_full_screen(persistent=False)
                if not current_shot or not current_shot.get("path") or not Path(current_shot["path"]).exists():
                    continue

                diff_res = self.compare_images(baseline_p, current_shot["path"], pixel_threshold=pixel_threshold)
                if diff_res["changed"]:
                    diff_res["elapsed_seconds"] = round(time.perf_counter() - start_t, 2)
                    diff_res["final_screenshot_path"] = current_shot["path"]
                    return diff_res
            except Exception:
                continue

        return {
            "changed": False,
            "elapsed_seconds": round(time.perf_counter() - start_t, 2),
            "summary": f"No visual change detected within {timeout_seconds} seconds.",
        }


diff_engine = VisualDiffEngine()


@registry.register(description="Compares two screenshot images to detect visual UI changes, difference score, and bounding region.")
def check_visual_difference(before_image_path: str, after_image_path: str, pixel_threshold: int = 15) -> Dict[str, Any]:
    return diff_engine.compare_images(before_image_path, after_image_path, pixel_threshold=pixel_threshold)
