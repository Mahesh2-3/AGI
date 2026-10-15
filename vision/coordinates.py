"""Standardized visual coordinate transformation and validation system.

Guarantees consistent mapping between original screen dimensions and resized
VLM inference dimensions, preventing mis-clicks and coordinate drift.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("JARVIS.Coordinates")


@dataclass
class CoordinateTransformer:
    """Manages coordinate conversions between screen pixels, resized model dimensions, and normalized space."""

    orig_w: int
    orig_h: int
    model_w: int
    model_h: int

    def __post_init__(self):
        self.orig_w = max(1, int(self.orig_w))
        self.orig_h = max(1, int(self.orig_h))
        self.model_w = max(1, int(self.model_w))
        self.model_h = max(1, int(self.model_h))
        self.scale_x = self.orig_w / self.model_w
        self.scale_y = self.orig_h / self.model_h

    def to_screen_pixels(self, x: Union[int, float], y: Union[int, float]) -> Tuple[int, int]:
        """Converts coordinates (normalized or pixel) into clamped original screen pixel coordinates."""
        try:
            fx, fy = float(x), float(y)
        except (ValueError, TypeError):
            return 0, 0

        # Case 1: Normalized coordinates [0.0, 1.0]
        if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
            px = int(round(fx * self.orig_w))
            py = int(round(fy * self.orig_h))
        # Case 2: Pixels in model's resized coordinate space
        elif (self.model_w != self.orig_w or self.model_h != self.orig_h) and fx <= self.model_w and fy <= self.model_h:
            px = int(round(fx * self.scale_x))
            py = int(round(fy * self.scale_y))
        # Case 3: Pixels in original screen space
        else:
            px = int(round(fx))
            py = int(round(fy))

        # Strict clamping to screen boundaries
        clamped_x = max(0, min(self.orig_w, px))
        clamped_y = max(0, min(self.orig_h, py))
        return clamped_x, clamped_y

    def to_normalized(self, x: Union[int, float], y: Union[int, float]) -> Tuple[float, float]:
        """Converts coordinates to normalized [0.0, 1.0] space relative to original screen."""
        px, py = self.to_screen_pixels(x, y)
        nx = round(px / self.orig_w, 4)
        ny = round(py / self.orig_h, 4)
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def transform_bbox(self, raw_box: Union[List[Any], Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
        """Parses, scales, orders, and clamps a bounding box into normalized and pixel representations."""
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
            return None

        try:
            coords = [float(v) for v in raw_box[:4]]
        except (ValueError, TypeError):
            return None

        # Check if coordinates are normalized or pixels
        is_normalized = max(coords) <= 1.0 and min(coords) >= 0.0

        if is_normalized:
            nx1, ny1, nx2, ny2 = coords
            px1 = int(round(nx1 * self.orig_w))
            py1 = int(round(ny1 * self.orig_h))
            px2 = int(round(nx2 * self.orig_w))
            py2 = int(round(ny2 * self.orig_h))
        else:
            # Check if model space vs screen space
            if (self.model_w != self.orig_w or self.model_h != self.orig_h) and max(coords[0], coords[2]) <= self.model_w and max(coords[1], coords[3]) <= self.model_h:
                px1 = int(round(coords[0] * self.scale_x))
                py1 = int(round(coords[1] * self.scale_y))
                px2 = int(round(coords[2] * self.scale_x))
                py2 = int(round(coords[3] * self.scale_y))
            else:
                px1 = int(round(coords[0]))
                py1 = int(round(coords[1]))
                px2 = int(round(coords[2]))
                py2 = int(round(coords[3]))

        # Ensure correct ordering min <= max
        xmin, xmax = min(px1, px2), max(px1, px2)
        ymin, ymax = min(py1, py2), max(py1, py2)

        # Clamp to screen boundary
        xmin = max(0, min(self.orig_w, xmin))
        xmax = max(0, min(self.orig_w, xmax))
        ymin = max(0, min(self.orig_h, ymin))
        ymax = max(0, min(self.orig_h, ymax))

        # Normalized coordinates: preserve exact normalized values if provided directly
        if is_normalized:
            norm_x1 = round(min(coords[0], coords[2]), 4)
            norm_y1 = round(min(coords[1], coords[3]), 4)
            norm_x2 = round(max(coords[0], coords[2]), 4)
            norm_y2 = round(max(coords[1], coords[3]), 4)
        else:
            norm_x1 = round(xmin / self.orig_w, 4)
            norm_y1 = round(ymin / self.orig_h, 4)
            norm_x2 = round(xmax / self.orig_w, 4)
            norm_y2 = round(ymax / self.orig_h, 4)

        center_x = int(round((xmin + xmax) / 2))
        center_y = int(round((ymin + ymax) / 2))

        return {
            "bbox": [norm_x1, norm_y1, norm_x2, norm_y2],
            "bbox_pixels": [xmin, ymin, xmax, ymax],
            "center": [center_x, center_y],
            "width": xmax - xmin,
            "height": ymax - ymin,
        }

    def is_point_inside_bbox(self, point: Tuple[int, int], bbox_pixels: List[int], strict_interior: bool = True) -> bool:
        """Verifies if a screen pixel coordinate lies within an element's bounding box interior."""
        if not bbox_pixels or len(bbox_pixels) < 4:
            return False
        x, y = point
        x1, y1, x2, y2 = bbox_pixels[:4]
        if strict_interior and (x2 - x1 > 2) and (y2 - y1 > 2):
            return x1 < x < x2 and y1 < y < y2
        return x1 <= x <= x2 and y1 <= y <= y2

    def resolve_click_point(
        self,
        raw_point: Optional[Union[List[Any], Tuple[Any, ...]]],
        bbox_pixels: Optional[List[int]],
    ) -> Optional[Tuple[int, int]]:
        """Resolves the safest click point: uses bounded point if inside bbox, otherwise box center."""
        candidate_point = None
        if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
            try:
                candidate_point = self.to_screen_pixels(raw_point[0], raw_point[1])
            except (ValueError, TypeError):
                candidate_point = None

        if bbox_pixels and len(bbox_pixels) >= 4:
            box_center = (
                int(round((bbox_pixels[0] + bbox_pixels[2]) / 2)),
                int(round((bbox_pixels[1] + bbox_pixels[3]) / 2)),
            )
            if candidate_point and self.is_point_inside_bbox(candidate_point, bbox_pixels):
                return candidate_point
            return box_center

        return candidate_point
