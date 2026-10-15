from pathlib import Path
from PIL import Image, ImageDraw
from tools.base import registry
from vision.visual_diff import diff_engine


def test_visual_diff_tool_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "check_visual_difference" in names


def test_visual_diff_detection(tmp_path):
    # Create baseline image (100x100 white)
    img_a_path = tmp_path / "frame_a.jpg"
    img_a = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img_a.save(str(img_a_path))

    # Identical comparison
    same_res = diff_engine.compare_images(img_a_path, img_a_path)
    assert same_res["changed"] is False
    assert same_res["rms_difference"] == 0.0

    # Create modified image (with black rectangle in region (50, 50) to (100, 100))
    img_b_path = tmp_path / "frame_b.jpg"
    img_b = img_a.copy()
    draw = ImageDraw.Draw(img_b)
    draw.rectangle([50, 50, 100, 100], fill=(0, 0, 0))
    img_b.save(str(img_b_path))

    diff_res = diff_engine.compare_images(img_a_path, img_b_path)
    assert diff_res["changed"] is True
    assert diff_res["rms_difference"] > 0.0
    assert diff_res["bounding_box"] is not None
    # Bounding box should span approximately (50, 50, 100, 100)
    bbox = diff_res["bounding_box"]
    assert bbox["left"] <= 55
    assert bbox["top"] <= 55
    assert bbox["right"] >= 95
    assert bbox["bottom"] >= 95
    assert diff_res["changed_pixel_count"] > 0
    assert diff_res["changed_pixel_percent"] > 0


def test_visual_diff_pixel_threshold(tmp_path):
    img_a_path = tmp_path / "frame_base.jpg"
    img_b_path = tmp_path / "frame_noise.jpg"

    # Base image with neutral gray
    img_a = Image.new("RGB", (50, 50), color=(100, 100, 100))
    img_a.save(str(img_a_path))

    # Modified image with subtle variation of +5 (compression / rendering artifact)
    img_b = Image.new("RGB", (50, 50), color=(105, 105, 105))
    img_b.save(str(img_b_path))

    # With default threshold=15, subtle variation (+5) must be ignored
    res_filtered = diff_engine.compare_images(img_a_path, img_b_path, pixel_threshold=15)
    assert res_filtered["changed"] is False
    assert res_filtered["changed_pixel_count"] == 0

    # With sensitive threshold=3, variation (+5) is detected
    res_detected = diff_engine.compare_images(img_a_path, img_b_path, pixel_threshold=3)
    assert res_detected["changed"] is True
    assert res_detected["changed_pixel_count"] == 50 * 50


def test_wait_for_ui_update_timeout(tmp_path):
    img_a_path = tmp_path / "baseline.jpg"
    Image.new("RGB", (50, 50), color=(255, 255, 255)).save(str(img_a_path))
    res = diff_engine.wait_for_ui_update(img_a_path, timeout_seconds=0.1, poll_interval=0.05)
    assert "changed" in res
    assert "elapsed_seconds" in res


if __name__ == "__main__":
    test_visual_diff_tool_registered()
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_visual_diff_detection(Path(tmp))
        test_visual_diff_pixel_threshold(Path(tmp))
        test_wait_for_ui_update_timeout(Path(tmp))
    print("✅ All visual diff tests passed successfully!")

