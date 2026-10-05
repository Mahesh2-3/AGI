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


if __name__ == "__main__":
    test_visual_diff_tool_registered()
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_visual_diff_detection(Path(tmp))
    print("✅ All visual diff tests passed successfully!")
