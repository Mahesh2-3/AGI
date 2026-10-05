from tools.base import registry
from vision.capture import capture_engine


def test_vision_capture_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "take_screenshot" in names
    assert "capture_window_screenshot" in names


def test_display_geometry():
    geom = capture_engine.get_display_geometry()
    assert "width" in geom
    assert "height" in geom
    assert geom["width"] > 0
    assert geom["height"] > 0


def test_capture_full_screen(tmp_path):
    dest = tmp_path / "test_screen.jpg"
    res = capture_engine.capture_full_screen(output_path=dest, quality=60)
    assert dest.exists()
    assert res["width"] > 0
    assert res["height"] > 0
    assert res["capture_latency_ms"] > 0


if __name__ == "__main__":
    test_vision_capture_registered()
    test_display_geometry()
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as tmp:
        test_capture_full_screen(Path(tmp))
    print("✅ All vision capture tests passed successfully!")
