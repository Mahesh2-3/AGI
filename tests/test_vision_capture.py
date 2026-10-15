from pathlib import Path
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


def test_capture_saves_to_persistent_screenshots_dir():
    assert capture_engine.screenshots_dir == capture_engine.persistent_dir
    res = capture_engine.capture_full_screen(quality=60)
    saved_path = res["path"]
    assert "screenshots" in saved_path
    from pathlib import Path
    assert Path(saved_path).exists()


def test_capture_sanitizes_workspace_path():
    # LLM models often hallucinate paths like /workspace/screenshots/...
    res = capture_engine.capture_full_screen(output_path="/workspace/screenshots/vscode_editor.png", quality=60)
    saved_path = res["path"]
    assert not saved_path.startswith("/workspace")
    assert "screenshots" in saved_path
    from pathlib import Path
    assert Path(saved_path).exists()
    assert Path(saved_path).name == "vscode_editor.png"


def test_capture_temporary_storage():
    res = capture_engine.capture_full_screen(persistent=False, quality=60)
    saved_path = Path(res["path"])
    assert saved_path.exists()
    assert saved_path.parent == capture_engine.temp_dir


def test_cleanup_old_ram_frames(tmp_path):
    # Set temp_dir to a clean tmp_path for test
    orig_temp = capture_engine.temp_dir
    capture_engine.temp_dir = tmp_path
    try:
        # Create 10 dummy files
        for i in range(10):
            p = tmp_path / f"frame_{i:02d}.jpg"
            p.write_bytes(b"dummy")
        assert len(list(tmp_path.iterdir())) == 10

        # Run cleanup with max_frames=4
        capture_engine._cleanup_old_ram_frames(max_frames=4)
        assert len(list(tmp_path.iterdir())) == 4
    finally:
        capture_engine.temp_dir = orig_temp


if __name__ == "__main__":
    test_vision_capture_registered()
    test_display_geometry()
    test_capture_saves_to_persistent_screenshots_dir()
    test_capture_sanitizes_workspace_path()
    test_capture_temporary_storage()
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as tmp:
        test_capture_full_screen(Path(tmp))
        test_cleanup_old_ram_frames(Path(tmp))
    print("✅ All vision capture tests passed successfully!")

