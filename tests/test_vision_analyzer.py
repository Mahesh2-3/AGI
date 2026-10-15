from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import io
import pytest
from PIL import Image

from vision.analyzer import analyze_image, _encode_image, _parse_structured_response


def test_analyze_image_missing_file():
    res = analyze_image("/path/to/nonexistent_image_12345.png", prompt="Find button")
    assert res["success"] is False
    assert "not exist" in res["error"]
    assert res["found"] is False


def test_analyze_image_invalid_file(tmp_path):
    bad_file = tmp_path / "corrupted.png"
    bad_file.write_text("not an image")
    res = analyze_image(bad_file, prompt="Describe screen")
    assert res["success"] is False
    assert "Invalid" in res["error"] or "corrupted" in res["error"]


def test_parse_structured_response_normalized_bbox():
    raw_json = json.dumps({
        "found": True,
        "target": "Submit button",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "confidence": 0.95,
        "explanation": "Green submit button visible in bottom right",
    })
    res = _parse_structured_response(raw_json, orig_w=1920, orig_h=1080, target_query="Submit button")
    assert res["success"] is True
    assert res["found"] is True
    assert res["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert res["bbox_pixels"] == [192, 216, 576, 432]
    assert res["point"] == [384, 324]
    assert res["confidence"] == 0.95
    assert "Green submit button" in res["explanation"]


def test_parse_structured_response_not_found():
    raw_json = json.dumps({
        "found": False,
        "target": "Bluetooth settings",
        "bbox": None,
        "explanation": "Bluetooth settings not open on screen",
    })
    res = _parse_structured_response(raw_json, orig_w=1920, orig_h=1080, target_query="Bluetooth")
    assert res["success"] is True
    assert res["found"] is False
    assert res["bbox"] is None
    assert res["point"] is None


def test_analyze_image_mock_ollama_call(tmp_path):
    test_img = tmp_path / "screen.png"
    img = Image.new("RGB", (640, 480), color="blue")
    img.save(test_img)

    mock_resp_data = json.dumps({
        "message": {
            "content": json.dumps({
                "found": True,
                "target": "Terminal window",
                "bbox": [0.1, 0.1, 0.5, 0.5],
                "confidence": 0.92,
                "explanation": "Active kitty terminal",
            })
        }
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_resp_data
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = analyze_image(test_img, prompt="Find the terminal window", structured=True)
        assert res["success"] is True
        assert res["found"] is True
        assert res["point"] == [192, 144]
        assert res["bbox_pixels"] == [64, 48, 320, 240]


def test_parse_structured_response_point_only():
    # Model returns valid point but omits bbox
    raw_json = json.dumps({
        "found": True,
        "target": "Search icon",
        "point": [0.5, 0.5],
        "explanation": "Search magnifying glass at center",
    })
    res = _parse_structured_response(raw_json, orig_w=1920, orig_h=1080, target_query="Search icon")
    assert res["success"] is True
    assert res["found"] is True
    assert res["point"] == [960, 540]
    assert res["bbox_pixels"] == [960, 540, 960, 540]


def test_parse_structured_response_missing_confidence():
    # Confidence omitted by model should not be assumed as 0.95
    raw_json = json.dumps({
        "found": False,
        "target": "Unknown element",
    })
    res = _parse_structured_response(raw_json, orig_w=1920, orig_h=1080)
    assert res["confidence"] == 0.0

