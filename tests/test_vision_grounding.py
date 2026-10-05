from tools.base import registry
from vision.grounding import VisualGroundingEngine


def test_vision_grounding_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "locate_element" in names
    assert "inspect_screen" in names


def test_parse_coordinate_response():
    engine = VisualGroundingEngine()

    raw_json = '{"found": true, "x": 450, "y": 320, "confidence": 0.95, "description": "Search button"}'
    parsed = engine._parse_coordinate_response(raw_json, 1920, 1080)
    assert parsed["found"] is True
    assert parsed["x"] == 450
    assert parsed["y"] == 320

    # Test coordinate clamping to bounds
    out_of_bounds = '{"found": true, "x": 2500, "y": -50, "confidence": 0.8}'
    clamped = engine._parse_coordinate_response(out_of_bounds, 1920, 1080)
    assert clamped["x"] == 1920
    assert clamped["y"] == 0

    # Test markdown code block extraction
    markdown_wrapped = '```json\n{"found": true, "x": 100, "y": 200, "confidence": 0.9}\n```'
    extracted = engine._parse_coordinate_response(markdown_wrapped, 1920, 1080)
    assert extracted["x"] == 100
    assert extracted["y"] == 200


if __name__ == "__main__":
    test_vision_grounding_registered()
    test_parse_coordinate_response()
    print("✅ All visual grounding tests passed successfully!")
