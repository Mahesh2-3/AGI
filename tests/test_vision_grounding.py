import json
from tools.base import registry
from vision.grounding import VisualGroundingEngine, VLMHealthTracker, get_vlm_status, vlm_tracker


def test_vision_grounding_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "locate_element" in names
    assert "locate_element_by_image" in names
    assert "inspect_screen" in names
    assert "get_vlm_status" in names


def test_vlm_health_tracker():
    tracker = VLMHealthTracker()
    total = len(tracker.models)
    assert total >= 1

    # Initially all configured models should be available
    avail = tracker.get_available_models()
    assert len(avail) == total
    summary = tracker.get_availability_summary()
    assert f"{total}/{total} VLMs available" in summary

    # Mark first model as rate limited
    first_model = tracker.models[0].model_id
    tracker.mark_rate_limited(first_model, cooldown_seconds=60.0)

    # Now available count should decrease by 1
    new_avail = tracker.get_available_models()
    assert len(new_avail) == total - 1
    new_summary = tracker.get_availability_summary()
    assert f"{total - 1}/{total} VLMs available" in new_summary
    assert "cooling down" in new_summary

    # Status report should reflect accurate numbers
    report = tracker.get_status_report()
    assert report["total_vlm_count"] == total
    assert report["available_vlm_count"] == total - 1
    assert report["rate_limited_count"] == 1


def test_get_vlm_status_tool():
    res = registry.execute("get_vlm_status", {})
    assert res.success is True
    assert "summary" in res.output
    assert "total_vlm_count" in res.output
    assert "available_vlm_count" in res.output
    assert "models" in res.output
    assert isinstance(res.output["models"], list)


def test_parse_coordinate_response():
    engine = VisualGroundingEngine()

    # 1. Normalized bounding box test (user example: [0.69, 0.67, 0.92, 0.82])
    bbox_json = '{"found": true, "target": "Save button", "bbox": [0.69, 0.67, 0.92, 0.82], "confidence": 0.95}'
    parsed_bbox = engine._parse_coordinate_response(bbox_json, 1920, 1080)
    assert parsed_bbox["found"] is True
    # Center x: round(((0.69 + 0.92)/2) * 1920) = round(0.805 * 1920) = 1546
    # Center y: round(((0.67 + 0.82)/2) * 1080) = round(0.745 * 1080) = 805
    assert parsed_bbox["x"] == 1546
    assert parsed_bbox["y"] == 805
    assert parsed_bbox["bbox"] == [0.69, 0.67, 0.92, 0.82]
    assert parsed_bbox["bbox_pixels"] == [1325, 724, 1766, 886]

    # 2. Backward compatibility direct x and y
    raw_json = '{"found": true, "x": 450, "y": 320, "confidence": 0.95, "description": "Search button"}'
    parsed = engine._parse_coordinate_response(raw_json, 1920, 1080)
    assert parsed["found"] is True
    assert parsed["x"] == 450
    assert parsed["y"] == 320

    # 3. Coordinate clamping to bounds
    out_of_bounds = '{"found": true, "x": 2500, "y": -50, "confidence": 0.8}'
    clamped = engine._parse_coordinate_response(out_of_bounds, 1920, 1080)
    assert clamped["x"] == 1920
    assert clamped["y"] == 0

    # 4. Markdown code block extraction
    markdown_wrapped = '```json\n{"found": true, "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9}\n```'
    extracted = engine._parse_coordinate_response(markdown_wrapped, 1920, 1080)
    assert extracted["x"] == round(0.2 * 1920)
    assert extracted["y"] == round(0.3 * 1080)
    assert extracted["bbox"] == [0.1, 0.2, 0.3, 0.4]


def test_inspect_screen_with_user_prompt():
    engine = VisualGroundingEngine()
    engine.set_current_user_prompt("what are you seeing on the screen")
    assert engine.get_current_user_prompt() == "what are you seeing on the screen"

    # Verify tool schema in registry
    inspect_tool = registry.get_tool("inspect_screen")
    assert inspect_tool is not None
    props = inspect_tool.schema["function"]["parameters"]["properties"]
    assert "user_prompt" in props
    assert "focus" in props
    assert "query" in props
    assert "kwargs" not in props  # ensure **kwargs was not exposed in schema

    # Mock _prepare_image and _call_vlm to verify prompt construction
    captured_prompts = []

    def mock_prepare_image(path):
        return b"fake_jpeg", 1920, 1080

    def mock_call_vlm(model_info, prompt, img_bytes, max_tokens=500):
        captured_prompts.append(prompt)
        return "I see VS Code and a Google Chrome browser open with ChatGPT."

    engine._prepare_image = mock_prepare_image
    engine._call_vlm = mock_call_vlm

    res = engine.inspect_screen(screenshot_path="dummy.png", user_prompt="check ChatGPT input field")
    assert res["description"] == "I see VS Code and a Google Chrome browser open with ChatGPT."
    assert res["user_intent"] == "check ChatGPT input field"
    assert len(captured_prompts) == 1
    assert "Describe what is currently visible" in captured_prompts[0]
    assert "USER REQUEST / INTENT: \"check ChatGPT input field\"" in captured_prompts[0]
    assert "Specifically analyze and answer what the user is looking for" in captured_prompts[0]


def test_ollama_qwen_vlm_registered():
    tracker = VLMHealthTracker()
    model_ids = [m.model_id for m in tracker.models]
    assert "ollama:qwen2.5vl:3b" in model_ids
    # Groq should not be in tracker models when using ollama vision provider
    assert not any(m.provider == "groq" for m in tracker.models)

    first_model = tracker.models[0]
    assert first_model.provider == "ollama"
    assert first_model.model_name == "qwen2.5vl:3b"


def test_parse_coordinate_response_with_bounded_point():
    engine = VisualGroundingEngine()

    # Bounded point inside pixel bbox
    raw_json = json.dumps({
        "found": True,
        "target": "Submit button",
        "bbox": [500, 300, 700, 350],
        "point": [600, 325],
        "confidence": 0.98,
        "description": "Submit button at center"
    })
    parsed = engine._parse_coordinate_response(raw_json, 1920, 1080)
    assert parsed["found"] is True
    assert parsed["x"] == 600
    assert parsed["y"] == 325
    assert parsed["point"] == [600, 325]
    assert parsed["bbox_pixels"] == [500, 300, 700, 350]

    # Bounded point with edge point falls back safely to center
    edge_json = json.dumps({
        "found": True,
        "target": "Input field",
        "bbox": [100, 200, 300, 250],
        "point": [300, 250],  # on edge
        "confidence": 0.95,
    })
    parsed_edge = engine._parse_coordinate_response(edge_json, 1920, 1080)
    assert parsed_edge["found"] is True
    assert parsed_edge["x"] == 200  # center x
    assert parsed_edge["y"] == 225  # center y

    # Normalized point only
    pt_only_json = json.dumps({
        "found": True,
        "target": "Icon",
        "point": [0.5, 0.5],
        "confidence": 0.9,
    })
    parsed_pt = engine._parse_coordinate_response(pt_only_json, 1920, 1080)
    assert parsed_pt["found"] is True
    assert parsed_pt["x"] == 960
    assert parsed_pt["y"] == 540


if __name__ == "__main__":
    test_vision_grounding_registered()
    test_vlm_health_tracker()
    test_get_vlm_status_tool()
    test_parse_coordinate_response()
    test_inspect_screen_with_user_prompt()
    test_ollama_qwen_vlm_registered()
    test_parse_coordinate_response_with_bounded_point()
    print("✅ All visual grounding and VLM tracker tests passed successfully!")
