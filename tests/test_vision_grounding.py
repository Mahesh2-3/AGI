from tools.base import registry
from vision.grounding import VisualGroundingEngine, VLMHealthTracker, get_vlm_status, vlm_tracker


def test_vision_grounding_registered():
    schemas = registry.get_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert "locate_element" in names
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


if __name__ == "__main__":
    test_vision_grounding_registered()
    test_vlm_health_tracker()
    test_get_vlm_status_tool()
    test_parse_coordinate_response()
    test_inspect_screen_with_user_prompt()
    print("✅ All visual grounding and VLM tracker tests passed successfully!")
