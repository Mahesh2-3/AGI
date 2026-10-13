import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from core.agent import JarvisAgent
from core.llm_client import LLMClient
from tools.base import ToolRegistry, ToolResult
from vision.grounding import VisualGroundingEngine


class FakeChatCompletionMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning = None

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeChatResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


def test_nested_arguments_unwrapping_in_agent():
    registry = ToolRegistry()
    received_args = {}

    @registry.register(description="Click element")
    def click_element(element_description: str) -> dict:
        received_args["element_description"] = element_description
        return {"clicked": True, "target": element_description}

    # Simulate Qwen/Ollama wrapping arguments inside nested dictionary
    nested_json = json.dumps({
        "function": "click_element",
        "arguments": {
            "element_description": "Today's Problem"
        }
    })
    tc = FakeToolCall("call_1", "click_element", nested_json)

    resp1 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(content="Clicked Today's Problem, Sir."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_configured = False
    mock_llm.chat.side_effect = [resp1, resp2]

    callback_args = {}
    def on_tool_call(fn_name, args):
        callback_args.update(args)

    agent = JarvisAgent(
        llm_client=mock_llm,
        tool_registry=registry,
        enable_fast_path=False,
        on_tool_call=on_tool_call,
    )
    reply = agent.step("click today's problem")

    # Assert tool executed with clean argument
    assert received_args == {"element_description": "Today's Problem"}
    assert callback_args == {"element_description": "Today's Problem"}
    assert "Clicked Today's Problem" in reply


def test_synthetic_tool_call_raw_json():
    registry = ToolRegistry()
    clicked_targets = []

    @registry.register(description="Click element")
    def click_element(element_description: str) -> dict:
        clicked_targets.append(element_description)
        return {"clicked": True, "target": element_description}

    # Model emits raw JSON in content with tool_calls=[]
    raw_json_content = json.dumps({
        "name": "click_element",
        "arguments": {
            "element_description": "Daily Challenge Link"
        }
    })

    resp1 = FakeChatResponse(FakeChatCompletionMessage(content=raw_json_content, tool_calls=[]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(content="Daily Challenge opened successfully, Sir."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_configured = False
    mock_llm.chat.side_effect = [resp1, resp2]

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("open the daily challenge")

    assert clicked_targets == ["Daily Challenge Link"]
    assert "Daily Challenge opened" in reply


def test_synthetic_tool_call_markdown_code_block():
    registry = ToolRegistry()
    focused_windows = []

    @registry.register(description="Focus window")
    def focus_window(query: str) -> dict:
        focused_windows.append(query)
        return {"focused": True, "window": query}

    markdown_content = (
        "I will focus the LeetCode browser window now.\n"
        "```json\n"
        "{\n"
        '  "function": "focus_window",\n'
        '  "arguments": {\n'
        '    "query": "leetcode"\n'
        "  }\n"
        "}\n"
        "```"
    )

    resp1 = FakeChatResponse(FakeChatCompletionMessage(content=markdown_content, tool_calls=[]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(content="LeetCode window brought to focus."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_configured = False
    mock_llm.chat.side_effect = [resp1, resp2]

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("switch to leetcode")

    assert focused_windows == ["leetcode"]
    assert "LeetCode window brought to focus" in reply


def test_locate_by_template_rejects_full_screenshot():
    engine = VisualGroundingEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a full-screen image (1920x1080)
        full_shot_path = Path(tmpdir) / "screen_full.jpg"
        img = Image.new("RGB", (1920, 1080), color=(100, 100, 100))
        img.save(full_shot_path)

        res = engine.locate_by_template(template_path=full_shot_path)
        assert res["found"] is False
        assert "full-screen screenshot" in res["error"]
        assert "click_element" in res["error"]
        assert "locate_element" in res["error"]


def test_locate_by_template_allows_small_template():
    engine = VisualGroundingEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small icon image (32x32)
        icon_path = Path(tmpdir) / "small_icon.png"
        icon_img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        icon_img.save(icon_path)

        # Create a screen image (1920x1080)
        screen_path = Path(tmpdir) / "screen.png"
        screen_img = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        # Paste icon at (100, 100)
        screen_img.paste(icon_img, (100, 100))
        screen_img.save(screen_path)

        res = engine.locate_by_template(template_path=icon_path, screenshot_path=screen_path)
        assert res["found"] is True
        assert res["x"] == 116  # 100 + 16
        assert res["y"] == 116  # 100 + 16
