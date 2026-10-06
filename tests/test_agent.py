from unittest.mock import MagicMock
from core.agent import JarvisAgent
from core.llm_client import LLMClient
from tools.base import ToolRegistry, ToolResult


def test_tool_registry():
    registry = ToolRegistry()

    @registry.register(description="Add two numbers.")
    def add(a: int, b: int) -> int:
        return a + b

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "add"
    assert "a" in schemas[0]["function"]["parameters"]["properties"]
    assert "b" in schemas[0]["function"]["parameters"]["properties"]

    result = registry.execute("add", {"a": 10, "b": 25})
    assert result.success is True
    assert result.output == 35


def test_react_loop_with_tool_call():
    registry = ToolRegistry()

    @registry.register(description="Get current status of Jarvis.")
    def get_status() -> str:
        return "All systems operational"

    mock_llm = MagicMock(spec=LLMClient)

    # First turn: LLM decides to call tool 'get_status'
    tool_call_obj = MagicMock()
    tool_call_obj.id = "call_abc123"
    tool_call_obj.function.name = "get_status"
    tool_call_obj.function.arguments = "{}"

    first_message = MagicMock()
    first_message.tool_calls = [tool_call_obj]
    first_message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_status", "arguments": "{}"}
            }
        ]
    }

    first_response = MagicMock()
    first_response.choices = [MagicMock(message=first_message)]

    # Second turn: LLM receives tool output and returns final answer
    second_message = MagicMock()
    second_message.tool_calls = None
    second_message.content = "All systems are currently running at peak efficiency, Sir."

    second_response = MagicMock()
    second_response.choices = [MagicMock(message=second_message)]

    mock_llm.chat.side_effect = [first_response, second_response]

    called_tools = []
    agent = JarvisAgent(
        llm_client=mock_llm,
        tool_registry=registry,
        on_tool_call=lambda name, args: called_tools.append((name, args)),
    )

    reply = agent.step("Jarvis, what is your current status?")

    assert "All systems are currently running" in reply
    assert len(called_tools) == 1
    assert called_tools[0][0] == "get_status"
    assert len(agent.messages) >= 4  # system, user, assistant(tool_call), tool, assistant(final)


def test_tool_execution_failure_detection():
    registry = ToolRegistry()

    @registry.register(description="Failing click operation")
    def fail_click() -> dict:
        return {"success": False, "error": "Visual grounding could not locate element"}

    result = registry.execute("fail_click", {})
    assert result.success is False
    assert "could not locate element" in result.error
    content = result.to_message_content()
    assert "⚠️ OPERATION FAILED" in content
    assert "SYSTEM DIRECTIVE" in content


def test_react_loop_operation_failure_summary():
    registry = ToolRegistry()

    @registry.register(description="Failing click tool")
    def click_btn() -> dict:
        return {"success": False, "error": "No visual update detected after click"}

    mock_llm = MagicMock(spec=LLMClient)

    tool_call_obj = MagicMock()
    tool_call_obj.id = "call_fail123"
    tool_call_obj.function.name = "click_btn"
    tool_call_obj.function.arguments = "{}"

    msg = MagicMock()
    msg.tool_calls = [tool_call_obj]
    msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_fail123", "type": "function", "function": {"name": "click_btn", "arguments": "{}"}}]
    }

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=msg)]

    # Loop always calls click_btn until max_iterations=1
    mock_llm.chat.return_value = mock_resp

    agent = JarvisAgent(
        llm_client=mock_llm,
        tool_registry=registry,
        max_iterations=1,
    )

    reply = agent.step("Click the button")
    assert "unable to complete the task because the following operations encountered issues" in reply
    assert "click_btn" in reply
    assert "No visual update detected after click" in reply


def test_agent_workflow_plan_generation():
    agent = JarvisAgent(llm_client=MagicMock(), tool_registry=ToolRegistry())

    # Arrow-chained user prompt
    arrow_plan = agent.generate_workflow_plan(
        "open browser in new workspace -> type chess.com -> scan the website -> interact with the DOM -> send a challenge to a friend"
    )
    assert "1. **Open browser in new workspace**" in arrow_plan
    assert "2. **Type chess.com**" in arrow_plan
    assert "3. **Scan the website**" in arrow_plan
    assert "4. **Interact with the dom**" in arrow_plan
    assert "5. **Send a challenge to a friend**" in arrow_plan

    # Browser semantic detection
    web_plan = agent.generate_workflow_plan("open chess.com and play a game")
    assert "Workspace Routing" in web_plan
    assert "High-Speed Navigation" in web_plan
    assert "DOM Scanning" in web_plan

    # Test callback is triggered in step
    planned_events = []
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Plan acknowledged."))]
    mock_llm.chat.return_value = mock_resp

    agent_with_callback = JarvisAgent(
        llm_client=mock_llm,
        tool_registry=ToolRegistry(),
        on_workflow_plan=lambda plan: planned_events.append(plan),
    )

    agent_with_callback.step("open browser -> test")
    assert len(planned_events) == 1
    assert "Open browser" in planned_events[0]


if __name__ == "__main__":
    test_tool_registry()
    test_react_loop_with_tool_call()
    test_tool_execution_failure_detection()
    test_react_loop_operation_failure_summary()
    test_agent_workflow_plan_generation()
    print("✅ All ReAct agent tests passed successfully!")
