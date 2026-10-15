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
    agent._generate_workflow_via_llm = MagicMock(return_value=None)

    reply = agent.step("Jarvis, what is your current status?")

    assert "All systems are currently running" in reply
    assert "⏱️ *Operation completed in" not in reply
    assert agent.last_step_duration > 0
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
    assert "open browser in new workspace" in arrow_plan
    assert "type chess.com" in arrow_plan
    assert "scan the website" in arrow_plan
    assert "interact with the DOM" in arrow_plan
    assert "->" in arrow_plan

    # Repository / commits workflow (user example)
    repo_plan = agent.generate_workflow_plan(
        "go to the repository commits and analyze the commits and tell me about the project"
    )
    assert "see the screen" in repo_plan
    assert "if browser has github tab that contains a repo" in repo_plan
    assert "go to the commits" in repo_plan
    assert "analyze commits using DOM" in repo_plan
    assert "else tell the user there is no github tab is opened" in repo_plan
    assert "->" in repo_plan

    # Browser semantic detection
    web_plan = agent.generate_workflow_plan("open chess.com and play a game")
    assert "browser" in web_plan
    assert "->" in web_plan

    # Perception query detection
    see_plan = agent.generate_workflow_plan("Sir > tell me what are you seeing on the screen")
    assert "screen" in see_plan or "screenshot" in see_plan
    assert "->" in see_plan

    # Existing window directive detection
    existing_win_plan = agent.generate_workflow_plan(
        "the web browser that is currently opened, type 'hello', don't open new web session"
    )
    assert "if target window is open" in existing_win_plan or "window" in existing_win_plan
    assert "->" in existing_win_plan

    # Test callback is triggered in step and sets grounding_engine user prompt
    from vision.grounding import grounding_engine
    planned_events = []
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Plan acknowledged."))]
    mock_llm.chat.return_value = mock_resp

    agent_with_callback = JarvisAgent(
        llm_client=mock_llm,
        tool_registry=ToolRegistry(),
        on_workflow_plan=lambda plan: planned_events.append(plan),
        enable_fast_path=False,
    )

    test_input = "tell me what are you seeing on the screen"
    agent_with_callback.step(test_input)
    assert len(planned_events) == 1
    assert "screen" in planned_events[0]
    assert "->" in planned_events[0]
    assert grounding_engine.get_current_user_prompt() == test_input


if __name__ == "__main__":
    test_tool_registry()
    test_react_loop_with_tool_call()
    test_tool_execution_failure_detection()
    test_react_loop_operation_failure_summary()
    test_agent_workflow_plan_generation()
    print("✅ All ReAct agent tests passed successfully!")
