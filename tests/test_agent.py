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


if __name__ == "__main__":
    test_tool_registry()
    test_react_loop_with_tool_call()
    print("✅ All ReAct agent tests passed successfully!")
