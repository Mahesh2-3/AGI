from unittest.mock import MagicMock
from core.agent import JarvisAgent
from core.llm_client import LLMClient
from tools.base import ToolRegistry


def test_a1_no_llm_call_for_planning():
    mock_llm = MagicMock(spec=LLMClient)
    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())

    plan = agent.generate_workflow_plan("Sir, please check the system volume and mute it")
    assert "->" in plan
    # Verify no LLM call was made during planning
    assert mock_llm.chat.call_count == 0


def test_a2_fast_path_works_with_active_goal():
    registry = ToolRegistry()

    @registry.register(description="Get time")
    def get_current_time() -> str:
        return "20:00:00"

    mock_llm = MagicMock(spec=LLMClient)
    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=True)

    # Set an active goal (e.g. earlier turn was working on coding)
    agent.active_goal = "Solve LeetCode problem in browser"

    # Now user asks for time (independent deterministic command)
    reply = agent.step("what time is it?")
    assert "20:00:00" in reply
    # Verify LLM was NOT invoked, fast path handled it in <15ms
    assert mock_llm.chat.call_count == 0
    # Active goal should remain preserved for subsequent goal steps
    assert agent.active_goal == "Solve LeetCode problem in browser"


def test_a4_performance_metrics_tracking():
    registry = ToolRegistry()

    @registry.register(description="Sample status tool")
    def status_tool() -> str:
        return "OK"

    mock_llm = MagicMock(spec=LLMClient)
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = "Everything is running smoothly, Sir."
    mock_llm.chat.return_value = MagicMock(choices=[MagicMock(message=msg)])
    mock_llm.last_metrics = {"model": "qwen3:8b", "duration_s": 0.12, "token_count": 10, "ttft_s": None}

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry)
    agent.step("How are things?")

    metrics = agent.get_performance_metrics()
    assert "planning_duration_s" in metrics
    assert "total_duration_s" in metrics
    assert len(metrics["llm_requests"]) == 1
    assert metrics["llm_requests"][0]["model"] == "qwen3:8b"


def test_a5_system_prompt_caching():
    agent = JarvisAgent(llm_client=MagicMock(), tool_registry=ToolRegistry())
    p1 = agent._build_system_prompt("first prompt")
    p2 = agent._build_system_prompt("first prompt")
    assert p1 == p2
    assert agent._cached_base_prompt is not None


def test_l1_agent_token_streaming():
    mock_llm = MagicMock(spec=LLMClient)
    chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Certainly, ", reasoning=None, tool_calls=None))])
    chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Sir.", reasoning=None, tool_calls=None))])
    mock_llm.chat_stream.return_value = iter([chunk1, chunk2])
    mock_llm.last_metrics = {"model": "qwen3:8b", "duration_s": 0.05, "token_count": 2, "ttft_s": 0.01}

    streamed_tokens = []
    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    reply = agent.step("hello", on_token=lambda tok: streamed_tokens.append(tok))

    assert "".join(streamed_tokens) == "Certainly, Sir."
    assert "Certainly, Sir." in reply


def test_a3_observational_query_terminal_gate():
    registry = ToolRegistry()

    @registry.register(description="Inspect screen")
    def inspect_screen(user_prompt: str = "") -> dict:
        return {"description": "Active window: Terminal with bash"}

    mock_llm = MagicMock(spec=LLMClient)

    # First turn: model calls inspect_screen
    fn_mock = MagicMock()
    fn_mock.name = "inspect_screen"
    fn_mock.arguments = '{"user_prompt": "what do you see on the screen"}'
    call_msg = MagicMock()
    call_msg.tool_calls = [
        MagicMock(
            id="call_1",
            function=fn_mock,
        )
    ]
    call_msg.content = None
    call_msg.reasoning = None

    # Second turn: terminal gate forces tools=None, model responds with text
    text_msg = MagicMock()
    text_msg.tool_calls = None
    text_msg.content = "Sir, on your screen I observe a terminal with bash."
    text_msg.reasoning = None

    mock_llm.chat.side_effect = [
        MagicMock(choices=[MagicMock(message=call_msg)]),
        MagicMock(choices=[MagicMock(message=text_msg)]),
    ]
    mock_llm.last_metrics = {"model": "qwen3:8b", "duration_s": 0.1, "token_count": 10, "ttft_s": None}

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("tell me what you see on the screen")

    # Verify 2 calls were made to LLM
    assert mock_llm.chat.call_count == 2
    # Verify the second call was made with tools=None (terminal gate closed)
    second_call_kwargs = mock_llm.chat.call_args_list[1][1]
    assert second_call_kwargs.get("tools") is None
    assert "Sir, on your screen I observe a terminal with bash." in reply
