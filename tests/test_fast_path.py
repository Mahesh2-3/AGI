import time
from unittest.mock import MagicMock
from core.agent import JarvisAgent
from core.fast_path import FastPathRouter
from core.llm_client import LLMClient
from tools.base import ToolRegistry, ToolResult


def test_fast_path_time_routing():
    registry = ToolRegistry()

    @registry.register(description="Get time")
    def get_current_time() -> str:
        return "17:45:00"

    router = FastPathRouter(registry=registry)

    # Various time queries
    res1 = router.route("Jarvis, what time is it?")
    assert res1 is not None
    assert "17:45:00" in res1

    res2 = router.route("what's the time")
    assert res2 is not None
    assert "17:45:00" in res2

    res3 = router.route("current time")
    assert res3 is not None
    assert "17:45:00" in res3


def test_fast_path_complex_query_falls_back():
    registry = ToolRegistry()
    router = FastPathRouter(registry=registry)

    # Complex cognitive request should return None so ReAct LLM handles it
    res = router.route("Jarvis, please analyze the differences between Wayland and X11")
    assert res is None


def test_fast_path_in_agent_bypasses_llm():
    registry = ToolRegistry()

    @registry.register(description="Get current time")
    def get_current_time() -> str:
        return "17:46:00"

    mock_llm = MagicMock(spec=LLMClient)
    # LLM should NOT be called at all for fast path query
    mock_llm.chat.side_effect = RuntimeError("LLM should not be reached for fast path!")

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=True)

    t0 = time.perf_counter()
    reply = agent.step("Jarvis, what time is it?")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert "17:46:00" in reply
    assert mock_llm.chat.call_count == 0
    assert elapsed_ms < 50  # Must be sub-50ms (typically < 5ms)


def test_fast_path_workspace_routing():
    registry = ToolRegistry()

    @registry.register(description="Make working workspace active")
    def make_working_workspace_active() -> dict:
        return {"was_active": False, "switched": True, "message": "Switched display to workspace 1"}

    router = FastPathRouter(registry=registry)
    res = router.route("if the jarvis working workspace is not the active one make the jarvis that workspace active")
    assert res is not None
    assert "Switched display to workspace 1" in res


if __name__ == "__main__":
    test_fast_path_time_routing()
    test_fast_path_complex_query_falls_back()
    test_fast_path_in_agent_bypasses_llm()
    test_fast_path_workspace_routing()
    print("✅ All fast path router tests passed successfully!")
