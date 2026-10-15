from unittest.mock import MagicMock
from core.agent import JarvisAgent
from core.llm_client import LLMClient
from tools.base import ToolRegistry, ToolResult


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


def test_agent_deduplicates_repeated_perception_calls():
    registry = ToolRegistry()
    inspect_call_count = 0

    @registry.register(description="Inspect screen")
    def inspect_screen(query: str = "desktop") -> dict:
        nonlocal inspect_call_count
        inspect_call_count += 1
        return {"description": "VS Code is open on desktop"}

    # Mock LLM calls inspect_screen twice in a row without any intervening GUI actions
    tc1 = FakeToolCall("call_1", "inspect_screen", '{"focus": "desktop", "query": "overview"}')
    tc2 = FakeToolCall("call_2", "inspect_screen", '{"query": "desktop", "user_prompt": "information"}')

    resp1 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc1]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc2]))
    resp3 = FakeChatResponse(FakeChatCompletionMessage(content="VS Code is currently active on your screen, Sir."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [resp1, resp2, resp3]

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("check my screen")

    # Crucial assertion: inspect_screen was only executed ONCE despite two tool calls from LLM
    assert inspect_call_count == 1
    assert "VS Code" in reply


def test_agent_invalidates_cache_on_state_change():
    registry = ToolRegistry()
    inspect_call_count = 0
    click_call_count = 0

    @registry.register(description="Inspect screen")
    def inspect_screen(query: str = "desktop") -> dict:
        nonlocal inspect_call_count
        inspect_call_count += 1
        return {"description": f"Screen observation #{inspect_call_count}"}

    @registry.register(description="Click element")
    def click_element(element_description: str) -> dict:
        nonlocal click_call_count
        click_call_count += 1
        return {"clicked": True}

    # Inspect -> Click -> Inspect -> Answer
    tc1 = FakeToolCall("call_1", "inspect_screen", '{"query": "first"}')
    tc2 = FakeToolCall("call_2", "click_element", '{"element_description": "submit button"}')
    tc3 = FakeToolCall("call_3", "inspect_screen", '{"query": "second"}')

    resp1 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc1]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc2]))
    resp3 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc3]))
    resp4 = FakeChatResponse(FakeChatCompletionMessage(content="Clicked and verified."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [resp1, resp2, resp3, resp4]

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("click submit and verify")

    # Because a state-changing tool (click_element) was dispatched, cache was invalidated
    # so second inspect_screen was permitted
    assert inspect_call_count == 2
    assert click_call_count == 1
    assert "Clicked and verified" in reply


def test_agent_suppresses_identical_consecutive_tool_call():
    registry = ToolRegistry()
    launch_count = 0

    @registry.register(description="Launch application")
    def launch_application(app_name: str) -> dict:
        nonlocal launch_count
        launch_count += 1
        return {"launched": app_name}

    tc1 = FakeToolCall("call_1", "launch_application", '{"app_name": "gedit"}')
    tc2 = FakeToolCall("call_2", "launch_application", '{"app_name": "gedit"}')

    resp1 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc1]))
    resp2 = FakeChatResponse(FakeChatCompletionMessage(tool_calls=[tc2]))
    resp3 = FakeChatResponse(FakeChatCompletionMessage(content="Gedit launched, Sir."))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.side_effect = [resp1, resp2, resp3]

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)
    reply = agent.step("open gedit")

    assert launch_count == 1
    assert "Gedit launched" in reply


def test_agent_preserves_active_goal_across_corrections():
    registry = ToolRegistry()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = FakeChatResponse(FakeChatCompletionMessage(content="Action acknowledged, Sir."))

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)

    # Turn 1: User gives primary action goal
    agent.step("Complete the daily problem in the already opened browser and submit it")
    assert agent.active_goal == "Complete the daily problem in the already opened browser and submit it"

    # Turn 2: User provides correction / direction
    agent.step("did the work in already opened one")
    # Goal is preserved!
    assert agent.active_goal == "Complete the daily problem in the already opened browser and submit it"
    # Last user message was enriched with active objective context
    last_user_msg = [m for m in agent.messages if m.get("role") == "user"][-1]["content"]
    assert "[CURRENT ACTIVE OBJECTIVE]" in last_user_msg
    assert "Complete the daily problem" in last_user_msg
    assert "[USER CORRECTION / DIRECTION]: did the work in already opened one" in last_user_msg

    # Turn 3: User says "stop" to cancel goal
    cancel_reply = agent.step("stop")
    assert agent.active_goal is None
    assert "cleared the active goal" in cancel_reply


def test_new_task_not_hijacked_by_prior_observational_query():
    registry = ToolRegistry()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat.return_value = FakeChatResponse(FakeChatCompletionMessage(content="Understood, Sir."))

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=registry, enable_fast_path=False)

    # Turn 1: Observational screen reading
    agent.step("what do you see on the screen ?")
    # Observational tasks must not leave a lingering active action goal
    assert agent.active_goal is None

    # Turn 2: New command with 7 words (previously hijacked by len <= 8 rule)
    agent.step("chat with the chatgpt for 10 messages")
    # Must be recognized as a NEW action task, not a follow-up to observation
    assert agent.active_goal == "chat with the chatgpt for 10 messages"

    last_user_msg = [m for m in agent.messages if m.get("role") == "user"][-1]["content"]
    assert "[CURRENT ACTIVE OBJECTIVE]" not in last_user_msg
    assert "what do you see on the screen" not in last_user_msg
    assert last_user_msg == "chat with the chatgpt for 10 messages"

    # Turn 3: Explicit continuation of the chatgpt task
    agent.step("in the browser")
    assert agent.active_goal == "chat with the chatgpt for 10 messages"
    cont_user_msg = [m for m in agent.messages if m.get("role") == "user"][-1]["content"]
    assert "[CURRENT ACTIVE OBJECTIVE]: chat with the chatgpt for 10 messages" in cont_user_msg
    assert "[USER CORRECTION / DIRECTION]: in the browser" in cont_user_msg

