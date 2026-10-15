import json
import pytest
from unittest.mock import MagicMock, patch
from core.agent import JarvisAgent
from core.llm_client import LLMClient
from tools.base import ToolRegistry


def make_fake_intent_response(intent: str, resolved_action: str = "", reasoning: str = "Test reasoning"):
    payload = {
        "intent": intent,
        "resolved_action": resolved_action,
        "reasoning": reasoning,
    }
    msg = MagicMock()
    msg.content = json.dumps(payload)
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_intent_empty_input_deterministic():
    agent = JarvisAgent(llm_client=MagicMock(spec=LLMClient), tool_registry=ToolRegistry())
    result = agent._classify_request_intent("")
    assert result["type"] == "general"
    assert result["is_followup"] is False
    assert result["is_observational"] is False
    assert result["is_new_action"] is False


def test_intent_emergency_stop_deterministic():
    agent = JarvisAgent(llm_client=MagicMock(spec=LLMClient), tool_registry=ToolRegistry())
    agent.active_goal = "Download 5GB dataset"

    for stop_cmd in ["stop", "cancel", "reset", "clear goal", "abort"]:
        result = agent._classify_request_intent(stop_cmd)
        assert result["type"] == "cancel"
        assert result["is_followup"] is False
        assert result["is_observational"] is False
        assert result["is_new_action"] is False


def test_intent_new_task_semantic():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="new_task",
        resolved_action="chat with chatgpt for 10 messages",
        reasoning="User wants to start a fresh interaction workflow",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    result = agent._classify_request_intent("chat with chatgpt for 10 messages")

    assert result["type"] == "new_task"
    assert result["is_new_action"] is True
    assert result["is_followup"] is False
    assert result["is_observational"] is False
    assert result["resolved_action"] == "chat with chatgpt for 10 messages"


def test_intent_compound_view_and_act_is_new_task():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="new_task",
        resolved_action="inspect screen and click submit button",
        reasoning="Compound request requiring both inspection and desktop action",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    result = agent._classify_request_intent("look at the screen and click submit")

    assert result["type"] == "new_task"
    assert result["is_new_action"] is True
    assert result["is_observational"] is False


def test_intent_continuation_with_context_resolution():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="continuation",
        resolved_action="open chess.com in Google Chrome",
        reasoning="Relative instruction referencing previous workflow in Firefox",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.last_completed_goal = "Open chess.com in Firefox"

    result = agent._classify_request_intent("do the same in Chrome")

    assert result["type"] == "continuation"
    assert result["is_followup"] is True
    assert result["is_new_action"] is False
    assert result["is_observational"] is False
    assert "Chrome" in result["resolved_action"]


def test_intent_continuation_relative_direction():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="continuation",
        resolved_action="click the secondary download button below the primary",
        reasoning="Follow-up correcting target button selection",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.active_goal = "Download the Linux installer"

    result = agent._classify_request_intent("click the other one")

    assert result["type"] == "continuation"
    assert result["is_followup"] is True
    assert result["is_new_action"] is False


def test_intent_continuation_close_it():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="continuation",
        resolved_action="close the calculator application",
        reasoning="Closing the application launched in active goal",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.active_goal = "Launch calculator"

    result = agent._classify_request_intent("close it")

    assert result["type"] == "continuation"
    assert result["is_followup"] is True
    assert "calculator" in result["resolved_action"]


def test_intent_observational_semantic():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="observational",
        resolved_action="inspect and describe active desktop windows",
        reasoning="Pure perception query without desktop manipulation",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    result = agent._classify_request_intent("what do you see on the screen ?")

    assert result["type"] == "observational"
    assert result["is_observational"] is True
    assert result["is_new_action"] is False
    assert result["is_followup"] is False


def test_intent_general_question_during_active_goal():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="general",
        resolved_action="explain Python programming language",
        reasoning="Conversational question not related to active goal and requiring no desktop execution",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.active_goal = "Write script in VSCode"

    result = agent._classify_request_intent("What is Python?")

    assert result["type"] == "general"
    assert result["is_followup"] is False
    assert result["is_new_action"] is False
    assert result["is_observational"] is False


def test_intent_cancel_semantic():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="cancel",
        resolved_action="abort the active file transfer",
        reasoning="User explicitly asked to stop active execution",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.active_goal = "Download large file"

    result = agent._classify_request_intent("please stop this task right now")

    assert result["type"] == "cancel"
    assert result["is_followup"] is False
    assert result["is_new_action"] is False


def test_intent_malformed_json_fallback_with_active_goal():
    mock_llm = MagicMock(spec=LLMClient)
    # Model emits non-JSON gibberish
    bad_msg = MagicMock()
    bad_msg.content = "I am not sure what JSON is but I think you should continue."
    bad_msg.tool_calls = None
    mock_llm.classify_intent.return_value = MagicMock(choices=[MagicMock(message=bad_msg)])

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    agent.active_goal = "Submit registration form"

    result = agent._classify_request_intent("press enter")

    assert result["type"] == "continuation"
    assert result["is_followup"] is True
    assert result["is_new_action"] is False
    assert "Safe fallback" in result["reasoning"]


def test_intent_malformed_json_fallback_without_active_goal():
    mock_llm = MagicMock(spec=LLMClient)
    bad_msg = MagicMock()
    bad_msg.content = "Some unparseable text without JSON braces"
    bad_msg.tool_calls = None
    mock_llm.classify_intent.return_value = MagicMock(choices=[MagicMock(message=bad_msg)])

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())

    result = agent._classify_request_intent("launch blender")

    assert result["type"] == "new_task"
    assert result["is_new_action"] is True
    assert result["is_followup"] is False
    assert "Safe fallback" in result["reasoning"]


def test_intent_qwen_think_tag_stripping():
    mock_llm = MagicMock(spec=LLMClient)
    # Qwen3 outputs thinking tokens before JSON
    think_content = (
        "<think>\n"
        "The user is asking a general knowledge question about Python.\n"
        "This does not operate desktop tools.\n"
        "</think>\n"
        '```json\n{"intent": "general", "resolved_action": "explain python", "reasoning": "general question"}\n```'
    )
    msg = MagicMock()
    msg.content = think_content
    msg.tool_calls = None
    mock_llm.classify_intent.return_value = MagicMock(choices=[MagicMock(message=msg)])

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    result = agent._classify_request_intent("What is Python?")

    assert result["type"] == "general"
    assert result["resolved_action"] == "explain python"


def test_intent_unknown_intent_type_fallback():
    mock_llm = MagicMock(spec=LLMClient)
    # Model hallucinated an unknown intent
    mock_llm.classify_intent.return_value = make_fake_intent_response(
        intent="explore_cosmos",
        resolved_action="look at stars",
    )

    agent = JarvisAgent(llm_client=mock_llm, tool_registry=ToolRegistry())
    result = agent._classify_request_intent("look at the stars")

    # Should safely fallback
    assert result["type"] == "new_task"
    assert result["is_new_action"] is True


@pytest.mark.skipif(
    not LLMClient().is_configured,
    reason="Ollama local GPU inference not available in current environment",
)
def test_live_qwen3_semantic_classification_suite():
    """Runs live end-to-end intent classification through local Ollama Qwen3."""
    client = LLMClient()
    agent = JarvisAgent(llm_client=client, tool_registry=ToolRegistry(), enable_fast_path=False)

    # 1. New task
    r1 = agent._classify_request_intent("chat with chatgpt for 10 messages")
    assert r1["type"] == "new_task"
    assert r1["is_new_action"] is True

    # 2. General question while active goal exists
    agent.active_goal = "Open terminal and compile C++ code"
    r2 = agent._classify_request_intent("What is Python?")
    assert r2["type"] == "general"
    assert r2["is_followup"] is False

    # 3. Continuation reference resolution
    agent.active_goal = "Click download button"
    r3 = agent._classify_request_intent("click the other one")
    assert r3["type"] == "continuation"
    assert r3["is_followup"] is True

    # 4. Observational
    r4 = agent._classify_request_intent("what do you see on the screen ?")
    assert r4["type"] == "observational"
    assert r4["is_observational"] is True

    # 5. Compound request
    r5 = agent._classify_request_intent("look at screen and click submit")
    assert r5["type"] == "new_task"
    assert r5["is_new_action"] is True
