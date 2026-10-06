from unittest.mock import MagicMock
from core.llm_client import LLMClient


class MockRateLimitError(Exception):
    def __init__(self, message="Rate limit reached for model openai/gpt-oss-120b"):
        super().__init__(message)
        self.status_code = 429


def test_llm_client_rate_limit_cascading():
    client = LLMClient(provider="groq")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    # Primary model fails with 429 RateLimitError, second model succeeds
    first_call_error = MockRateLimitError()
    second_call_success = MagicMock(choices=[MagicMock(message=MagicMock(content="Cascaded answer"))])

    mock_sdk.chat.completions.create.side_effect = [first_call_error, second_call_success]

    fallback_logs = []
    client.on_fallback = lambda old_m, new_m, reason: fallback_logs.append((old_m, new_m, reason))

    messages = [{"role": "user", "content": "Hello"}]
    res = client.chat(messages=messages)

    assert res.choices[0].message.content == "Cascaded answer"
    assert len(fallback_logs) == 1
    assert fallback_logs[0][0] == client.model_cascade[0]
    assert fallback_logs[0][1] == client.model_cascade[1]
    assert "rate limit" in fallback_logs[0][2].lower()


def test_llm_client_non_rate_limit_error_not_swallowed():
    client = LLMClient(provider="groq")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    mock_sdk.chat.completions.create.side_effect = ValueError("Invalid argument")

    messages = [{"role": "user", "content": "Hello"}]
    try:
        client.chat(messages=messages)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid argument" in str(e)


def test_llm_client_cooldown():
    client = LLMClient(provider="groq")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    # Call 1: model 1 fails, model 2 succeeds
    call1_err = MockRateLimitError()
    call1_ok = MagicMock(choices=[MagicMock(message=MagicMock(content="Success 1"))])
    # Call 2: should directly call model 2 without retrying model 1
    call2_ok = MagicMock(choices=[MagicMock(message=MagicMock(content="Success 2"))])

    mock_sdk.chat.completions.create.side_effect = [call1_err, call1_ok, call2_ok]

    client.chat(messages=[{"role": "user", "content": "1"}])
    # Verify cooldown was recorded
    assert client.model_cascade[0] in client._model_cooldowns

    # Second chat call should use model 2 directly
    res2 = client.chat(messages=[{"role": "user", "content": "2"}])
    assert res2.choices[0].message.content == "Success 2"
    # Ensure call 2 requested model_cascade[1]
    last_call_kwargs = mock_sdk.chat.completions.create.call_args_list[-1].kwargs
    assert last_call_kwargs["model"] == client.model_cascade[1]


def test_llm_client_multi_key_rotation():
    client = LLMClient(provider="groq")
    mock_sdk1 = MagicMock()
    mock_sdk2 = MagicMock()

    client._groq_keys = ["gsk_key1", "gsk_key2"]
    client._groq_clients = {"gsk_key1": mock_sdk1, "gsk_key2": mock_sdk2}
    client._client = mock_sdk1

    # Key 1 hits 429 rate limit, Key 2 succeeds on the primary model
    mock_sdk1.chat.completions.create.side_effect = MockRateLimitError("429 TPM limit reached")
    mock_sdk2.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Answer from Key 2"))]
    )

    fallback_notes = []
    client.on_fallback = lambda old, new, reason: fallback_notes.append((old, new, reason))

    res = client.chat(messages=[{"role": "user", "content": "Hello pool"}])

    assert res.choices[0].message.content == "Answer from Key 2"
    assert "gsk_key1" in client._key_cooldowns
    # Primary model was retained without degrading
    assert mock_sdk2.chat.completions.create.call_args.kwargs["model"] == client.model_cascade[0]
    assert len(fallback_notes) == 1
    assert "rotating to alternate Groq key" in fallback_notes[0][2]


if __name__ == "__main__":
    test_llm_client_rate_limit_cascading()
    test_llm_client_non_rate_limit_error_not_swallowed()
    test_llm_client_cooldown()
    test_llm_client_multi_key_rotation()
    print("✅ All LLM cascade tests passed successfully!")
