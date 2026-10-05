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


if __name__ == "__main__":
    test_llm_client_rate_limit_cascading()
    test_llm_client_non_rate_limit_error_not_swallowed()
    print("✅ All LLM cascade tests passed successfully!")
