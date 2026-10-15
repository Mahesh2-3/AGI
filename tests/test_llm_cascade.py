from unittest.mock import MagicMock, patch
import time
from config.settings import settings
from core.llm_client import LLMClient


def test_llm_client_default_ollama():
    client = LLMClient()
    assert client.provider == "ollama"
    assert client.model == settings.DEFAULT_MODEL
    assert client.is_configured is True


def test_llm_client_model_configuration():
    client = LLMClient(model="qwen3:8b")
    assert client.model == "qwen3:8b"


def test_llm_client_latency_tracking():
    client = LLMClient(model="qwen3:8b")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    mock_resp = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))],
        usage=MagicMock(completion_tokens=25),
    )
    mock_sdk.chat.completions.create.return_value = mock_resp

    with patch("core.model_manager.model_manager.prepare_for_model") as mock_prep:
        res = client.chat(messages=[{"role": "user", "content": "Hello"}])
        mock_prep.assert_called_once_with("qwen3:8b")

    assert res.choices[0].message.content == "Test response"
    assert client.last_metrics["model"] == "qwen3:8b"
    assert client.last_metrics["token_count"] == 25
    assert client.last_metrics["duration_s"] >= 0.0


def test_llm_client_streaming_ttft():
    client = LLMClient(model="qwen3:8b")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))])
    chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content=" Sir"))])
    mock_sdk.chat.completions.create.return_value = iter([chunk1, chunk2])

    with patch("core.model_manager.model_manager.prepare_for_model") as mock_prep:
        stream = client.chat_stream(messages=[{"role": "user", "content": "Hello"}])
        collected = list(stream)
        mock_prep.assert_called_once_with("qwen3:8b")

    assert len(collected) == 2
    assert client.last_metrics["streamed"] is True
    assert client.last_metrics["token_count"] == 2
    assert client.last_metrics["ttft_s"] is not None
    assert client.last_metrics["duration_s"] >= 0.0


def test_llm_client_residency_extra_body():
    client = LLMClient(model="qwen3:8b")
    mock_sdk = MagicMock()
    client._client = mock_sdk

    mock_sdk.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        usage=None,
    )

    with patch("core.model_manager.model_manager.prepare_for_model"):
        client.chat(messages=[{"role": "user", "content": "hi"}])

    kwargs = mock_sdk.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"keep_alive": client.keep_alive}
