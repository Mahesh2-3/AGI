from unittest.mock import MagicMock, patch
import json
import io
import time
from core.model_manager import OllamaModelManager


def test_get_loaded_models_mock():
    manager = OllamaModelManager(base_url="http://mock:11434")
    mock_resp_data = json.dumps({
        "models": [
            {"name": "qwen3:4b", "model": "qwen3:4b"},
            {"name": "qwen2.5vl:3b", "model": "qwen2.5vl:3b"},
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_resp_data
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        loaded = manager.get_loaded_models()
        assert loaded == ["qwen3:4b", "qwen2.5vl:3b"]


def test_unload_model_mock():
    manager = OllamaModelManager(base_url="http://mock:11434")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        success = manager.unload_model("qwen3:4b")
        assert success is True
        # Verify keep_alive: 0 in request payload
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "qwen3:4b"
        assert payload["keep_alive"] == 0


def test_prepare_for_model_unloads_competing():
    manager = OllamaModelManager(base_url="http://mock:11434")
    with patch.object(manager, "get_loaded_models", return_value=["qwen3:4b"]):
        with patch.object(manager, "unload_model", return_value=True) as mock_unload:
            metrics = manager.prepare_for_model("qwen2.5vl:3b")
            mock_unload.assert_called_once_with("qwen3:4b")
            assert metrics["was_resident"] is False
            assert metrics["unloaded_models"] == ["qwen3:4b"]


def test_prepare_for_model_differentiates_4b_and_8b():
    # M2 verification: qwen3:4b and qwen3:8b must NOT be treated as the same model!
    manager = OllamaModelManager(base_url="http://mock:11434")
    with patch.object(manager, "get_loaded_models", return_value=["qwen3:4b"]):
        with patch.object(manager, "unload_model", return_value=True) as mock_unload:
            metrics = manager.prepare_for_model("qwen3:8b")
            mock_unload.assert_called_once_with("qwen3:4b")
            assert metrics["was_resident"] is False


def test_prepare_for_model_keeps_same_model():
    manager = OllamaModelManager(base_url="http://mock:11434")
    with patch.object(manager, "get_loaded_models", return_value=["qwen3:4b"]):
        with patch.object(manager, "unload_model") as mock_unload:
            metrics = manager.prepare_for_model("qwen3:4b")
            mock_unload.assert_not_called()
            assert metrics["was_resident"] is True


def test_unload_on_idle_force():
    manager = OllamaModelManager(base_url="http://mock:11434")
    with patch.object(manager, "unload_all_models", return_value=1) as mock_unload_all:
        manager.unload_on_idle(force=True)
        mock_unload_all.assert_called_once()


def test_unload_on_idle_timeout():
    manager = OllamaModelManager(base_url="http://mock:11434")
    # Simulate recent use (within timeout) -> should not unload
    manager.last_used_timestamp = time.time()
    with patch.object(manager, "unload_all_models", return_value=1) as mock_unload_all:
        manager.unload_on_idle(force=False)
        mock_unload_all.assert_not_called()

    # Simulate timeout elapsed
    manager.last_used_timestamp = time.time() - 3600
    with patch.object(manager, "unload_all_models", return_value=1) as mock_unload_all:
        manager.unload_on_idle(force=False)
        mock_unload_all.assert_called_once()
