"""Tests for llm/client.py — LocalLLMClient, health check, timeouts, and error handling."""

from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from llm.client import LLMResponse, LLMStatus, LocalLLMClient
from utils.config import Config


class TestLocalLLMClientConfig:
    def test_default_config_values(self):
        client = LocalLLMClient()
        assert client.base_url == Config.OLLAMA_BASE_URL
        assert client.model == Config.OLLAMA_MODEL
        assert client.timeout == Config.OLLAMA_TIMEOUT
        assert client.temperature == Config.OLLAMA_TEMPERATURE

    def test_custom_parameters_override(self):
        client = LocalLLMClient(
            base_url="http://127.0.0.1:11434/",
            model="tinyllama:latest",
            timeout=5.0,
            temperature=0.2,
        )
        assert client.base_url == "http://127.0.0.1:11434"
        assert client.model == "tinyllama:latest"
        assert client.timeout == 5.0
        assert client.temperature == 0.2


class TestOllamaAvailability:
    @patch("urllib.request.urlopen")
    def test_check_ollama_available_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = LocalLLMClient()
        available, msg = client.check_ollama_available()
        assert available is True
        assert "active" in msg.lower()

    @patch("urllib.request.urlopen")
    def test_check_ollama_available_connection_refused(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = LocalLLMClient()
        available, msg = client.check_ollama_available()
        assert available is False
        assert "unreachable" in msg.lower()

    @patch("urllib.request.urlopen")
    def test_check_ollama_available_http_error(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 503
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = LocalLLMClient()
        available, msg = client.check_ollama_available()
        assert available is False


class TestModelAvailability:
    @patch("urllib.request.urlopen")
    def test_check_model_available_present(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models": [{"name": "qwen2.5:3b"}, {"name": "llama3:latest"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = LocalLLMClient(model="qwen2.5:3b")
        available, msg = client.check_model_available()
        assert available is True
        assert "available locally" in msg.lower()

    @patch("urllib.request.urlopen")
    def test_check_model_available_missing(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models": [{"name": "llama3:latest"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = LocalLLMClient(model="nonexistent-model")
        available, msg = client.check_model_available()
        assert available is False
        assert "not found" in msg.lower()


class TestLLMGenerate:
    @patch.object(LocalLLMClient, "check_ollama_available", return_value=(False, "Ollama down"))
    def test_generate_returns_unavailable_when_ollama_down(self, mock_check):
        client = LocalLLMClient()
        resp = client.generate("test prompt")
        assert resp.status == LLMStatus.UNAVAILABLE
        assert "Ollama down" in resp.error_message

    @patch.object(LocalLLMClient, "check_ollama_available", return_value=(True, "Ollama OK"))
    @patch.object(LocalLLMClient, "check_model_available", return_value=(False, "Model missing"))
    def test_generate_returns_model_unavailable(self, mock_m, mock_o):
        client = LocalLLMClient()
        resp = client.generate("test prompt")
        assert resp.status == LLMStatus.MODEL_UNAVAILABLE
        assert "Model missing" in resp.error_message

    @patch.object(LocalLLMClient, "check_ollama_available", return_value=(True, "Ollama OK"))
    @patch.object(LocalLLMClient, "check_model_available", return_value=(True, "Model OK"))
    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen, mock_m, mock_o):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"response": "{\\"summary\\": \\"success\\"}"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = LocalLLMClient()
        resp = client.generate("test prompt")
        assert resp.status == LLMStatus.COMPLETED
        assert resp.content == '{"summary": "success"}'
        assert resp.duration_seconds >= 0

    @patch.object(LocalLLMClient, "check_ollama_available", return_value=(True, "Ollama OK"))
    @patch.object(LocalLLMClient, "check_model_available", return_value=(True, "Model OK"))
    @patch("urllib.request.urlopen")
    def test_generate_timeout_handled(self, mock_urlopen, mock_m, mock_o):
        mock_urlopen.side_effect = TimeoutError("Timed out")
        client = LocalLLMClient()
        resp = client.generate("test prompt")
        assert resp.status == LLMStatus.REQUEST_FAILED
        assert "Timed out" in resp.error_message


class TestLLMResponseSerialization:
    def test_to_dict_format(self):
        resp = LLMResponse(status=LLMStatus.COMPLETED, content="test", model="qwen2.5:3b")
        d = resp.to_dict()
        assert d["status"] == "completed"
        assert d["content"] == "test"
        assert d["model"] == "qwen2.5:3b"
