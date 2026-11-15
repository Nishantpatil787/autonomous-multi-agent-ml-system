"""Local LLM Client and health check utilities using Ollama HTTP API.

Zero cloud API calls, zero paid tokens, pure local HTTP integration
using Python standard library (urllib).
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("LLMClient")


class LLMStatus(str, Enum):
    """Execution and availability status of the local LLM subsystem."""

    DISABLED = "disabled"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    REQUEST_FAILED = "request_failed"
    INVALID_RESPONSE = "invalid_response"
    COMPLETED = "completed"


@dataclass
class LLMResponse:
    """Standardized result returned by LocalLLMClient."""

    status: LLMStatus
    content: Optional[str] = None
    structured_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    model: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to a serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


class LocalLLMClient:
    """Client for local Ollama instances via native HTTP requests."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or Config.OLLAMA_MODEL
        self.timeout = timeout if timeout is not None else Config.OLLAMA_TIMEOUT
        self.temperature = (
            temperature if temperature is not None else Config.OLLAMA_TEMPERATURE
        )

    def check_ollama_available(self) -> Tuple[bool, str]:
        """Verify whether the Ollama server is responding."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return True, "Ollama server is active and reachable."
                return False, f"Ollama returned HTTP status {resp.status}."
        except urllib.error.URLError as err:
            return False, f"Ollama service unreachable: {err.reason}"
        except Exception as exc:
            return False, f"Ollama connection check failed: {str(exc)}"

    def check_model_available(self, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """Check if the designated model is installed on the local Ollama instance."""
        target_model = model_name or self.model
        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return False, f"Ollama returned HTTP {resp.status} on tags check."
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match full name (e.g. 'qwen2.5:3b') or base name
                if any(m == target_model or m.startswith(f"{target_model}:") for m in models):
                    return True, f"Model '{target_model}' is available locally."
                return (
                    False,
                    f"Model '{target_model}' not found in installed models: {models}",
                )
        except Exception as exc:
            return False, f"Failed to check model availability: {str(exc)}"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_format: bool = True,
    ) -> LLMResponse:
        """Send a prompt to Ollama and obtain the response."""
        # 1. Health check Ollama service
        ollama_ok, ollama_msg = self.check_ollama_available()
        if not ollama_ok:
            return LLMResponse(
                status=LLMStatus.UNAVAILABLE,
                error_message=ollama_msg,
                model=self.model,
            )

        # 2. Check model availability
        model_ok, model_msg = self.check_model_available(self.model)
        if not model_ok:
            return LLMResponse(
                status=LLMStatus.MODEL_UNAVAILABLE,
                error_message=model_msg,
                model=self.model,
            )

        # 3. Formulate generate payload
        generate_url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if system_instruction:
            payload["system"] = system_instruction
        if json_format:
            payload["format"] = "json"

        # 4. Execute request
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                generate_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            import time

            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed = time.perf_counter() - t0
                if resp.status != 200:
                    return LLMResponse(
                        status=LLMStatus.REQUEST_FAILED,
                        error_message=f"HTTP {resp.status} received from Ollama generate.",
                        model=self.model,
                        duration_seconds=round(elapsed, 4),
                    )
                resp_json = json.loads(resp.read().decode("utf-8"))
                raw_response = resp_json.get("response", "")
                return LLMResponse(
                    status=LLMStatus.COMPLETED,
                    content=raw_response,
                    model=self.model,
                    duration_seconds=round(elapsed, 4),
                )
        except urllib.error.URLError as err:
            return LLMResponse(
                status=LLMStatus.REQUEST_FAILED,
                error_message=f"Ollama request failed: {err.reason}",
                model=self.model,
            )
        except Exception as exc:
            return LLMResponse(
                status=LLMStatus.REQUEST_FAILED,
                error_message=f"Ollama request error: {str(exc)}",
                model=self.model,
            )
