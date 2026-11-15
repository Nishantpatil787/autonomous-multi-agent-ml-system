"""LLM module entrypoint for Phase 8.

Exports:
    - LocalLLMClient: native HTTP client for Ollama
    - LLMResponse: response dataclass
    - LLMStatus: status enum
    - LLMService: high-level coordinator
"""

from llm.client import LLMResponse, LLMStatus, LocalLLMClient
from llm.service import LLMService

__all__ = [
    "LocalLLMClient",
    "LLMResponse",
    "LLMStatus",
    "LLMService",
]
