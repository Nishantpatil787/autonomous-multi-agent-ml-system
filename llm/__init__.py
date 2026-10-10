"""Optional local LLM layer over Ollama (Phase 8).

Phase 8 adds an assistant, not an authority. The module is OFF by default: without
``OLLAMA_ENABLED`` or the ``--llm`` flag nothing here opens a socket, builds a
prompt or imports a client, and Phases 1-7 behave exactly as before.

The layer has four parts, each in its own module:

``llm.client``
    The only module that knows Ollama speaks HTTP. Owns the transport, the
    loopback check, the timeout, the bounded retry and the availability checks.
    Uses the standard library, so the project gains **no dependency**.
``llm.prompts``
    Every instruction, and the allowlist that decides what the model may see.
``llm.parser``
    Turns generated text into a validated record, or into a typed failure.
``llm.service``
    Decides whether to call at all, and guarantees no method raises.

Typical use::

    from llm import LLMService

    service = LLMService(enabled=True)
    result = service.interpret_data_analysis(state)
    print(result.status.value)      # completed / unavailable / disabled / ...
    print(result.interpretation)    # validated content, or None

The boundary that makes this safe to ship: every public result is an
:class:`~llm.state.LLMResult` carrying *text and a status*. There is no method here
that returns a metric, a model, a ranking or a routing decision, so the language
model cannot influence the deterministic pipeline even if its output were
adversarial.
"""

__version__ = "0.8.0"

from llm.client import (
    LLMConnectionError,
    LLMError,
    LLMNotConfiguredError,
    LLMUnsupportedHostError,
    LocalLLMClient,
    build_client,
    is_local_url,
    require_local_url,
    urllib_transport,
)
from llm.parser import (
    INTERPRETATION_FIELDS,
    PLAN_FIELDS,
    LLMParseError,
    ParsedLLMResponse,
    excerpt,
    extract_json_block,
    parse_interpretation,
    parse_plan,
    parse_response,
)
from llm.prompts import (
    INTERPRETATION_SCHEMA_TEXT,
    PLAN_SCHEMA_TEXT,
    SYSTEM_INSTRUCTION,
    PromptContext,
    build_context,
    build_data_analysis_context,
    build_evaluation_context,
    build_optimization_context,
    build_prompt,
    build_repair_prompt,
    sanitize_value,
)
from llm.service import LLMService, build_service
from llm.state import (
    LLM_CONTEXT_FIELDS,
    LLMConfidence,
    LLMErrorCategory,
    LLMHealth,
    LLMInterpretation,
    LLMOptimizationPlan,
    LLMResponse,
    LLMResult,
    LLMRole,
    LLMStatus,
    allowed_context_fields,
    find_forbidden_claims,
)

__all__ = [
    "__version__",
    # state
    "LLMStatus",
    "LLMErrorCategory",
    "LLMConfidence",
    "LLMRole",
    "LLMResponse",
    "LLMHealth",
    "LLMInterpretation",
    "LLMOptimizationPlan",
    "LLMResult",
    "LLM_CONTEXT_FIELDS",
    "allowed_context_fields",
    "find_forbidden_claims",
    # client
    "LocalLLMClient",
    "LLMError",
    "LLMConnectionError",
    "LLMNotConfiguredError",
    "LLMUnsupportedHostError",
    "build_client",
    "urllib_transport",
    "is_local_url",
    "require_local_url",
    # prompts
    "SYSTEM_INSTRUCTION",
    "INTERPRETATION_SCHEMA_TEXT",
    "PLAN_SCHEMA_TEXT",
    "PromptContext",
    "build_context",
    "build_data_analysis_context",
    "build_evaluation_context",
    "build_optimization_context",
    "build_prompt",
    "build_repair_prompt",
    "sanitize_value",
    # parser
    "INTERPRETATION_FIELDS",
    "PLAN_FIELDS",
    "LLMParseError",
    "ParsedLLMResponse",
    "parse_response",
    "parse_interpretation",
    "parse_plan",
    "extract_json_block",
    "excerpt",
    # service
    "LLMService",
    "build_service",
]