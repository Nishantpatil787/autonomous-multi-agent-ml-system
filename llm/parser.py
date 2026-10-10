"""Validation of the local model's structured response (Phase 8).

A language model answers with text. This module turns that text into either a
validated :class:`~llm.state.LLMInterpretation`, a validated
:class:`~llm.state.LLMOptimizationPlan`, or a typed failure. It is the reason the
project can let a model speak at all without letting it decide anything.

Four things happen here, in this order:

1. **Extraction.** Small local models wrap JSON in prose or markdown fences, so the
   first balanced ``{...}`` object is located with a scanner that respects strings
   and escapes, rather than trusting the text to start with ``{``.
2. **Shape validation.** A required key that is missing or of the wrong type is a
   schema violation. Only two normalisations are performed, and both are recorded
   as notes: a scalar where a list was requested, and an unrecognised confidence
   label. Nothing else is coerced.
3. **Claim checking.** Phrases from :data:`~utils.config.LLM_FORBIDDEN_CLAIMS` are
   detected and recorded. The text is *not* rewritten: silently editing generated
   prose would hide what the model actually said.
4. **Discarding.** Unknown fields are dropped, and their names are recorded in
   ``notes`` so an unexpected shape is visible instead of invisible.

Nothing here raises: every failure returns an :class:`LLMParseError`, because a
malformed model answer must never stop the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from llm.state import (
    LLMConfidence,
    LLMErrorCategory,
    LLMInterpretation,
    LLMOptimizationPlan,
    LLMRole,
    find_forbidden_claims,
)
from utils import config

#: Field names the interpretation schema accepts. Anything else is discarded.
INTERPRETATION_FIELDS: Tuple[str, ...] = (
    "summary",
    "observations",
    "interpretation",
    "hypotheses",
    "recommendations",
    "confidence",
)

#: Field names the optimisation-plan schema accepts.
PLAN_FIELDS: Tuple[str, ...] = (
    "summary",
    "candidate_models",
    "focus",
    "reason",
    "confidence",
)

#: Substrings that make a focus entry look like code or a search-space directive.
#: The planner may describe *what* to prioritise; it may not write *how* to search.
#:
#: These are deliberately punctuation-anchored. An earlier version rejected the bare
#: word ``class``, which silently threw away the legitimate focus phrase "class
#: imbalance" - the most obvious thing to suggest for an imbalanced target. Only
#: unambiguous code syntax is rejected here.
_CODE_MARKERS: Tuple[str, ...] = (
    "optuna",
    "trial.",
    "suggest_",
    "study.",
    "create_study",
    "=",
    "->",
    "()",
    "{",
    "}",
)

#: Patterns that identify Python or search-space syntax unambiguously.
_CODE_PATTERNS: Tuple[Any, ...] = (
    # ``import numpy`` / ``from optuna import ...``
    re.compile(r"\b(?:import|from)\s+\w"),
    # ``def search(trial):`` - a definition needs an argument list or a colon.
    re.compile(r"\bdef\s+\w+\s*[:(]"),
    # ``class Foo:`` - but not the phrase "class imbalance".
    re.compile(r"\bclass\s+\w+\s*[:(]"),
    # ``return trial.suggest_float(...)``
    re.compile(r"\breturn\s+\w+\s*[:(]"),
    # ``lambda x: x`` and any other lambda expression.
    re.compile(r"\blambda\b"),
    # Any call-like expression: ``suggest_float('alpha', 1e-5, 1)``, ``fit(X, y)``.
    re.compile(r"\w+\s*\("),
)


@dataclass(frozen=True)
class LLMParseError:
    """Why a model response could not be turned into a validated record.

    Attributes:
        category: Which kind of failure it was.
        detail: A one-line, actionable description.
        raw_excerpt: A bounded excerpt of the offending text, for the log.
    """

    category: LLMErrorCategory
    detail: str = ""
    raw_excerpt: str = ""

    @property
    def repairable(self) -> bool:
        """Return True when one more request could plausibly fix this."""
        return self.category in (
            LLMErrorCategory.MALFORMED_JSON,
            LLMErrorCategory.SCHEMA_VIOLATION,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "category": self.category.value,
            "detail": self.detail,
            "raw_excerpt": self.raw_excerpt,
        }


class _SchemaViolation(Exception):
    """Internal control-flow signal for a validation failure.

    :class:`LLMParseError` is a *value*, not an exception: callers receive it as a
    return value so that a malformed model answer cannot become a pipeline failure.
    Raising it would conflate the two roles and make ``except`` clauses depend on a
    dataclass, so a private exception carries it internally and is converted back at
    the module boundary. It never leaves this module.
    """

    def __init__(self, error: LLMParseError) -> None:
        """Wrap the value that will be returned to the caller.

        Args:
            error: The typed failure describing what went wrong.
        """
        super().__init__(error.detail)
        self.error = error

    @classmethod
    def of(
        cls, category: LLMErrorCategory, detail: str
    ) -> "_SchemaViolation":
        """Build a signal from a category and a detail message.

        Args:
            category: Which kind of validation failed.
            detail: A one-line explanation.

        Returns:
            The signal, ready to be raised.
        """
        return cls(LLMParseError(category=category, detail=detail))


def excerpt(text: str, limit: Optional[int] = None) -> str:
    """Return a bounded single-line excerpt of a model response.

    Args:
        text: The raw response text.
        limit: Maximum length. Defaults to the configured log bound.

    Returns:
        A single-line excerpt safe to put in a log line.
    """
    bound = limit if limit is not None else config.LLM_RAW_RESPONSE_LOG_LIMIT
    flattened = " ".join(str(text or "").split())
    if len(flattened) <= bound:
        return flattened
    return flattened[: bound - 3] + "..."


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def extract_json_block(text: str) -> Optional[str]:
    """Return the first balanced JSON object in ``text``.

    A brace scanner rather than a regular expression, because a model response may
    contain braces inside string values. String literals and their escapes are
    tracked, so a ``}`` inside a quoted string does not end the object early.

    Args:
        text: The raw model response.

    Returns:
        The substring from the first ``{`` to its matching ``}``, or None when the
        text contains no balanced object.
    """
    if not text:
        return None

    start = -1
    depth = 0
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            if depth == 0:
                # A closing brace before any opening one: nothing to balance.
                return None
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : index + 1]

    return None


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if present.

    Args:
        text: The raw model response.

    Returns:
        The text without its fence.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def load_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[LLMParseError]]:
    """Parse the model's response into a mapping.

    Args:
        text: The raw model response.

    Returns:
        ``(object, None)`` on success, or ``(None, error)`` when the response is
        empty, unparseable, or not a JSON object.
    """
    cleaned = _strip_code_fence(str(text or ""))
    if not cleaned:
        return None, LLMParseError(
            category=LLMErrorCategory.EMPTY_RESPONSE,
            detail="The local model returned an empty response.",
        )

    # Two attempts, in order of specificity. A response may be a bare object, or
    # the object embedded in prose. Extracting it first handles both and keeps
    # trailing commentary out of the payload.
    block = extract_json_block(cleaned)
    candidates = [block] if block is not None else []
    # The whole text is also tried, because a model that was asked for an object
    # and answered with an array or a bare string produced *valid JSON of the wrong
    # shape*. That is a schema violation, and calling it malformed would send the
    # repair request down the wrong path.
    candidates.append(cleaned)

    first_failure: Optional[LLMParseError] = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError) as error:
            if first_failure is None:
                first_failure = LLMParseError(
                    category=LLMErrorCategory.MALFORMED_JSON,
                    detail=(
                        "The response looked like JSON but did not parse: "
                        f"{_flat(error)}."
                    ),
                    raw_excerpt=excerpt(candidate),
                )
            continue

        if not isinstance(payload, dict):
            return None, LLMParseError(
                category=LLMErrorCategory.SCHEMA_VIOLATION,
                detail=(
                    "The response parsed as "
                    f"{type(payload).__name__} but a JSON object was required."
                ),
                raw_excerpt=excerpt(cleaned),
            )
        return payload, None

    if first_failure is None:
        first_failure = LLMParseError(
            category=LLMErrorCategory.MALFORMED_JSON,
            detail=(
                "The response contained no balanced JSON object, so it could not be "
                "parsed. A single request is made to obtain the object again."
            ),
            raw_excerpt=excerpt(cleaned),
        )
    return None, first_failure


def _flat(value: Any, limit: int = 160) -> str:
    """Flatten a value to one bounded line."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# Field coercion
# --------------------------------------------------------------------------- #


def _coerce_text_list(
    value: Any,
    field_name: str,
    notes: List[str],
) -> Tuple[str, ...]:
    """Coerce a value into a tuple of non-empty strings.

    The two coercions performed here are deliberately narrow, and both leave a
    note: a single string becomes a one-item list, and non-string, non-list values
    are stringified rather than dropped, so a model that answers ``{"focus": 3}``
    produces a visible note instead of a silently empty field. Anything structurally
    impossible - a nested object where a list of strings was requested - is a schema
    violation and is reported to the caller.

    Args:
        value: The raw field value.
        field_name: The field name, used in the notes.
        notes: Accumulator for normalisation notes.

    Returns:
        The coerced tuple.

    Raises:
        _SchemaViolation: When the value cannot be a list of strings at all. The
            public entry points convert it back into a returned value.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        notes.append(
            f"'{field_name}' was a single string, not a list; it was read as a "
            "one-item list."
        )
        return (text,)
    if isinstance(value, (int, float, bool)):
        notes.append(
            f"'{field_name}' was a {type(value).__name__}, not a list; it was read as "
            "a one-item list."
        )
        return (str(value),)
    if isinstance(value, Mapping):
        raise _SchemaViolation.of(
            LLMErrorCategory.SCHEMA_VIOLATION,
            f"'{field_name}' was a JSON object but a list of strings was required.",
        )
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for entry in value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    items.append(text)
            elif isinstance(entry, (int, float, bool)):
                items.append(str(entry))
            elif isinstance(entry, Mapping):
                raise _SchemaViolation.of(
                    LLMErrorCategory.SCHEMA_VIOLATION,
                    f"'{field_name}' contained an object; a list of strings was "
                    "required.",
                )
            elif entry is not None:
                items.append(_flat(entry))
        return tuple(items)
    raise _SchemaViolation.of(
        LLMErrorCategory.SCHEMA_VIOLATION,
        f"'{field_name}' had an unsupported type: {type(value).__name__}.",
    )


def _require_summary(payload: Mapping[str, Any], required: bool) -> str:
    """Return the summary, or raise when it is required and unusable.

    Args:
        payload: The parsed object.
        required: True for the interpretation schema, where a summary is the one
            mandatory field.

    Returns:
        The summary text, which may be empty when ``required`` is False.

    Raises:
        _SchemaViolation: When the summary is missing, empty or not a string. The
            public entry points convert it back into a returned value.
    """
    if "summary" not in payload:
        if required:
            raise _SchemaViolation.of(
                LLMErrorCategory.SCHEMA_VIOLATION,
                "The required field 'summary' was missing.",
            )
        return ""
    value = payload["summary"]
    if not isinstance(value, str):
        raise _SchemaViolation.of(
            LLMErrorCategory.SCHEMA_VIOLATION,
            f"'summary' was a {type(value).__name__} but a string was required.",
        )
    text = value.strip()
    if not text and required:
        raise _SchemaViolation.of(
            LLMErrorCategory.SCHEMA_VIOLATION,
            "'summary' was present but empty.",
        )
    return text


def _discard_unknown(payload: Mapping[str, Any], accepted: Sequence[str]) -> List[str]:
    """Return the names of fields the schema does not accept.

    Args:
        payload: The parsed object.
        accepted: The accepted field names.

    Returns:
        The unknown field names, in response order.
    """
    return [str(name) for name in payload if str(name) not in accepted]


def _all_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Join every string value of the given fields, for the phrase check.

    Args:
        payload: The parsed object.
        fields: The accepted field names.

    Returns:
        All relevant text, so a forbidden phrase is caught wherever it appears.
    """
    parts: List[str] = []
    for name in fields:
        value = payload.get(name)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #


def parse_interpretation(payload: Mapping[str, Any], model: str = "") -> Tuple[Optional[LLMInterpretation], Optional[LLMParseError]]:
    """Validate a parsed object against the interpretation schema.

    Args:
        payload: The parsed JSON object.
        model: The model name, recorded on the result.

    Returns:
        ``(interpretation, None)`` on success, or ``(None, error)``. Never raises.
    """
    notes: List[str] = []

    try:
        summary = _require_summary(payload, required=True)
        observations = _coerce_text_list(payload.get("observations"), "observations", notes)
        interpretations = _coerce_text_list(
            payload.get("interpretation"), "interpretation", notes
        )
        hypotheses = _coerce_text_list(payload.get("hypotheses"), "hypotheses", notes)
        recommendations = _coerce_text_list(
            payload.get("recommendations"), "recommendations", notes
        )
    except _SchemaViolation as signal:
        return None, signal.error

    unknown = _discard_unknown(payload, INTERPRETATION_FIELDS)
    if unknown:
        notes.append(
            "Ignored field(s) not in the schema: " + ", ".join(sorted(unknown)) + "."
        )

    if not any((observations, interpretations, hypotheses, recommendations)):
        notes.append(
            "The response contained no observations, interpretation, hypotheses or "
            "recommendations."
        )

    confidence = LLMConfidence.parse(payload.get("confidence"))
    if payload.get("confidence") is not None and confidence is LLMConfidence.UNKNOWN:
        notes.append(
            "The reported confidence "
            f"{_flat(payload.get('confidence'), 40)!r} was not one of "
            f"{', '.join(config.LLM_CONFIDENCE_LEVELS)}; it was recorded as unknown."
        )

    claims = find_forbidden_claims(_all_text(payload, INTERPRETATION_FIELDS))
    if claims:
        notes.append(
            "The text contains phrasing that overstates certainty ("
            + ", ".join(f"'{claim}'" for claim in claims)
            + "); treat it as unreliable."
        )

    return (
        LLMInterpretation(
            summary=summary,
            observations=observations,
            interpretations=interpretations,
            hypotheses=hypotheses,
            recommendations=recommendations,
            confidence=confidence,
            model=model,
            forbidden_claims=claims,
            notes=tuple(notes),
        ),
        None,
    )


# --------------------------------------------------------------------------- #
# Optimization plan
# --------------------------------------------------------------------------- #


def _looks_like_code(text: str) -> bool:
    """Return True when a focus entry looks like code rather than a priority.

    The planner is allowed to say *what* to prioritise. It is not allowed to write
    code, name a parameter API or define a search space, because the deterministic
    search owns all of that.

    Args:
        text: One focus entry.

    Returns:
        True when the entry contains a code marker.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _CODE_MARKERS):
        return True
    return any(pattern.search(lowered) for pattern in _CODE_PATTERNS)


def parse_plan(payload: Mapping[str, Any], model: str = "") -> Tuple[Optional[LLMOptimizationPlan], Optional[LLMParseError]]:
    """Validate a parsed object against the optimisation-plan schema.

    Candidate models are filtered against the project's own registry, so the
    planner cannot introduce a model the system does not support. Focus entries
    that look like code are rejected for the same reason a routing decision cannot
    be taken by the model: the deterministic side must stay in control.

    Args:
        payload: The parsed JSON object.
        model: The model name, recorded on the result.

    Returns:
        ``(plan, None)`` on success, or ``(None, error)``.
    """
    notes: List[str] = []

    try:
        summary = _require_summary(payload, required=False)
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            notes.append("'reason' was not a string; it was recorded as text.")
            reason = _flat(reason)
        candidates = _coerce_text_list(
            payload.get("candidate_models"), "candidate_models", notes
        )
        focus = _coerce_text_list(payload.get("focus"), "focus", notes)
    except _SchemaViolation as signal:
        return None, signal.error

    known_models = {str(name) for name in config.CLASSIFICATION_MODELS}
    accepted_models: List[str] = []
    rejected_models: List[str] = []
    for name in candidates:
        if name in known_models:
            if name not in accepted_models:
                accepted_models.append(name)
        else:
            rejected_models.append(name)

    if rejected_models:
        notes.append(
            "Rejected model name(s) not in this project's registry: "
            + ", ".join(rejected_models)
            + ". The optimization agent keeps its own candidate rule."
        )

    accepted_focus: List[str] = []
    rejected_focus: List[str] = []
    for entry in focus:
        if _looks_like_code(entry):
            rejected_focus.append(entry)
        elif entry not in accepted_focus:
            accepted_focus.append(entry)

    if rejected_focus:
        notes.append(
            "Rejected focus entr(ies) that read as code or as a search-space "
            "directive: "
            + ", ".join(rejected_focus)
            + ". This plan cannot control the search."
        )

    unknown = _discard_unknown(payload, PLAN_FIELDS)
    if unknown:
        notes.append(
            "Ignored field(s) not in the schema: " + ", ".join(sorted(unknown)) + "."
        )

    confidence = LLMConfidence.parse(payload.get("confidence"))
    if payload.get("confidence") is not None and confidence is LLMConfidence.UNKNOWN:
        notes.append(
            "The reported confidence "
            f"{_flat(payload.get('confidence'), 40)!r} was not one of "
            f"{', '.join(config.LLM_CONFIDENCE_LEVELS)}; it was recorded as unknown."
        )

    claims = find_forbidden_claims(_all_text(payload, PLAN_FIELDS))
    if claims:
        notes.append(
            "The text contains phrasing that overstates certainty ("
            + ", ".join(f"'{claim}'" for claim in claims)
            + "); treat it as unreliable."
        )

    return (
        LLMOptimizationPlan(
            summary=summary,
            candidate_models=tuple(accepted_models),
            focus_areas=tuple(accepted_focus),
            reason=reason.strip() if isinstance(reason, str) else "",
            confidence=confidence,
            model=model,
            rejected_models=tuple(rejected_models),
            rejected_focus_areas=tuple(rejected_focus),
            forbidden_claims=claims,
            notes=tuple(notes),
        ),
        None,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParsedLLMResponse:
    """Either a validated record or a typed failure.

    A single return type keeps the service layer free of a tuple unpack at every
    call site, and makes the "no exception ever escapes" contract structural.
    """

    interpretation: Optional[LLMInterpretation] = None
    plan: Optional[LLMOptimizationPlan] = None
    error: Optional[LLMParseError] = None
    notes: Tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Return True when a validated record was produced."""
        return self.error is None

    @property
    def output(self) -> Optional[Any]:
        """Return the validated record, whichever kind it is."""
        return self.interpretation if self.interpretation is not None else self.plan

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "ok": self.ok,
            "interpretation": (
                self.interpretation.to_dict() if self.interpretation else None
            ),
            "plan": self.plan.to_dict() if self.plan else None,
            "error": self.error.to_dict() if self.error else None,
            "notes": list(self.notes),
        }


def parse_response(text: str, role: LLMRole, model: str = "") -> ParsedLLMResponse:
    """Parse and validate a model response for one role.

    Args:
        text: The raw model response.
        role: The role whose schema the response is held to.
        model: The model name, recorded on the result.

    Returns:
        A :class:`ParsedLLMResponse`: a validated record or a typed failure.
    """
    payload, error = load_json_object(text)
    if error is not None:
        return ParsedLLMResponse(error=error)

    assert payload is not None  # guaranteed when error is None
    if role.is_plan:
        plan, error = parse_plan(payload, model=model)
        return ParsedLLMResponse(plan=plan, error=error)

    interpretation, error = parse_interpretation(payload, model=model)
    return ParsedLLMResponse(interpretation=interpretation, error=error)


__all__ = [
    "INTERPRETATION_FIELDS",
    "PLAN_FIELDS",
    "LLMParseError",
    "ParsedLLMResponse",
    "excerpt",
    "extract_json_block",
    "load_json_object",
    "parse_interpretation",
    "parse_plan",
    "parse_response",
]