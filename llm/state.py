"""Typed records for the optional local LLM layer (Phase 8).

Everything the language model produces is reduced to plain, frozen dataclasses and
enums here, before it can reach the graph state or a report. Three rules shape
this module:

* **The LLM is never an authority.** No record here can change a metric, a model
  ranking, a search result or a workflow status. The records carry text and a
  status, and nothing else.
* **Nothing live is stored.** There is no HTTP client, no response object and no
  raw model output in any of these types, so they serialise cleanly and can sit in
  the LangGraph state (unlike a scikit-learn estimator or an Optuna ``Study``).
* **A failure is data.** Every unsuccessful call becomes a record with a
  :class:`LLMStatus` and an :class:`LLMErrorCategory`, never an exception that
  escapes into the workflow.

:func:`llm.state.collect_state_keys` is the single helper the graph uses to copy
the machine learning state into a prompt context. It is an explicit allowlist:
a field that is not named there can never reach the model, so adding a new field
to :class:`~utils.state.AgentState` cannot silently start leaking it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Status vocabularies
# --------------------------------------------------------------------------- #


class LLMStatus(str, Enum):
    """Outcome of an optional interpretation step.

    ``DISABLED``
        The layer is off, so no client was built and nothing was requested.
    ``AVAILABLE``
        A health check confirmed that Ollama is reachable *and* the configured
        model is installed. This says nothing about the quality of an answer.
    ``COMPLETED``
        A request was made and a valid structured response was parsed.
    ``UNAVAILABLE``
        The Ollama server could not be reached at all.
    ``MODEL_UNAVAILABLE``
        Ollama answered, but the configured model is not installed locally.
    ``REQUEST_FAILED``
        The request was sent and failed: a timeout, a connection error or an
        HTTP error status.
    ``INVALID_RESPONSE``
        The model answered, but the text was not usable: unparseable JSON, or
        JSON that failed validation.
    """

    DISABLED = "disabled"
    AVAILABLE = "available"
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    REQUEST_FAILED = "request_failed"
    INVALID_RESPONSE = "invalid_response"

    @property
    def is_failure(self) -> bool:
        """Return True when the step produced no usable interpretation."""
        return self in (
            LLMStatus.UNAVAILABLE,
            LLMStatus.MODEL_UNAVAILABLE,
            LLMStatus.REQUEST_FAILED,
            LLMStatus.INVALID_RESPONSE,
        )

    @property
    def produced_output(self) -> bool:
        """Return True only when generated text is attached to the result."""
        return self is LLMStatus.COMPLETED


class LLMErrorCategory(str, Enum):
    """Why an interpretation step produced nothing.

    The categories are deliberately specific: "the LLM failed" is not actionable,
    whereas "the server is not running" and "the model is not installed" call for
    completely different responses from the user.
    """

    NONE = "none"
    NOT_CONFIGURED = "not_configured"
    UNSUPPORTED_HOST = "unsupported_host"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    UNEXPECTED_RESPONSE = "unexpected_response"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VIOLATION = "schema_violation"
    EMPTY_RESPONSE = "empty_response"
    FORBIDDEN_CLAIM = "forbidden_claim"


class LLMConfidence(str, Enum):
    """Self-reported confidence of a generated interpretation.

    This is the model's own statement about its text and nothing more. It is
    stored as a typed value so a report can print it, but it is never treated as
    a probability and never gates a decision.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: Any) -> "LLMConfidence":
        """Return the matching confidence, or ``UNKNOWN``.

        A model that returns an unrecognised label must not be able to widen its
        own reported confidence, and it must not crash the run either.

        Args:
            value: Whatever the model produced for the ``confidence`` field.

        Returns:
            The matching member, or :attr:`UNKNOWN`.
        """
        text = str(value or "").strip().lower()
        for member in (cls.LOW, cls.MEDIUM, cls.HIGH):
            if text == member.value:
                return member
        return cls.UNKNOWN


class LLMRole(str, Enum):
    """Which interpretation step produced a record.

    The role fixes the context that is sent and the response schema that is
    expected, so a record can always be traced back to the step that made it.
    """

    DATA_ANALYSIS = "data_analysis"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"

    @property
    def display_name(self) -> str:
        """Return the human readable name used in logs and reports."""
        return LLM_ROLE_DISPLAY_NAMES[self]

    @property
    def is_plan(self) -> bool:
        """Return True when the role expects an optimisation plan.

        The optimisation planner answers with a different schema from the two
        interpretation steps: a plan is a priority hint, not prose about results.
        """
        return self is LLMRole.OPTIMIZATION


LLM_ROLE_DISPLAY_NAMES: Dict[LLMRole, str] = {
    LLMRole.DATA_ANALYSIS: "Data Analysis Interpretation",
    LLMRole.EVALUATION: "Evaluation Interpretation",
    LLMRole.OPTIMIZATION: "Optimization Plan",
}


# --------------------------------------------------------------------------- #
# The prompt context allowlist
# --------------------------------------------------------------------------- #

#: Role to the ``AgentState`` fields that may be summarised for the model. This
#: is the whole privacy boundary of Phase 8 in one table: a value that is not
#: produced by one of these builders cannot reach the prompt.
LLM_CONTEXT_FIELDS: Dict[LLMRole, Tuple[str, ...]] = {
    LLMRole.DATA_ANALYSIS: (
        "dataset_name",
        "rows",
        "columns",
        "numerical_feature_count",
        "categorical_feature_count",
        "unsupported_column_count",
        "total_missing_values",
        "missing_percentage",
        "affected_column_count",
        "duplicate_count",
        "target_column",
        "target_detection_method",
        "target_detection_confidence",
        "problem_type",
        "class_count",
        "class_distribution",
        "quality_issue_count",
        "issues_by_severity",
    ),
    LLMRole.EVALUATION: (
        "evaluation_status",
        "dataset_name",
        "problem_type",
        "holdout_samples",
        "baseline_models_evaluated",
        "best_baseline_model",
        "best_baseline_f1",
        "best_baseline_accuracy",
        "best_baseline_cv_f1",
        "best_baseline_parameters",
        "model_scores",
        "holdout_vs_cv_gap",
        "gap_classification",
        "largest_absolute_gap",
        "execution_time_seconds",
        "optimization_decision",
        "optimization_decision_reason",
    ),
    LLMRole.OPTIMIZATION: (
        "optimization_status",
        "optimization_decision",
        "optimization_decision_reason",
        "baseline_best_model",
        "tuned_models",
        "trial_budget_per_model",
        "objective_metric",
        "available_search_parameters",
        "improvement_observed",
        "holdout_f1_delta",
    ),
}


def allowed_context_fields(role: LLMRole) -> Tuple[str, ...]:
    """Return the context field names allowed for one role.

    Args:
        role: The interpretation step.

    Returns:
        The allowed field names, in a stable order.
    """
    return LLM_CONTEXT_FIELDS[role]


# --------------------------------------------------------------------------- #
# Typed records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMResponse:
    """A raw, untrusted answer from the local model.

    Held only for the lifetime of one request so the parser can inspect it. It is
    never stored in the graph state: :class:`LLMInterpretation` is what survives.
    """

    text: str
    model: str
    duration: float = 0.0
    attempts: int = 1
    repaired: bool = False

    @property
    def is_empty(self) -> bool:
        """Return True when the model returned nothing usable at all."""
        return not self.text or not self.text.strip()


@dataclass(frozen=True)
class LLMHealth:
    """The result of a local availability check."""

    status: LLMStatus
    detail: str = ""
    model: str = ""
    available_models: Tuple[str, ...] = ()
    duration: float = 0.0

    @property
    def available(self) -> bool:
        """Return True when the checked service or model can be used."""
        return self.status is LLMStatus.AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "status": self.status.value,
            "detail": self.detail,
            "model": self.model,
            "available_models": list(self.available_models),
            "duration": self.duration,
        }


@dataclass(frozen=True)
class LLMInterpretation:
    """Validated structured output from one interpretation step.

    The three lists are the three registers the project has used since Phase 5:
    ``observations`` restate measured values, ``interpretations`` are readings of
    them, and ``hypotheses`` are unverified possibilities. Keeping them in
    separate fields is what makes it impossible to present one as another by
    accident.
    """

    summary: str = ""
    observations: Tuple[str, ...] = ()
    interpretations: Tuple[str, ...] = ()
    hypotheses: Tuple[str, ...] = ()
    recommendations: Tuple[str, ...] = ()
    confidence: LLMConfidence = LLMConfidence.UNKNOWN
    model: str = ""
    #: Phrases from ``config.LLM_FORBIDDEN_CLAIMS`` that appeared in the text.
    #: Recorded rather than removed: the text is never silently rewritten, and a
    #: violation is reported so it cannot pass as a validated claim.
    forbidden_claims: Tuple[str, ...] = ()
    #: Normalisations the parser had to apply, e.g. a scalar coerced to a list.
    notes: Tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return True when the interpretation carries no content at all."""
        return not self.summary and not any(
            (
                self.observations,
                self.interpretations,
                self.hypotheses,
                self.recommendations,
            )
        )

    @property
    def has_forbidden_claims(self) -> bool:
        """Return True when the text overstated certainty somewhere."""
        return bool(self.forbidden_claims)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "summary": self.summary,
            "observations": list(self.observations),
            "interpretations": list(self.interpretations),
            "hypotheses": list(self.hypotheses),
            "recommendations": list(self.recommendations),
            "confidence": self.confidence.value,
            "model": self.model,
            "forbidden_claims": list(self.forbidden_claims),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LLMOptimizationPlan:
    """A validated optimisation *suggestion*.

    The deterministic search stays in charge: the plan cannot widen a search
    space, add a model the registry does not contain, or influence which models
    were selected for tuning. It is recorded so a reader can see what the model
    suggested, and nothing more.
    """

    summary: str = ""
    #: Only models that exist in the project's own registry survive validation.
    candidate_models: Tuple[str, ...] = ()
    #: Short phrases describing what to prioritise. Code-like or search-space
    #: directives are rejected here rather than forwarded anywhere.
    focus_areas: Tuple[str, ...] = ()
    reason: str = ""
    confidence: LLMConfidence = LLMConfidence.UNKNOWN
    model: str = ""
    rejected_models: Tuple[str, ...] = ()
    rejected_focus_areas: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return True when the plan carries no usable suggestion."""
        return not self.summary and not self.candidate_models and not self.focus_areas

    @property
    def has_forbidden_claims(self) -> bool:
        """Return True when the text overstated certainty somewhere."""
        return bool(self.forbidden_claims)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "summary": self.summary,
            "candidate_models": list(self.candidate_models),
            "focus_areas": list(self.focus_areas),
            "reason": self.reason,
            "confidence": self.confidence.value,
            "model": self.model,
            "rejected_models": list(self.rejected_models),
            "rejected_focus_areas": list(self.rejected_focus_areas),
            "forbidden_claims": list(self.forbidden_claims),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LLMResult:
    """The outcome of one optional interpretation step.

    This is the only LLM type that reaches the graph state. It always exists,
    even when the layer is disabled, so a caller never has to test for ``None``
    to find out what happened.

    An unsuccessful result carries no interpretation and no plan: there is nothing
    to show, and an empty one could be mistaken for "the model found nothing to
    say" rather than "the model was unavailable".
    """

    role: LLMRole
    status: LLMStatus
    model: str = ""
    detail: str = ""
    error_category: LLMErrorCategory = LLMErrorCategory.NONE
    duration: float = 0.0
    attempts: int = 0
    repaired: bool = False
    context_fields: Tuple[str, ...] = ()
    interpretation: Optional[LLMInterpretation] = None
    plan: Optional[LLMOptimizationPlan] = None

    @property
    def succeeded(self) -> bool:
        """Return True when a validated response was produced."""
        return self.status is LLMStatus.COMPLETED

    @property
    def output(self) -> Optional[Any]:
        """Return the interpretation or the plan, whichever this role produced.

        Returns:
            The validated output, or None when the step produced nothing.
        """
        return self.interpretation if self.interpretation is not None else self.plan

    @property
    def failed(self) -> bool:
        """Return True when the step produced no usable output.

        :attr:`~LLMStatus.DISABLED` is not a failure: nothing was attempted, so
        there is nothing to recover from and nothing is degraded.
        """
        return self.status.is_failure

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation.

        Contains no client object and no raw model text, only validated content
        and the status that explains why it may be missing.
        """
        return {
            "role": self.role.value,
            "role_display_name": self.role.display_name,
            "status": self.status.value,
            "model": self.model,
            "detail": self.detail,
            "error_category": self.error_category.value,
            "duration": self.duration,
            "attempts": self.attempts,
            "repaired": self.repaired,
            "context_fields": list(self.context_fields),
            "interpretation": (
                self.interpretation.to_dict() if self.interpretation else None
            ),
            "plan": self.plan.to_dict() if self.plan else None,
            "forbidden_claims": self.forbidden_claims,
        }

    @property
    def forbidden_claims(self) -> List[str]:
        """Return every forbidden phrase found, across interpretation and plan."""
        found: List[str] = []
        for payload in (self.interpretation, self.plan):
            if payload is not None:
                found.extend(payload.forbidden_claims)
        return found


# --------------------------------------------------------------------------- #
# Phrase detection
# --------------------------------------------------------------------------- #


def _forbidden_phrases() -> Tuple[str, ...]:
    """Return the configured forbidden phrases, imported lazily.

    The import is inside the function so this module stays importable before
    :mod:`utils.config` has finished setting up the logging directories.
    """
    from utils import config

    return tuple(config.LLM_FORBIDDEN_CLAIMS)


def find_forbidden_claims(text: str) -> Tuple[str, ...]:
    """Return the configured forbidden phrases present in ``text``.

    Args:
        text: Generated text to inspect. Case-insensitive.

    Returns:
        The phrases found, in the order they appear in the configuration and
        without duplicates. Empty when the text is clean.
    """
    if not text:
        return ()
    lowered = text.lower()
    return tuple(
        phrase for phrase in _forbidden_phrases() if phrase.lower() in lowered
    )


def collect_state_keys(*names: str) -> Tuple[str, ...]:
    """Return the given field names as a stable, de-duplicated tuple.

    Args:
        *names: Field names collected from an allowlist table.

    Returns:
        The names in order, without repeats.
    """
    seen: Dict[str, None] = {}
    for name in names:
        seen.setdefault(name, None)
    return tuple(seen)


#: Re-exported so callers can build a context without importing three modules.
__all__ = [
    "LLM_CONTEXT_FIELDS",
    "LLM_ROLE_DISPLAY_NAMES",
    "LLMConfidence",
    "LLMErrorCategory",
    "LLMHealth",
    "LLMInterpretation",
    "LLMOptimizationPlan",
    "LLMResponse",
    "LLMResult",
    "LLMRole",
    "LLMStatus",
    "allowed_context_fields",
    "collect_state_keys",
    "find_forbidden_claims",
]