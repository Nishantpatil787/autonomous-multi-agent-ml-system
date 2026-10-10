"""Central prompt templates and context construction (Phase 8).

Two responsibilities, kept together on purpose because they must agree:

* **Templates.** Every instruction the local model receives is assembled here, in
  one place. Nothing else in the project writes prose to a model, so the wording
  discipline can be reviewed in a single file and cannot drift between call sites.
* **Context.** What the model is allowed to see is decided here, through an
  explicit per-role allowlist (:data:`llm.state.LLM_CONTEXT_FIELDS`). A value that
  no builder produces cannot reach the prompt.

The context deliberately contains **summaries, not data**: shapes and counts,
class shares, metric values and recorded statuses. It never contains raw rows,
file contents, environment variables, credentials or absolute filesystem paths.
:func:`sanitize_value` is the last line of defence for that, and it shortens rather
than drops anything oversized, with an explicit marker so a reader can tell the
difference between a short value and a truncated one.

Prompt generation is deterministic: the same ``AgentState`` and role always
produce the same string. There is no timestamp, no counter and no random content,
which is what makes the prompt content assertable in tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Tuple

from llm.state import LLMRole, allowed_context_fields
from utils import config
from utils.state import AgentState

#: Bounded so a large summary cannot become an unbounded prompt. Oversized text is
#: truncated with :data:`TRUNCATION_MARKER`, never silently dropped.
MAX_VALUE_CHARS = 400

#: Marker appended to a truncated value.
TRUNCATION_MARKER = " ...[truncated]"

#: Upper bound on a single prompt. Comfortably inside a 3B model's context while
#: still leaving room for the instruction block and the response.
MAX_PROMPT_CHARS = 6000

#: Absolute-path shapes that must never be sent. Windows drive paths, UNC paths
#: and anything with more than one slash are reduced to their final segment.
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def sanitize_value(value: Any, max_chars: int = MAX_VALUE_CHARS) -> Any:
    """Reduce a value to something safe and bounded to send to the model.

    Handles the three things that must never leave the process: an absolute
    filesystem path, a container deeper than the prompt budget, and a string long
    enough to dominate the prompt. Booleans, ``None`` and numbers pass through
    unchanged, so measured values are never altered in transit.

    Args:
        value: The value to sanitise.
        max_chars: Maximum length of a resulting string.

    Returns:
        A JSON-safe value: a scalar, or a list of sanitised scalars.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        bounded = list(value)[: config.LLM_CONTEXT_MAX_LIST_ITEMS]
        return [sanitize_value(item, max_chars) for item in bounded]
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_value(item, max_chars)
            for key, item in list(value.items())[: config.LLM_CONTEXT_MAX_LIST_ITEMS]
        }
    if value is None:
        return None

    text = redact_path(str(value))
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max(0, max_chars - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER
    return text


def redact_path(text: str) -> str:
    """Replace an absolute filesystem path with its final segment.

    The model has no use for the directory a dataset happens to live in, and the
    path can reveal a username. Any text that looks like an absolute path is
    therefore reduced to the file name; ordinary prose is returned unchanged.

    Args:
        text: The text to inspect.

    Returns:
        The text with any absolute path replaced by its last segment.
    """
    stripped = text.strip()
    if not stripped:
        return text
    looks_absolute = (
        _WINDOWS_PATH.match(stripped) is not None
        or stripped.startswith("\\\\")
        or stripped.startswith("//")
        or stripped.count("/") > 1
        or stripped.count("\\") > 1
    )
    if not looks_absolute:
        return text
    try:
        name = PureWindowsPath(stripped).name or PurePosixPath(stripped).name
    except (ValueError, IndexError):
        return text
    return name or text


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


def _bullet_list(items: Tuple[str, ...]) -> str:
    """Render a tuple of statements as a bullet block."""
    return "\n".join(f"- {item}" for item in items)


def _build_system_instruction() -> str:
    """Assemble the shared system instruction from the configured wording.

    Every clause comes from :mod:`utils.config`, so the wording is reviewable in
    one place and a test can assert the presence of each rule rather than matching
    a whole blob of text.
    """
    rules = (
        config.LLM_ROLE_STATEMENT,
        config.LLM_AUTHORITY_RULE,
        config.LLM_NO_METRIC_INVENTION,
        config.LLM_NO_STATISTICAL_CLAIM,
        config.LLM_NO_RETRAINING,
        config.LLM_NO_DATA_MODIFICATION,
        config.LLM_OBSERVATION_SEPARATION,
        config.LLM_PRIVACY_RULE,
    )
    forbidden = _bullet_list(config.LLM_FORBIDDEN_CLAIMS)
    preferred = _bullet_list(config.LLM_ALLOWED_CLAIM_WORDS)
    return "\n".join(
        [
            "You are an assistant interpreting results produced by a deterministic "
            "machine-learning pipeline. Treat all supplied metrics as authoritative.",
            "",
            "Rules:",
            _bullet_list(rules),
            "",
            "Never use any of these phrases:",
            forbidden,
            "",
            "Prefer these instead:",
            preferred,
            "",
            "Reply with a single JSON object and nothing else: no prose before or "
            "after it, no markdown code fences, no commentary.",
        ]
    )


#: The system instruction sent with every request. Built once, at import time, from
#: :mod:`utils.config`.
SYSTEM_INSTRUCTION: str = _build_system_instruction()

#: Response contract for the two interpretation roles. Requesting an explicit
#: object, and asking for the three registers separately, is what makes the
#: observed/interpreted/hypothesised distinction structural rather than a hope.
INTERPRETATION_SCHEMA_TEXT = (
    'Return exactly this shape:\n'
    '{\n'
    '  "summary": "two or three sentences describing what the run measured",\n'
    '  "observations": ["a measured value from the context, restated plainly"],\n'
    '  "interpretation": ["a reading of an observation, marked as interpretation"],\n'
    '  "hypotheses": ["a possible explanation that still needs investigation"],\n'
    '  "recommendations": ["a concrete next step for a person to consider"],\n'
    '  "confidence": "low" | "medium" | "high"\n'
    '}\n'
    'Every list may be empty, but "summary" must be a non-empty string.'
)

#: Response contract for the optimisation planner. The candidate models are
#: validated against the project's own registry afterwards, and a focus entry that
#: looks like code or a search space is rejected, so the planner cannot widen what
#: the deterministic search is allowed to do.
PLAN_SCHEMA_TEXT = (
    'Return exactly this shape:\n'
    '{\n'
    '  "summary": "one or two sentences on what is worth tuning",\n'
    '  "candidate_models": ["model names exactly as they appear in the context"],\n'
    '  "focus": ["short phrases, e.g. regularisation strength, class imbalance"],\n'
    '  "reason": "why this focus, in terms of the measured results",\n'
    '  "confidence": "low" | "medium" | "high"\n'
    '}\n'
    'Describe priorities only. Do not write code, do not name Optuna or any '
    'parameter API, and do not define search spaces: the existing search is '
    'already in control.'
)

_ROLE_TASK_TEXT: Dict[LLMRole, str] = {
    LLMRole.DATA_ANALYSIS: (
        "Task: explain this dataset. Describe what the measured shape, feature "
        "types, missing values and class distribution mean for modelling this "
        "dataset. Put every measured value in 'observations', your reading of them "
        "in 'interpretation', and anything unproven in 'hypotheses'."
    ),
    LLMRole.EVALUATION: (
        "Task: explain the model comparison. State which model the recorded "
        "ranking favoured and restate the measured scores in 'observations'. Explain "
        "the hold-out versus cross-validation difference as an interpretation, and "
        "keep every possible cause in 'hypotheses'. Mention the recorded "
        "optimization decision as a fact about the pipeline, not as your own "
        "opinion."
    ),
    LLMRole.OPTIMIZATION: (
        "Task: propose a high-level optimisation plan. This is a priority hint for "
        "an existing hyperparameter search, not a replacement for it. Name "
        "candidate models exactly as the context names them."
    ),
}


def schema_text(role: LLMRole) -> str:
    """Return the response contract for one role.

    Args:
        role: The interpretation step.

    Returns:
        The schema text for that role's expected response.
    """
    return PLAN_SCHEMA_TEXT if role.is_plan else INTERPRETATION_SCHEMA_TEXT


def task_text(role: LLMRole) -> str:
    """Return the task description for one role.

    Args:
        role: The interpretation step.

    Returns:
        The task text for that role.
    """
    return _ROLE_TASK_TEXT[role]


def repair_instruction(role: LLMRole) -> str:
    """Return the instruction for the single controlled repair attempt.

    The repair asks for the same object again with no extra reasoning, which is the
    cheapest way to recover from a truncated or fence-wrapped response. It is
    bounded to one attempt, so a model that cannot comply cannot cause a loop.

    Args:
        role: The interpretation step whose response was unusable.

    Returns:
        The repair instruction.
    """
    return (
        "Your previous reply could not be used because it was not a single valid "
        "JSON object matching the requested shape.\n"
        f"{task_text(role)}\n"
        f"{schema_text(role)}\n"
        "Reply with the JSON object only. No prose, no code fences, no trailing text."
    )


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptContext:
    """The summarised, allowlisted context for one prompt.

    ``fields`` is an ordered tuple of pairs rather than a dict so the rendered
    prompt is byte-for-byte reproducible: dict ordering is an implementation
    detail, a declared tuple is not.
    """

    role: LLMRole
    fields: Tuple[Tuple[str, Any], ...] = ()

    @property
    def field_names(self) -> Tuple[str, ...]:
        """Return the context field names, in prompt order."""
        return tuple(name for name, _ in self.fields)

    def as_dict(self) -> Dict[str, Any]:
        """Return the context as a plain mapping."""
        return {name: value for name, value in self.fields}

    def to_json(self) -> str:
        """Render the context as deterministic JSON.

        Returns:
            A JSON object string with the fields in declared order.
        """
        return json.dumps(
            {name: value for name, value in self.fields},
            indent=2,
            sort_keys=False,
            default=str,
        )

    def is_empty(self) -> bool:
        """Return True when the context carries no field at all."""
        return not self.fields


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a measured value for transmission without changing what it is.

    Rounding to four decimals matches how the reports already print metrics, so
    the number the model reads is the number the user sees.
    """
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _class_distribution(state: AgentState) -> List[Dict[str, Any]]:
    """Summarise the recorded class distribution."""
    rows: List[Dict[str, Any]] = []
    for label, entry in list(state.class_distribution.items())[: config.LLM_CONTEXT_MAX_LIST_ITEMS]:
        rows.append(
            {"label": label, "count": entry.count, "percentage": entry.percentage}
        )
    return rows


def build_data_analysis_context(state: AgentState) -> PromptContext:
    """Summarise the Data Analysis Agent result for the model.

    Only shapes, counts and shares are sent. Column values, row samples and file
    contents never leave the process.

    Args:
        state: The authoritative machine learning state.

    Returns:
        The allowlisted context for the data-analysis role.
    """
    shape = state.dataset_shape
    missing = state.missing_values
    raw: Dict[str, Any] = {
        "dataset_name": state.dataset_name,
        "rows": shape.rows if shape else 0,
        "columns": shape.columns if shape else 0,
        "numerical_feature_count": len(state.numerical_features),
        "categorical_feature_count": len(state.categorical_features),
        "unsupported_column_count": len(state.unsupported_columns),
        "total_missing_values": state.total_missing_values,
        "missing_percentage": missing.total_percentage if missing else 0.0,
        "affected_column_count": missing.affected_columns if missing else 0,
        "duplicate_count": state.duplicate_count,
        "target_column": state.target_column,
        "target_detection_method": state.target_detection_method,
        "target_detection_confidence": state.target_detection_confidence,
        "problem_type": state.problem_type,
        "class_count": len(state.class_distribution),
        "class_distribution": _class_distribution(state),
        "quality_issue_count": len(state.data_quality_issues),
        "issues_by_severity": state.issues_by_severity(),
    }
    return _assemble(LLMRole.DATA_ANALYSIS, raw)


def build_evaluation_context(state: AgentState) -> PromptContext:
    """Summarise the training and evaluation results for the model.

    Args:
        state: The authoritative machine learning state.

    Returns:
        The allowlisted context for the evaluation role.
    """
    report = state.evaluation_report
    best = report.best_model if report is not None else None
    gap = report.gap_analysis if report is not None else None

    scores: List[Dict[str, Any]] = []
    if report is not None:
        for item in report.model_evaluations[: config.LLM_CONTEXT_MAX_LIST_ITEMS]:
            scores.append(
                {
                    "model_name": item.model_name,
                    "holdout_f1": _round(item.f1_score),
                    "holdout_accuracy": _round(item.accuracy),
                    "cv_f1_mean": _round(item.cv_f1_mean),
                    "f1_gap": _round(item.f1_gap),
                    "f1_gap_classification": item.f1_gap_classification,
                    "experiment_status": item.experiment_status,
                }
            )

    raw: Dict[str, Any] = {
        "evaluation_status": state.evaluation_status,
        "dataset_name": state.dataset_name,
        "problem_type": state.problem_type,
        "holdout_samples": report.test_samples if report is not None else 0,
        "baseline_models_evaluated": len(scores),
        "best_baseline_model": best.best_baseline_model if best is not None else "",
        "best_baseline_f1": _round(best.f1_score) if best is not None else None,
        "best_baseline_accuracy": _round(best.accuracy) if best is not None else None,
        "best_baseline_cv_f1": _round(best.cv_f1_mean) if best is not None else None,
        "best_baseline_parameters": best.parameters if best is not None else {},
        "model_scores": scores,
        "holdout_vs_cv_gap": _round(best.f1_gap) if best is not None else None,
        "gap_classification": (
            best.f1_gap_classification if best is not None else "not_available"
        ),
        "execution_time_seconds": _round(state.execution_time, 3),
        "optimization_decision": state.optimization_status,
        "optimization_decision_reason": state.optimization_report.selection_note
        if state.optimization_report is not None
        else "",
    }
    if gap is not None and gap.largest_absolute_gap is not None:
        raw["largest_absolute_gap"] = _round(gap.largest_absolute_gap)
    return _assemble(LLMRole.EVALUATION, raw)


def build_optimization_context(state: AgentState) -> PromptContext:
    """Summarise the optimization result for the planner.

    The context lists the search dimensions that already exist, so the planner can
    prioritise them without being able to invent a new one.

    Args:
        state: The authoritative machine learning state.

    Returns:
        The allowlisted context for the optimisation-planner role.
    """
    report = state.optimization_report
    tuned: List[Dict[str, Any]] = []
    best = report.best_optimized_model if report is not None else None

    if report is not None:
        for result in report.optimization_results[: config.LLM_CONTEXT_MAX_LIST_ITEMS]:
            tuned.append(
                {
                    "model_name": result.model_name,
                    "status": result.status,
                    "baseline_cv_f1": _round(result.baseline_cv_f1_mean),
                    "optimized_cv_f1": _round(result.optimized_cv_f1_mean),
                    "holdout_f1": _round(result.holdout_f1_score),
                    "improvement_direction": result.improvement_direction,
                    "trials": result.total_trials,
                }
            )

    raw: Dict[str, Any] = {
        "optimization_status": state.optimization_status,
        "optimization_decision": state.optimization_status,
        "optimization_decision_reason": report.selection_note if report is not None else "",
        "baseline_best_model": best.baseline_model_name if best is not None else "",
        "tuned_models": tuned,
        "trial_budget_per_model": report.n_trials_per_model if report is not None else 0,
        "objective_metric": report.objective_metric if report is not None else "",
        "available_search_parameters": _search_parameters(report),
        "improvement_observed": best.improvement_direction if best is not None else "",
        "holdout_f1_delta": _round(best.holdout_f1_delta) if best is not None else None,
    }
    return _assemble(LLMRole.OPTIMIZATION, raw)


def _search_parameters(report: Any) -> List[str]:
    """Return the union of search dimensions the existing search already uses.

    Read from the recorded search-space descriptions rather than from the config
    module, so the planner sees what this run actually explored.

    Args:
        report: The recorded optimization report, or None.

    Returns:
        The distinct search-space descriptions, bounded.
    """
    if report is None:
        return []
    seen: List[str] = []
    for result in report.optimization_results:
        for entry in result.search_space_description:
            text = str(entry)
            if text not in seen:
                seen.append(text)
    return seen[: config.LLM_CONTEXT_MAX_LIST_ITEMS]


def _assemble(role: LLMRole, raw: Dict[str, Any]) -> PromptContext:
    """Build a context from a raw mapping, keeping only allowlisted keys.

    A key that is not in the role's allowlist is dropped. That is the enforcement
    point for the privacy boundary: a builder cannot accidentally widen what is
    sent by adding a key to its own mapping.

    Args:
        role: The interpretation step.
        raw: The candidate values, keyed by context field name.

    Returns:
        The sanitised, ordered context.
    """
    allowed = allowed_context_fields(role)
    fields: List[Tuple[str, Any]] = []
    for name in allowed:
        if name not in raw:
            continue
        fields.append((name, sanitize_value(raw[name])))
        if len(fields) >= config.LLM_CONTEXT_MAX_FIELDS:
            break
    return PromptContext(role=role, fields=tuple(fields))


def build_context(state: AgentState, role: LLMRole) -> PromptContext:
    """Build the context for any role.

    Args:
        state: The authoritative machine learning state.
        role: The interpretation step.

    Returns:
        The allowlisted context for that role.
    """
    if role is LLMRole.DATA_ANALYSIS:
        return build_data_analysis_context(state)
    if role is LLMRole.EVALUATION:
        return build_evaluation_context(state)
    return build_optimization_context(state)


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #


def build_prompt(context: PromptContext) -> str:
    """Assemble the user prompt for a context.

    The order is fixed - task, rules, schema, context - so the prompt reads the
    same way every time and stays assertable.

    Args:
        context: The allowlisted context to present.

    Returns:
        The complete user prompt.
    """
    parts = [
        task_text(context.role),
        "",
        "Context (all values below were measured by the deterministic pipeline "
        "and are authoritative):",
        context.to_json(),
        "",
        schema_text(context.role),
    ]
    prompt = "\n".join(parts)
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt

    # The context is the only part that can grow, so it is the part that is
    # shortened. Truncating is announced rather than silent.
    overflow = len(prompt) - MAX_PROMPT_CHARS
    trimmed = context.to_json()
    trimmed = trimmed[: max(0, len(trimmed) - overflow - len(TRUNCATION_MARKER))]
    parts[3] = trimmed + TRUNCATION_MARKER
    return "\n".join(parts)


def build_repair_prompt(context: PromptContext) -> str:
    """Assemble the prompt for the single controlled repair attempt.

    Args:
        context: The same allowlisted context used for the first attempt.

    Returns:
        The repair prompt.
    """
    return "\n".join(
        [
            repair_instruction(context.role),
            "",
            "Context:",
            context.to_json(),
            "",
            schema_text(context.role),
        ]
    )


__all__ = [
    "INTERPRETATION_SCHEMA_TEXT",
    "MAX_PROMPT_CHARS",
    "MAX_VALUE_CHARS",
    "PLAN_SCHEMA_TEXT",
    "PromptContext",
    "SYSTEM_INSTRUCTION",
    "TRUNCATION_MARKER",
    "build_context",
    "build_data_analysis_context",
    "build_evaluation_context",
    "build_optimization_context",
    "build_prompt",
    "build_repair_prompt",
    "redact_path",
    "repair_instruction",
    "sanitize_value",
    "schema_text",
    "task_text",
]