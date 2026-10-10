"""Dashboard data layer for the Streamlit app (Phase 9).

Everything the dashboard displays is prepared here, and **none of it is computed
here**. This module reads the three artifacts that
:mod:`tools.persistence_tools` wrote and turns them into the small, plain values
a view needs: a status row, a comparison table, a chart series, a prediction
result.

The split exists for two reasons:

* ``dashboard.py`` is the only module that imports ``streamlit``, so the logic
  that shapes the data is testable with the ordinary test runner and no Streamlit
  runtime is needed to check it;
* this module imports no agent and no graph module, so the dashboard cannot
  accidentally start a workflow. It reads files; it does not run the pipeline.

Consequences that are deliberate and worth stating:

* no training, no hyperparameter search, no evaluation, no splitting;
* no imputation, scaling or encoding is duplicated here. A prediction is one
  ``pipeline.predict`` call, because the fitted pipeline in the artifact already
  contains the preprocessor that Phase 3 built;
* no ``joblib.load`` call appears here. Loading goes through
  :func:`tools.persistence_tools.load_persisted_model`, which only ever loads the
  artifact this project wrote;
* every function returns data and never raises for a missing artifact. The
  dashboard has to start when nothing has been persisted yet, and the only way to
  guarantee that is to make "no artifact" a value rather than an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from tools.persistence_tools import (
    ArtifactPaths,
    ArtifactReadError,
    PersistedModel,
    artifact_paths,
    load_persisted_model,
    load_run_results,
)
from utils import config

#: Value shown wherever a number was never recorded. Chosen over 0.0 because a
#: zero would be read as a real measurement, and over None because a table cell
#: renders None as an unhelpful blank.
NOT_RECORDED = "-"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_metric(value: Optional[float], digits: int = 4) -> str:
    """Return a metric as a display string.

    Args:
        value: The recorded value, or None when it was never recorded.
        digits: Decimal places, matching the CLI reports.

    Returns:
        A fixed-point string, or :data:`NOT_RECORDED` for a missing value. The
        missing case is deliberately a visible marker rather than a plausible
        number: a metric that does not exist must not look like one that does.
    """
    if value is None:
        return NOT_RECORDED
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return NOT_RECORDED


def format_parameter(value: Any) -> str:
    """Return one hyperparameter value as a display string.

    Args:
        value: A value taken from a recorded ``best_params`` mapping.

    Returns:
        ``str(value)`` for anything that is not a floating point number, and a
        6-significant-digit string for one. The extra precision matters for a
        parameter like ``C=14.5663...``, where 4 decimals would hide the value that
        was actually searched over.
    """
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_parameters(parameters: Optional[Dict[str, Any]]) -> str:
    """Return a parameter mapping as one readable line, keys sorted.

    Sorting is what makes two runs of the same pipeline produce byte-identical
    text, which matters because these strings end up in a report a reader may
    diff.

    Args:
        parameters: The recorded parameters, or None.

    Returns:
        ``"key=value, key=value"``, or :data:`NOT_RECORDED` when empty.
    """
    if not parameters:
        return NOT_RECORDED
    return ", ".join(
        f"{key}={format_parameter(parameters[key])}" for key in sorted(parameters)
    )


# --------------------------------------------------------------------------- #
# The loaded dashboard state
# --------------------------------------------------------------------------- #
@dataclass
class DashboardState:
    """Everything one dashboard render needs, already loaded.

    A single object, so the view never has to decide for itself whether an
    artifact is present. ``available`` is the one flag the view branches on, and
    it is False in exactly one situation: there is no usable artifact to show.

    ``problem`` carries the reason when loading failed, which is a different
    situation from "no artifact exists" and deserves a different message: the
    first means run the workflow, the second means the artifacts are damaged.
    """

    available: bool = False
    model: Optional[PersistedModel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    paths: Optional[ArtifactPaths] = None
    problem: str = ""

    # -- convenience accessors -------------------------------------------- #
    @property
    def results_available(self) -> bool:
        """True when the report bundle was read, not just the model."""
        return bool(self.results)

    @property
    def dataset(self) -> Dict[str, Any]:
        """The Phase 2 dataset section, empty when the bundle is absent."""
        return dict(self.results.get("dataset") or {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """The Phase 5 evaluation section, empty when the bundle is absent."""
        return dict(self.results.get("evaluation") or {})

    @property
    def optimization(self) -> Dict[str, Any]:
        """The Phase 6 optimization section, empty when the bundle is absent."""
        return dict(self.results.get("optimization") or {})

    @property
    def final_model(self) -> Dict[str, Any]:
        """The Phase 9 selection section, empty when the bundle is absent."""
        return dict(self.results.get("final_model") or {})

    @property
    def baseline_rows(self) -> List[Dict[str, Any]]:
        """The Phase 4 comparison rows."""
        return list(self.results.get("baseline_models") or [])

    @property
    def optimized_rows(self) -> List[Dict[str, Any]]:
        """The Phase 6 tuned-model rows."""
        return list(self.results.get("optimized_models") or [])

    @property
    def llm(self) -> Dict[str, Any]:
        """The Phase 8 interpretation section."""
        return dict(self.results.get("llm") or {})

    @property
    def model_name(self) -> str:
        """Name of the persisted model, from whichever source recorded it."""
        return str(
            self.final_model.get("model_name")
            or self.metadata.get("model_name")
            or (self.model.model_name if self.model is not None else "")
            or NOT_RECORDED
        )

    @property
    def target_column(self) -> str:
        """Name of the column the persisted model predicts."""
        return str(
            self.metadata.get("target_column")
            or self.dataset.get("target_column")
            or (self.model.target_column if self.model is not None else "")
            or NOT_RECORDED
        )

    @property
    def task_type(self) -> str:
        """The recorded task type, e.g. ``"binary_classification"``."""
        return str(
            self.metadata.get("task_type")
            or self.dataset.get("problem_type")
            or NOT_RECORDED
        )

    @property
    def class_labels(self) -> List[str]:
        """Class labels, taken from the metadata the pipeline was saved with."""
        if self.metadata.get("class_labels"):
            return [str(item) for item in self.metadata["class_labels"]]
        if self.model is not None:
            return self.model.class_labels
        return []

    def feature_columns(self) -> List[str]:
        """Return every column the persisted pipeline consumes."""
        if self.metadata.get("feature_columns"):
            return [str(item) for item in self.metadata["feature_columns"]]
        if self.model is not None:
            return self.model.feature_columns
        return []

    def numerical_features(self) -> List[str]:
        """Return the numerical feature names the pipeline expects."""
        if self.metadata.get("numerical_features"):
            return [str(item) for item in self.metadata["numerical_features"]]
        return self.model.numerical_features if self.model is not None else []

    def categorical_features(self) -> List[str]:
        """Return the categorical feature names the pipeline expects."""
        if self.metadata.get("categorical_features"):
            return [str(item) for item in self.metadata["categorical_features"]]
        return self.model.categorical_features if self.model is not None else []


def load_dashboard_state(output_dir: Optional[Path] = None) -> DashboardState:
    """Load the artifacts the dashboard displays, and never raise.

    The dashboard must start whether or not a workflow has ever been run, so this
    function turns every failure into a value:

    * no artifact at all -> ``available=False``, ``problem=""``;
    * an artifact that cannot be read or does not match its metadata ->
      ``available=False``, ``problem=<the reason>``.

    The two are kept apart on purpose. The first is the expected state of a fresh
    checkout and needs one line of guidance. The second means something was
    replaced or truncated, and telling the user to "run the workflow again" is a
    different instruction.

    Args:
        output_dir: Directory holding the artifacts, or None for the configured
            one. It names a *directory*, never a model file: there is no argument
            through which a caller could point this project at an arbitrary file.

    Returns:
        A :class:`DashboardState`. Never raises.
    """
    state = DashboardState()
    try:
        state.paths = artifact_paths(output_dir)
    except Exception as error:  # noqa: BLE001 - a bad config must not kill the app
        state.problem = f"The artifact directory could not be resolved: {error}"
        return state

    if not state.paths.model.is_file():
        return state

    try:
        model = load_persisted_model(output_dir)
    except ArtifactReadError as error:
        state.problem = str(error)
        return state
    except Exception as error:  # noqa: BLE001 - defensive, reported not hidden
        state.problem = (
            f"{config.DASHBOARD_INVALID_ARTIFACT} ({type(error).__name__}: {error})"
        )
        return state

    if model is None:  # pragma: no cover - guarded by the is_file() check above
        return state

    state.model = model
    state.metadata = dict(model.metadata)
    state.available = True

    try:
        state.results = load_run_results(output_dir)
    except ArtifactReadError as error:
        # The model loaded, so the dashboard can still show it and predict with
        # it. Only the report sections are lost, and the note says so.
        state.problem = str(error)

    return state

# --------------------------------------------------------------------------- #
# Section 2: system status
# --------------------------------------------------------------------------- #
def build_status_section(state: DashboardState) -> List[Dict[str, str]]:
    """Return the "is the system ready" rows.

    Every row is answered from recorded state, including the two that describe
    optional features. "LLM interpretation" is answered with what was stored, not
    with a health check: the dashboard does not contact Ollama, so the answer is
    always a fact about the last run rather than a guess about the present.

    Args:
        state: The loaded dashboard state.

    Returns:
        A list of ``{"label": ..., "value": ...}`` rows. Empty when no artifact is
        available, in which case the view shows the "no artifact" message instead.
    """
    if not state.available:
        return []

    optimization = state.optimization
    final = state.final_model
    llm = state.llm

    if optimization:
        optimization_status = str(optimization.get("overall_status") or NOT_RECORDED)
    else:
        optimization_status = "not run"

    llm_results = list(llm.get("results") or [])
    if not llm_results:
        llm_status = "not used"
    else:
        succeeded = [item for item in llm_results if item.get("status") == "completed"]
        llm_status = f"{len(succeeded)}/{len(llm_results)} generated"

    selection_source = str(final.get("selection_source") or NOT_RECORDED)
    if selection_source == "optimized":
        selection_source = "optimized (Phase 6 winner)"
    elif selection_source == "baseline":
        selection_source = "baseline (Phase 5 winner)"

    return [
        {"label": "Persisted model available", "value": "yes"},
        {"label": "Selected model", "value": state.model_name},
        {"label": "Selection source", "value": selection_source},
        {"label": "Task type", "value": state.task_type},
        {"label": "Target column", "value": state.target_column},
        {"label": "Optimization", "value": optimization_status},
        {"label": "LLM interpretation", "value": llm_status},
    ]


# --------------------------------------------------------------------------- #
# Section 3: dataset overview
# --------------------------------------------------------------------------- #
def _issue_summary(dataset: Dict[str, Any]) -> str:
    """Render the Phase 2 issue counts as a single readable phrase.

    The breakdown is a nested mapping, and interpolating it directly would put a
    Python dict repr - braces, quotes and all - into a table cell.

    Args:
        dataset: The recorded dataset section.

    Returns:
        For example ``"6 (4 info, 2 warning, 0 error)"``.
    """
    total = dataset.get("issue_count")
    if total is None:
        return NOT_RECORDED
    breakdown = dataset.get("issues_by_severity") or {}
    if not isinstance(breakdown, dict) or not breakdown:
        return str(total)
    parts = [
        f"{count} {severity}"
        for severity, count in breakdown.items()
        if count is not None
    ]
    return f"{total} ({', '.join(parts)})" if parts else str(total)


def build_dataset_section(state: DashboardState) -> Dict[str, Any]:
    """Return the Phase 2 dataset facts, already shaped for display.

    Nothing is recomputed from the CSV: the row count, the missing-value totals
    and the class distribution are all copies of what the Data Analysis Agent
    recorded during the run that wrote the artifacts. That is the reason the
    dashboard never has to open a dataset.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``metrics`` (label/value rows), ``class_distribution``
        (one row per class), ``numerical_features``, ``categorical_features``,
        ``excluded_features`` and ``available``.
    """
    dataset = state.dataset
    if not dataset:
        return {"available": False}

    metrics = [
        {"label": "Dataset", "value": str(dataset.get("name") or NOT_RECORDED)},
        {"label": "Rows", "value": str(dataset.get("rows", NOT_RECORDED))},
        {"label": "Columns", "value": str(dataset.get("columns", NOT_RECORDED))},
        {
            "label": "Numerical features",
            "value": str(dataset.get("numerical_feature_count", NOT_RECORDED)),
        },
        {
            "label": "Categorical features",
            "value": str(dataset.get("categorical_feature_count", NOT_RECORDED)),
        },
        {
            "label": "Missing values",
            "value": str(dataset.get("total_missing_values", NOT_RECORDED)),
        },
        {
            "label": "Columns with missing values",
            "value": str(dataset.get("affected_missing_columns", NOT_RECORDED)),
        },
        {"label": "Duplicate rows", "value": str(dataset.get("duplicate_count", NOT_RECORDED))},
        {
            "label": "Target column",
            "value": str(dataset.get("target_column") or NOT_RECORDED),
        },
        {
            "label": "Target detection",
            "value": str(
                dataset.get("target_detection_method") or NOT_RECORDED
            )
            + (
                f" (confidence {dataset['target_detection_confidence']})"
                if dataset.get("target_detection_confidence")
                else ""
            ),
        },
        {"label": "Data quality issues", "value": _issue_summary(dataset)},
    ]

    distribution = [
        {
            "class": label,
            "count": item.get("count"),
            "percentage": item.get("percentage"),
        }
        for label, item in sorted(
            (dataset.get("class_distribution") or {}).items(), key=lambda pair: str(pair[0])
        )
    ]

    return {
        "available": True,
        "metrics": metrics,
        "class_distribution": distribution,
        "numerical_features": list(dataset.get("numerical_features") or []),
        "categorical_features": list(dataset.get("categorical_features") or []),
        "excluded_features": list(dataset.get("excluded_features") or []),
        "problem_type_reason": str(dataset.get("problem_type_reason") or ""),
        "summary_text": str(dataset.get("summary_text") or ""),
    }


# --------------------------------------------------------------------------- #
# Section 4: model comparison
# --------------------------------------------------------------------------- #
def build_comparison_section(state: DashboardState) -> Dict[str, Any]:
    """Return the baseline and optimized comparison tables.

    BASELINE and OPTIMIZED rows are kept in one table with an explicit ``Kind``
    column rather than merged into a single ranking. They are not comparable in
    the same way: the baseline numbers come from the Phase 4 hold-out evaluation,
    the optimized numbers from the Phase 6 post-search hold-out score, and Phase 6
    does not record precision, recall or a confusion matrix for a tuned model. A
    single ranked list would imply those are interchangeable, and they are not.

    The ``Persisted`` column marks exactly one row. The same model name can
    legitimately appear twice - once as its Phase 4 baseline and once as its
    Phase 6 tuned version - so the mark is placed by matching the model name
    *and* the recorded ``selection_source``, not the name alone. Marking by name
    alone would flag two different fitted models as the same artifact.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``rows``, ``f1_chart``, ``baseline_count``,
        ``optimized_count``, ``persisted_label``, ``persisted_note`` and
        ``available``.
    """
    if not state.available:
        return {"available": False, "rows": [], "f1_chart": []}

    final = state.final_model
    selected_name = str(final.get("model_name") or state.model_name)
    selected_source = str(final.get("selection_source") or "baseline")

    def is_persisted(name: str, kind: str) -> bool:
        """True for the single row that matches the recorded selection.

        Args:
            name: The model name on the row.
            kind: ``"BASELINE"`` or ``"OPTIMIZED"``.

        Returns:
            True when both the name and the kind match the selection.
        """
        return name == selected_name and kind.lower() == selected_source.lower()

    rows: List[Dict[str, Any]] = []
    for entry in state.baseline_rows:
        name = str(entry.get("model_name") or NOT_RECORDED)
        rows.append(
            {
                "Model": name,
                "Kind": "BASELINE",
                "Accuracy": format_metric(entry.get("accuracy")),
                "Precision": format_metric(entry.get("precision")),
                "Recall": format_metric(entry.get("recall")),
                "F1": format_metric(entry.get("f1_score")),
                "CV F1": format_metric(entry.get("cv_f1_mean")),
                "Rank": str(entry.get("rank", NOT_RECORDED)),
                "Persisted": "yes" if is_persisted(name, "BASELINE") else "",
                "Status": "Phase 4 hold-out evaluation",
            }
        )

    for entry in state.optimized_rows:
        name = str(entry.get("model_name") or NOT_RECORDED)
        if str(entry.get("optimization_status") or "") != "completed":
            # A model the search never finished has no optimized hold-out score.
            # It is listed, with the reason, rather than shown as a zero.
            rows.append(
                {
                    "Model": name,
                    "Kind": "OPTIMIZED",
                    "Accuracy": NOT_RECORDED,
                    "Precision": NOT_RECORDED,
                    "Recall": NOT_RECORDED,
                    "F1": format_metric(entry.get("baseline_f1_score")),
                    "CV F1": format_metric(entry.get("optimized_cv_f1_mean")),
                    "Rank": "not ranked",
                    "Persisted": "",
                    "Status": str(entry.get("reason") or entry.get("error_message") or ""),
                }
            )
            continue
        rows.append(
            {
                "Model": name,
                "Kind": "OPTIMIZED",
                "Accuracy": NOT_RECORDED,
                "Precision": NOT_RECORDED,
                "Recall": NOT_RECORDED,
                "F1": format_metric(entry.get("optimized_f1_score")),
                "CV F1": format_metric(entry.get("optimized_cv_f1_mean")),
                "Rank": str(entry.get("improvement_direction") or "search completed"),
                "Persisted": "yes" if is_persisted(name, "OPTIMIZED") else "",
                "Status": "Phase 6 post-search hold-out score",
            }
        )

    # One bar per (model, kind). A tuned model shares its name with its baseline,
    # so the name alone is not a unique axis label; ``Kind`` is the grouping
    # Streamlit draws the two bars with, which keeps them legible as a pair.
    chart: List[Dict[str, Any]] = []
    for entry in state.baseline_rows:
        chart.append(
            {
                "Model": str(entry.get("model_name") or NOT_RECORDED),
                "F1": entry.get("f1_score"),
                "Kind": "Baseline (Phase 4)",
            }
        )
    for entry in state.optimized_rows:
        if str(entry.get("optimization_status") or "") != "completed":
            continue
        chart.append(
            {
                "Model": str(entry.get("model_name") or NOT_RECORDED),
                "F1": entry.get("optimized_f1_score"),
                "Kind": "Optimized (Phase 6)",
            }
        )

    persisted_rows = [row for row in rows if row["Persisted"] == "yes"]
    persisted_label = (
        f"{persisted_rows[0]['Model']} ({persisted_rows[0]['Kind']})"
        if persisted_rows
        else selected_name
    )

    return {
        "available": True,
        "rows": rows,
        "f1_chart": chart,
        "baseline_count": len(state.baseline_rows),
        "optimized_count": len(state.optimized_rows),
        "persisted_label": persisted_label,
        "persisted_note": (
            "Precision, recall and accuracy are shown for BASELINE rows only: Phase 4 "
            "evaluates the untuned model and records them, while Phase 6 records only "
            "f1_score and accuracy for a tuned one. A '-' means the value was never "
            "measured, not that it was zero."
        ),
    }


# --------------------------------------------------------------------------- #
# Section 5: evaluation
# --------------------------------------------------------------------------- #
def build_evaluation_section(state: DashboardState) -> Dict[str, Any]:
    """Return the Phase 5 evaluation of the *selected* model.

    The metrics come from ``final_model.metrics`` rather than from the evaluation
    table, because the selected model is often not the Phase 5 winner: when Phase 6
    won, the evaluation table would describe a different model. The provenance is
    carried along in ``metrics_source`` and shown, so a reader can tell a Phase 4
    measurement from a Phase 6 one.

    The generalization gap is the cross-validation/hold-out difference the Phase 5
    agent already computed. It is copied, not recomputed, and it is reported as
    unavailable when Phase 4 recorded no training-set metrics - which is the normal
    case in this project.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``metrics``, ``cv_metrics``, ``metrics_source``,
        ``generalization_gap``, ``interpretation`` and ``available``.
    """
    final = state.final_model
    evaluation = state.evaluation
    if not state.available:
        return {"available": False}

    metrics = dict(final.get("metrics") or evaluation.get("best_model") or {})
    cv_metrics = dict(final.get("cv_metrics") or {})

    metric_rows = [
        {"label": "Accuracy", "value": format_metric(metrics.get("accuracy"))},
        {"label": "Precision", "value": format_metric(metrics.get("precision"))},
        {"label": "Recall", "value": format_metric(metrics.get("recall"))},
        {"label": "F1", "value": format_metric(metrics.get("f1_score"))},
    ]

    cv_rows = [
        {
            "label": "Cross-validated F1 mean",
            "value": format_metric(cv_metrics.get("cv_f1_mean")),
        },
        {
            "label": "Cross-validated accuracy mean",
            "value": format_metric(cv_metrics.get("cv_accuracy_mean")),
        },
        {
            "label": "Folds",
            "value": str(cv_metrics.get("cv_n_splits") or NOT_RECORDED),
        },
    ]

    gap = evaluation.get("gap_analysis") or {}
    holdout = metrics.get("f1_score")
    cv_f1 = cv_metrics.get("cv_f1_mean")
    gap_available = isinstance(holdout, (int, float)) and isinstance(cv_f1, (int, float))

    return {
        "available": True,
        "model_name": state.model_name,
        "metrics": metric_rows,
        "cv_metrics": cv_rows,
        "metrics_source": str(final.get("metrics_source") or NOT_RECORDED),
        "generalization_gap": (
            {
                "value": float(holdout) - float(cv_f1),
                "text": (
                    f"Hold-out F1 {format_metric(holdout)} minus cross-validated F1 "
                    f"{format_metric(cv_f1)} = "
                    f"{format_metric(float(holdout) - float(cv_f1))}."
                ),
            }
            if gap_available
            else None
        ),
        "interpretation": str(gap.get("interpretation") or ""),
        "gap_summary": str(gap.get("summary") or ""),
        "gap_disclaimer": str(gap.get("disclaimer") or ""),
        "ranking_rule": str(evaluation.get("ranking_rule") or ""),
        "holdout_samples": evaluation.get("test_samples"),
        "observations": list(evaluation.get("observations") or []),
    }

# --------------------------------------------------------------------------- #
# Section 6: confusion matrix
# --------------------------------------------------------------------------- #
def build_confusion_section(state: DashboardState) -> Dict[str, Any]:
    """Return the stored confusion matrix of the selected model, if there is one.

    Only Phase 4 stores a confusion matrix, so this section is genuinely empty
    whenever the selected model is a Phase 6 winner. That is not a failure and it
    is not filled in with the untuned model's matrix: those numbers describe a
    different model. The recorded ``note`` says why, so the absence reads as a
    known limitation rather than a bug.

    Works for binary and multiclass alike, because the stored value is a label
    list plus a square count matrix and nothing about the display assumes two
    classes.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``available``, ``labels``, ``matrix``, ``total``,
        ``note``, ``frame`` and ``per_class``. ``frame`` is a tidy
        ``DataFrame`` in long form, which is the shape Streamlit's built-in chart
        commands accept.
    """
    final = state.final_model
    stored = dict(final.get("confusion_matrix") or {})

    if not stored:
        return {
            "available": False,
            "note": str(final.get("confusion_matrix_note") or config.DASHBOARD_NO_CONFUSION_MATRIX),
        }

    labels = [str(item) for item in (stored.get("labels") or [])]
    matrix = [[int(value) for value in row] for row in (stored.get("matrix") or [])]
    if not labels or not matrix or len(matrix) != len(labels):
        return {
            "available": False,
            "note": (
                "The stored confusion matrix is not square or has no labels, so it "
                "cannot be drawn. This is a malformed artifact, not a missing one."
            ),
        }

    correct = sum(matrix[index][index] for index in range(len(labels)))
    total = int(stored.get("total") or sum(sum(row) for row in matrix))

    rows = []
    per_class = []
    for true_index, true_label in enumerate(labels):
        row_total = sum(matrix[true_index])
        for predicted_index, predicted_label in enumerate(labels):
            count = matrix[true_index][predicted_index]
            rows.append(
                {
                    "True label": true_label,
                    "Predicted label": predicted_label,
                    "Count": count,
                }
            )
            if true_index == predicted_index:
                per_class.append(
                    {
                        "label": true_label,
                        "support": row_total,
                        "correct": count,
                        "recall": (count / row_total) if row_total else None,
                    }
                )

    return {
        "available": True,
        "labels": labels,
        "matrix": matrix,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy_from_matrix": (correct / total) if total else None,
        "note": str(final.get("confusion_matrix_note") or ""),
        "frame": pd.DataFrame(rows),
        "per_class": per_class,
        "is_binary": len(labels) == 2,
    }


# --------------------------------------------------------------------------- #
# Section 7: optimization
# --------------------------------------------------------------------------- #
def build_optimization_section(state: DashboardState) -> Dict[str, Any]:
    """Return what Phase 6 did, without ever running a search.

    Every value is copied from the recorded results: the trial counts, the
    baseline-versus-optimized hold-out scores and the winning parameters. Optuna
    is not imported here and no study is created or resumed, so opening the
    dashboard cannot cost a second search.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``performed``, ``selected_model``, ``trials``,
        ``baseline_f1``, ``optimized_f1``, ``improvement``, ``improvement_text``,
        ``best_params``, ``best_params_text``, ``note``, ``rows`` and ``available``.
    """
    if not state.available:
        # Every unavailable section still says why, so the view never has to
        # guess and never renders an empty panel with no explanation.
        return {"available": False, "note": config.DASHBOARD_NO_ARTIFACT}

    optimization = state.optimization
    if not optimization or not optimization.get("performed"):
        return {
            "available": False,
            "note": config.DASHBOARD_NO_OPTIMIZATION,
        }

    best = dict(optimization.get("best_optimized_model") or {})
    if not best:
        return {
            "available": False,
            "note": (
                "The search ran but recorded no usable winner, so there is nothing to "
                "compare. The persisted model is the best baseline."
            ),
        }

    final = state.final_model
    selected_source = str(final.get("selection_source") or "")
    baseline_f1 = best.get("baseline_holdout_f1_score")
    optimized_f1 = best.get("holdout_f1_score")

    rows = []
    for entry in state.optimized_rows:
        rows.append(
            {
                "Model": str(entry.get("model_name") or NOT_RECORDED),
                "Status": str(entry.get("optimization_status") or NOT_RECORDED),
                "Baseline hold-out F1": format_metric(entry.get("baseline_f1_score")),
                "Optimized hold-out F1": format_metric(entry.get("optimized_f1_score")),
                "Baseline CV F1": format_metric(entry.get("baseline_cv_f1_mean")),
                "Optimized CV F1": format_metric(entry.get("optimized_cv_f1_mean")),
                "Trials": str(entry.get("total_trials") or NOT_RECORDED),
                "Completed": str(entry.get("completed_trials") or NOT_RECORDED),
                "Best trial": str(entry.get("best_trial_number") or NOT_RECORDED),
                "CV improvement": format_metric(entry.get("improvement")),
            }
        )

    improvement = best.get("improvement")
    direction = str(best.get("improvement_direction") or "")
    percentage = best.get("improvement_percentage")

    # Phase 6 records ``total_trials`` per model, so the number attached to the
    # winning recommendation is one model's trial budget and not the cost of the
    # whole run. Both are reported separately with the names that say which is
    # which, because "40 trials" and "20 trials" are both true here and a single
    # field could only be one of them.
    winner_trials = best.get("total_trials")
    trials_per_model = optimization.get("n_trials_per_model")
    searched = [
        entry
        for entry in state.optimized_rows
        if str(entry.get("optimization_status") or "") == "completed"
    ]
    total_trials = sum(
        int(entry.get("total_trials") or 0) for entry in searched
    ) or None

    return {
        "available": True,
        "performed": True,
        "selected_model": state.model_name,
        "selection_source": selected_source,
        "winner_trials": winner_trials,
        "winner_completed_trials": best.get("completed_trials"),
        "best_trial_number": best.get("best_trial_number"),
        "total_trials": total_trials,
        "trials_per_model": trials_per_model,
        "models_searched": len(searched),
        "sampler": str(optimization.get("sampler") or NOT_RECORDED),
        "sampler_seed": optimization.get("sampler_seed"),
        "cv_n_splits": optimization.get("cv_n_splits"),
        "objective_metric": str(optimization.get("objective_metric") or NOT_RECORDED),
        "baseline_f1": baseline_f1,
        "optimized_f1": optimized_f1,
        "improvement": improvement,
        "improvement_direction": direction,
        "improvement_text": (
            f"Cross-validated F1 improved by {format_metric(improvement)}"
            + (f" ({format_metric(percentage)}%)" if percentage is not None else "")
            + (f" - {direction}" if direction else "")
            if improvement is not None
            else NOT_RECORDED
        ),
        "holdout_delta": best.get("holdout_f1_delta"),
        "holdout_note": str(optimization.get("holdout_note") or ""),
        "best_params": dict(best.get("best_params") or {}),
        "best_params_text": format_parameters(best.get("best_params")),
        "selection_reason": str(best.get("selection_reason") or ""),
        "performance_summary": str(best.get("performance_summary") or ""),
        "note": (
            "The optimized candidate won the final selection."
            if selected_source == "optimized"
            else (
                "The search ran, but the best baseline won the final selection, so the "
                "baseline model is the one that was persisted."
                if selected_source == "baseline"
                else ""
            )
        ),
        "rows": rows,
        "models_selected": list(optimization.get("models_selected_for_optimization") or []),
        "models_not_selected": list(optimization.get("models_not_optimized") or []),
    }


# --------------------------------------------------------------------------- #
# Section 8: LLM interpretation
# --------------------------------------------------------------------------- #
def build_llm_section(state: DashboardState) -> Dict[str, Any]:
    """Return the stored Phase 8 interpretations.

    The section works with nothing stored at all: it then reports that no local
    model was used and stops. Nothing here contacts Ollama, so the dashboard never
    waits on a model, never fails because a model server is down, and shows exactly
    what the run that wrote the artifact produced.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``available``, ``entries``, ``note`` and ``statement``.
        ``entries`` holds one mapping per role with the four text fields the view
        renders.
    """
    llm = state.llm
    entries: List[Dict[str, Any]] = []

    for record in llm.get("results") or []:
        entries.append(
            {
                "role": str(record.get("role") or ""),
                "role_display_name": str(
                    record.get("role_display_name") or record.get("role") or ""
                ),
                "status": str(record.get("status") or NOT_RECORDED),
                "model": str(record.get("model") or NOT_RECORDED),
                "interpretation": str(record.get("interpretation") or ""),
                "observations": list(record.get("observations") or []),
                "recommendations": list(record.get("recommendations") or []),
                "confidence": record.get("confidence"),
                "confidence_note": str(record.get("confidence_note") or ""),
                "detail": str(record.get("detail") or ""),
            }
        )

    if not entries:
        return {
            "available": False,
            "entries": [],
            "note": config.DASHBOARD_NO_LLM,
            "statement": config.LLM_ROLE_STATEMENT,
        }

    return {
        "available": True,
        "entries": entries,
        "note": "",
        "statement": config.LLM_ROLE_STATEMENT,
        "failed": [entry for entry in entries if entry["status"] != "completed"],
    }


# --------------------------------------------------------------------------- #
# Section 11: experiment tracking
# --------------------------------------------------------------------------- #


def build_experiment_section(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the MLflow tracking facts for the dashboard.

    This module deliberately imports no ML library, and no agent, no graph and no
    llm package. That is what keeps the dashboard readable when the optional
    layers are broken or absent, and it is why the tracking data arrives as an
    argument instead of being fetched here: ``dashboard.py`` asks
    ``tools.mlflow_tools`` for it, inside a guard, and passes the plain mapping
    in. The result is that this function is a pure transformation, testable with
    a literal dictionary and unable to start a run even by accident.

    Args:
        summary: The mapping from ``mlflow_tools.experiment_summary``, or None
            when the view could not obtain one.

    Returns:
        A mapping with ``available``, ``enabled``, ``installed``,
        ``experiment_name``, ``tracking_uri``, ``latest``, ``runs`` and ``note``.
        Every key is always present, so the view never has to distinguish a
        missing field from an absent capability.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "enabled": False,
        "installed": False,
        "experiment_name": NOT_RECORDED,
        "tracking_uri": NOT_RECORDED,
        "latest": {},
        "runs": [],
        "note": config.DASHBOARD_NO_EXPERIMENT_TRACKING,
    }
    if not isinstance(summary, Mapping) or not summary:
        return empty

    latest = summary.get("latest") or {}
    rows: List[Dict[str, str]] = []
    for run in summary.get("runs") or []:
        rows.append(
            {
                "run_id": str(run.get("run_id") or NOT_RECORDED),
                "short_run_id": str(run.get("run_id") or "")[:8] or NOT_RECORDED,
                "start_time": str(run.get("start_time") or NOT_RECORDED),
                "run_name": str(run.get("run_name") or NOT_RECORDED),
                "task_type": str(run.get("task_type") or NOT_RECORDED),
                "dataset": str(run.get("dataset") or NOT_RECORDED),
                "model": str(run.get("model") or NOT_RECORDED),
                "f1_score": format_metric(_optional_metric(run.get("f1_score"))),
                "accuracy": format_metric(_optional_metric(run.get("accuracy"))),
                "optimization": str(run.get("optimization_status") or NOT_RECORDED),
                "status": str(run.get("status") or NOT_RECORDED),
                "workflow_status": str(run.get("workflow_status") or NOT_RECORDED),
            }
        )

    return {
        "available": bool(summary.get("available")),
        "enabled": bool(summary.get("enabled")),
        "installed": bool(summary.get("installed")),
        "experiment_name": str(
            summary.get("experiment_name") or config.MLFLOW_EXPERIMENT_NAME
        ),
        # The tracking URI is shown as configured. It is a local path or a URI
        # the developer chose; nothing here lets a reader change it, so it is
        # displayed rather than treated as input.
        "tracking_uri": str(summary.get("tracking_uri") or NOT_RECORDED),
        "latest": {
            "run_id": str(latest.get("run_id") or NOT_RECORDED),
            "short_run_id": str(latest.get("run_id") or "")[:8] or NOT_RECORDED,
            "run_name": str(latest.get("run_name") or NOT_RECORDED),
            "status": str(latest.get("status") or NOT_RECORDED),
            "workflow_status": str(latest.get("workflow_status") or NOT_RECORDED),
            "start_time": str(latest.get("start_time") or NOT_RECORDED),
            "model": str(latest.get("model") or NOT_RECORDED),
            "f1_score": format_metric(_optional_metric(latest.get("f1_score"))),
            "accuracy": format_metric(_optional_metric(latest.get("accuracy"))),
            "optimization_status": str(
                latest.get("optimization_status") or NOT_RECORDED
            ),
        },
        "runs": rows,
        "note": str(summary.get("note") or ""),
    }


def _optional_metric(value: Any) -> Optional[float]:
    """Coerce a metric that may be absent or unparsable.

    Args:
        value: The raw value from the tracking summary.

    Returns:
        The float, or None when there is no usable number. None renders as
        ``NOT_RECORDED`` rather than as 0.0, which would be a false measurement.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Section 10: model information
# --------------------------------------------------------------------------- #
def build_model_information_section(state: DashboardState) -> List[Dict[str, str]]:
    """Return the artifact facts, with no secrets and no absolute paths.

    The stored metadata deliberately holds no environment variables, no API keys,
    no credentials and no absolute paths - only file *names* relative to the output
    directory - so this section can show everything it holds. The one path printed
    is the directory the artifacts live in, which is derived from configuration
    rather than recorded, and it is shown because a reader needs to know where to
    look; it is not part of the artifact.

    Args:
        state: The loaded dashboard state.

    Returns:
        A list of label/value rows. Empty when no artifact is available.
    """
    if not state.available:
        return []

    metadata = state.metadata
    final = state.final_model
    paths = state.paths
    pipeline_steps = list(metadata.get("pipeline_steps") or [])
    versions = dict(metadata.get("library_versions") or {})

    rows = [
        {"label": "Artifact", "value": str(metadata.get("artifact") or NOT_RECORDED)},
        {"label": "Metadata file", "value": paths.metadata.name if paths else NOT_RECORDED},
        {"label": "Results file", "value": paths.results.name if paths else NOT_RECORDED},
        {"label": "Model name", "value": str(metadata.get("model_name") or NOT_RECORDED)},
        {"label": "Model type", "value": str(metadata.get("model_type") or NOT_RECORDED)},
        {
            "label": "Selection source",
            "value": str(final.get("selection_source") or NOT_RECORDED),
        },
        {"label": "Target column", "value": state.target_column},
        {"label": "Task type", "value": state.task_type},
        {"label": "Feature count", "value": str(metadata.get("feature_count") or NOT_RECORDED)},
        {
            "label": "Class count",
            "value": str(metadata.get("class_count") or NOT_RECORDED),
        },
        {"label": "Optimization used", "value": str(final.get("optimization_used"))},
        {"label": "Final fit strategy", "value": str(final.get("final_fit_strategy") or metadata.get("final_fit", {}).get("strategy", "")) or NOT_RECORDED},
        {"label": "Pipeline steps", "value": " -> ".join(pipeline_steps) or NOT_RECORDED},
        {"label": "Schema version", "value": str(metadata.get("schema_version") or NOT_RECORDED)},
        {
            "label": "Artifact size",
            "value": f"{metadata.get('artifact_size_bytes')} bytes"
            if metadata.get("artifact_size_bytes")
            else NOT_RECORDED,
        },
    ]

    if versions:
        rows.append(
            {"label": "Library versions", "value": ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))}
        )
    if paths is not None:
        # Shown relative to the project root rather than as an absolute path: the
        # location is still identifiable, and the page does not leak the user's
        # home directory into a screen that may be shared or screenshotted.
        rows.append(
            {
                "label": "Artifact directory",
                "value": _relative_to_project(paths.directory),
            }
        )

    return rows


def _relative_to_project(directory: Path) -> str:
    """Return ``directory`` relative to the project root, or its name.

    Args:
        directory: The artifact directory.

    Returns:
        A short, portable path such as ``"models"``. A directory outside the
        project - a temporary directory in a test, or a configured output
        elsewhere - falls back to its final component, which is enough to
        identify it without printing an absolute path.
    """
    try:
        return str(directory.resolve().relative_to(Path(__file__).resolve().parent.parent)).replace("\\", "/")
    except ValueError:
        return directory.name

# --------------------------------------------------------------------------- #
# Section 9: prediction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictionField:
    """One input the prediction form needs.

    ``kind`` is ``"number"`` or ``"text"``, chosen from the recorded feature type
    rather than guessed from the values. ``default`` is filled from the recorded
    training statistics so the form starts at something realistic instead of at
    zero, which for a standardised column would be a valid but meaningless row.

    ``options`` is populated for a categorical column when the recorded data
    analysis recorded the values it saw, so the form can offer a choice instead of
    a free-text box. An unseen value is still accepted: the fitted encoder was
    built with ``handle_unknown="ignore"``, so the pipeline decides what to do.
    """

    name: str
    kind: str
    default: Any = None
    options: Sequence[str] = ()

    @property
    def is_numeric(self) -> bool:
        """True when this input should be rendered as a number."""
        return self.kind == "number"


@dataclass(frozen=True)
class PredictionOutcome:
    """The result of one prediction, or the reason there is none.

    ``ok=False`` carries a message instead of a prediction. This is how a failed
    prediction reaches the view: as data, so the form stays on screen with the
    user's input intact rather than replacing the page with an error.
    """

    ok: bool = False
    predicted_class: Optional[str] = None
    probabilities: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    model_name: str = ""
    message: str = ""
    detail: str = ""

    @property
    def supports_probability(self) -> bool:
        """True when class probabilities were returned."""
        return bool(self.probabilities)


def build_prediction_fields(state: DashboardState) -> List[PredictionField]:
    """Return the input form, derived from the persisted metadata.

    The field list comes from the metadata written next to the pipeline, not from
    the CSV and not from the fitted transformer. That is what keeps the form and the
    saved model in agreement: if the two ever disagreed, the pipeline would reject
    the row, and the disagreement would be visible in the artifact rather than
    hidden in the form.

    Args:
        state: The loaded dashboard state.

    Returns:
        One :class:`PredictionField` per column the pipeline consumes, numerical
        columns first. Empty when no artifact is available.
    """
    if not state.available:
        return []

    dataset = state.dataset
    statistics = dict(dataset.get("numerical_statistics") or {})
    categories = dict(dataset.get("categorical_statistics") or {})

    fields: List[PredictionField] = []
    for name in state.numerical_features():
        stats = dict(statistics.get(name) or {})
        default = stats.get("mean")
        fields.append(
            PredictionField(
                name=name,
                kind="number",
                default=float(default) if isinstance(default, (int, float)) else 0.0,
            )
        )

    for name in state.categorical_features():
        stats = dict(categories.get(name) or {})
        options = [str(item) for item in (stats.get("top_values") or {})]
        fields.append(
            PredictionField(
                name=name,
                kind="text",
                default=options[0] if options else "",
                options=tuple(options),
            )
        )

    return fields


def build_prediction_frame(
    fields: Sequence[PredictionField],
    values: Dict[str, Any],
) -> pd.DataFrame:
    """Turn the form values into the one-row frame the pipeline expects.

    The column order matches the recorded feature order and the target column is
    **not** added: including it would make the pipeline predict a target that is
    already present in the input, which is the classic way to turn an honest
    prediction into a meaningless one.

    Args:
        fields: The fields :func:`build_prediction_fields` returned.
        values: Column name to submitted value. A missing key is filled with the
            field's default, so a partially filled form still produces a frame.

    Returns:
        A single-row `DataFrame` with exactly the pipeline's input columns.
    """
    row: Dict[str, Any] = {}
    for field_spec in fields:
        value = values.get(field_spec.name, field_spec.default)
        if field_spec.is_numeric and value is None:
            # An empty numeric box means "not supplied", not "measured as zero".
            # Zero would be a fabricated measurement for a column like glucose;
            # NaN reaches the imputer Phase 3 fitted, which is the behaviour the
            # model was actually trained with.
            value = float("nan")
        row[field_spec.name] = value
    return pd.DataFrame([row], columns=[field_spec.name for field_spec in fields])


def _unknown_categories(
    fields: Sequence[PredictionField],
    values: Dict[str, Any],
) -> List[str]:
    """List the text fields whose value is not one of the recorded options.

    Args:
        fields: The fields the form was built from.
        values: Column name to submitted value.

    Returns:
        One message per offending field, naming the value and the allowed set.
        Empty when every text value is one the dataset contained.
    """
    problems: List[str] = []
    for field_spec in fields:
        if field_spec.is_numeric or not field_spec.options:
            continue
        value = values.get(field_spec.name, field_spec.default)
        if value is None:
            continue
        if str(value) not in [str(option) for option in field_spec.options]:
            allowed = ", ".join(str(option) for option in field_spec.options)
            problems.append(
                f"{field_spec.name}={value!r} is not one of: {allowed}"
            )
    return problems


def run_prediction(
    state: DashboardState,
    fields: Sequence[PredictionField],
    values: Dict[str, Any],
) -> PredictionOutcome:
    """Predict one row with the **saved** pipeline.

    This is the whole point of persisting the artifact: the preprocessing the
    model was fitted with - imputation, one-hot encoding, scaling - lives inside
    the saved `Pipeline`, so this function calls ``predict`` once and lets the
    pipeline do the rest. Duplicating any of those steps here would be a second
    implementation of Phase 3, and the two would eventually disagree.

    ``predict_proba`` is called only when the saved estimator has that method, so
    an estimator without probabilities (none of the five in this project's registry
    lack it, but the check costs nothing) produces a prediction without them rather
    than raising.

    Args:
        state: The loaded dashboard state.
        fields: The fields the form was built from.
        values: Column name to submitted value.

    Returns:
        A :class:`PredictionOutcome`. Never raises for a missing artifact: the
        ``ok=False`` outcome carries the message the view displays.
    """
    if not state.available or state.model is None:
        return PredictionOutcome(ok=False, message=config.DASHBOARD_NO_ARTIFACT)

    if not fields:
        return PredictionOutcome(
            ok=False,
            message=(
                "The persisted metadata records no feature columns, so there is no "
                "form to fill in. Re-run the workflow to rewrite the artifacts."
            ),
        )

    frame = build_prediction_frame(fields, values)
    pipeline = state.model.pipeline

    unknown = _unknown_categories(fields, values)
    if unknown:
        # The encoder would happily encode an unseen level as all-zeros, which
        # silently ignores the field. Refusing keeps the prediction honest: a
        # number produced from an input the model was never shown would look
        # like a measurement of something else.
        return PredictionOutcome(
            ok=False,
            message="One or more fields hold a value the saved model never saw.",
            detail="; ".join(unknown),
        )

    try:
        raw = pipeline.predict(frame)
    except Exception as error:  # noqa: BLE001 - shown to the user, never hidden
        return PredictionOutcome(
            ok=False,
            message="The saved model rejected this input.",
            detail=f"{type(error).__name__}: {error}",
        )

    predicted = str(raw[0]) if len(raw) else NOT_RECORDED

    probabilities: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    if state.model.supports_probability:
        try:
            scores = pipeline.predict_proba(frame)[0]
            labels = state.class_labels
            probabilities = [
                {
                    "class": labels[index] if index < len(labels) else str(index),
                    "probability": float(value),
                }
                for index, value in enumerate(scores)
            ]
            confidence = float(max(scores)) if len(scores) else None
        except Exception:  # noqa: BLE001 - probabilities are optional extra detail
            probabilities = None
            confidence = None

    return PredictionOutcome(
        ok=True,
        predicted_class=predicted,
        probabilities=probabilities,
        confidence=confidence,
        model_name=state.model_name,
        message=(
            ""
            if probabilities is not None
            else (
                f"{state.model_name} does not expose predict_proba, so no class "
                "probabilities are shown. The predicted class comes from the same "
                "saved pipeline either way."
            )
        ),
    )


def build_prediction_section(state: DashboardState) -> Dict[str, Any]:
    """Return everything the prediction section needs, in one mapping.

    Bundled so the view makes a single call and so the "can we predict at all?"
    question is answered in one place: no artifact, or no recorded feature columns,
    means the section shows a message instead of a form.

    Args:
        state: The loaded dashboard state.

    Returns:
        A mapping with ``available``, ``fields``, ``note``, ``supports_probability``
        and ``class_labels``.
    """
    if not state.available or state.model is None:
        return {
            "available": False,
            "fields": [],
            "note": config.DASHBOARD_NO_ARTIFACT,
            "supports_probability": False,
            "class_labels": [],
        }

    fields = build_prediction_fields(state)
    if not fields:
        return {
            "available": False,
            "fields": [],
            "note": (
                "The persisted metadata records no feature columns, so there is no "
                "form to fill in. Re-run the workflow to rewrite the artifacts."
            ),
            "supports_probability": False,
            "class_labels": state.class_labels,
        }

    return {
        "available": True,
        "fields": fields,
        "note": "",
        "supports_probability": state.model.supports_probability,
        "class_labels": state.class_labels,
        "target_column": state.target_column,
        "model_name": state.model_name,
    }


def build_benchmark_section(report_path: Any = None) -> Dict[str, Any]:
    """Return the Phase 11 benchmark comparison for the dashboard, read-only.

    The dashboard never runs the benchmark: this builder only reads the
    ``reports/phase11_comparison.json`` the CLI produced. When the file is
    missing or unreadable the section still renders, with the exact
    "not available" message so the dashboard never depends on Phase 11 having
    been executed.

    Args:
        report_path: Explicit path to the comparison JSON (tests inject a path);
            defaults to ``reports/phase11_comparison.json`` next to the project
            root.

    Returns:
        A mapping with ``available``, ``note``, ``generated_at`` and
        ``datasets``. Every key is always present. ``datasets`` holds one entry
        per recorded dataset with ``dataset``, ``verdict``, ``delta_f1``,
        ``delta_accuracy``, ``delta_runtime``, ``delta_experiments`` and a
        ``rows`` list (one conventional + one autonomous row) shaped for a
        Streamlit table.
    """
    empty: Dict[str, Any] = {
        "available": False,
        "note": config.DASHBOARD_NO_BENCHMARK,
        "generated_at": NOT_RECORDED,
        "datasets": [],
    }
    if report_path is None:
        report_path = (
            Path(__file__).resolve().parent.parent / "reports" / "phase11_comparison.json"
        )
    path = Path(report_path)
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return empty

    datasets: List[Dict[str, Any]] = []
    for dataset_key, entry in (payload.get("datasets") or {}).items():
        conventional = entry.get("conventional") or {}
        autonomous = entry.get("autonomous") or {}
        delta = entry.get("delta") or {}
        rows = []
        for label, record in (
            ("Conventional", conventional),
            ("Autonomous", autonomous),
        ):
            rows.append(
                {
                    "workflow": label,
                    "model": str(record.get("model") or NOT_RECORDED),
                    "selection_source": str(
                        record.get("selection_source") or NOT_RECORDED
                    ),
                    "f1_score": format_metric(_optional_metric(record.get("f1_score"))),
                    "accuracy": format_metric(_optional_metric(record.get("accuracy"))),
                    "cv_f1_mean": format_metric(
                        _optional_metric(record.get("cv_f1_mean"))
                    ),
                    "runtime_seconds": format_metric(
                        _optional_metric(record.get("runtime_seconds")), digits=3
                    ),
                    "experiments": str(record.get("experiments") or 0),
                }
            )
        datasets.append(
            {
                "dataset": dataset_key,
                "verdict": str(entry.get("verdict") or NOT_RECORDED),
                "delta_f1": format_metric(_optional_metric(delta.get("f1"))),
                "delta_accuracy": format_metric(_optional_metric(delta.get("accuracy"))),
                "delta_cv_f1": format_metric(_optional_metric(delta.get("cv_f1"))),
                "delta_runtime": format_metric(
                    _optional_metric(delta.get("runtime_seconds")), digits=3
                ),
                "delta_experiments": str(delta.get("experiments") or NOT_RECORDED),
                "rows": rows,
                "notes": list(entry.get("notes") or []),
            }
        )

    metadata = payload.get("metadata") or {}
    return {
        "available": bool(datasets),
        "note": "",
        "generated_at": str(metadata.get("generated_at") or NOT_RECORDED),
        "conventional_grid_size": metadata.get("conventional_grid_size"),
        "autonomous_n_trials": metadata.get("autonomous_n_trials"),
        "datasets": datasets,
    }


__all__ = [
    "NOT_RECORDED",
    "DashboardState",
    "PredictionField",
    "PredictionOutcome",
    "build_comparison_section",
    "build_confusion_section",
    "build_dataset_section",
    "build_evaluation_section",
    "build_experiment_section",
    "build_llm_section",
    "build_model_information_section",
    "build_optimization_section",
    "build_benchmark_section",
    "build_prediction_fields",
    "build_prediction_frame",
    "build_prediction_section",
    "build_status_section",
    "format_metric",
    "format_parameter",
    "format_parameters",
    "load_dashboard_state",
    "run_prediction",
]
