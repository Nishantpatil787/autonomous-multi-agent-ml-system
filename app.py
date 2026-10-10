"""Main application entry point for the Autonomous Multi-Agent ML System.

Phase 1: project foundation.
Phase 2: Data Analysis Agent.
Phase 3: Preprocessing Agent.
Phase 4: Model Selection Agent and Training Agent.
Phase 5: Evaluation Agent.
Phase 6: Optimization Agent (Optuna hyperparameter search).
Phase 7: LangGraph orchestration of the existing agents (--graph).
Phase 8: optional local LLM layer for explanation and planning (--graph --llm).
    It is off by default, it needs a local Ollama model, and it explains results
    without ever becoming the authority for one. The core machine learning
    pipeline does not require an LLM.
Phase 9: Model Persistence Agent and the Streamlit dashboard. The graph now ends
    by writing models/best_model.joblib plus its metadata, and the dashboard reads
    exactly those files. It never retrains and never re-evaluates.
Phase 10: local MLflow experiment tracking for every --graph run, recorded in a
    separate store under experiments/tracking and never influencing a decision.
Phase 11: a controlled benchmark comparing the conventional workflow (tools only,
    manual grid of 21 configurations derived from the Phase 6 search spaces)
    against the autonomous workflow (graph + Optuna), run on both committed
    datasets with the same split, the same preprocessing principles, the same
    candidate families and the same metrics. Results and charts are written to
    reports/phase11_* (JSON, CSV and three PNG charts) and the dashboard gains a
    read-only section. Run it with 'python app.py --benchmark'.

This is a small command line entry point used to verify the project and to run
the agents manually. It is intentionally minimal: the interactive surface is the
Streamlit dashboard in dashboard.py, which displays what this command produced.

Two execution paths are available and they call exactly the same seven agents.
Without --graph the agents are invoked directly in sequence (the path that
existed through Phase 6, unchanged apart from the new final step). With --graph a
LangGraph workflow invokes them and adds explicit routing, failure handling, a
bounded retry policy and the optimization decision. Both produce the same machine
learning results.

No language model is involved in the machine learning path at all. The Phase 8
layer is optional and can only be turned on together with --graph, and even then
it only appends an explanation after a deterministic step: it never decides, and
it cannot change a metric, a model or a routing decision.

Usage examples:
    python app.py
    python app.py --dataset datasets/sample_classification.csv
    python app.py --dataset datasets/sample_multiclass.csv --json
    python app.py --dataset data/titanic.csv --target survived
    python app.py --no-preprocess
    python app.py --no-train
    python app.py --no-cross-validation
    python app.py --no-evaluate
    python app.py --no-optimize
    python app.py --graph
    python app.py --graph --no-optimize
    python app.py --graph --llm
    python app.py --graph --no-save-model

Then, in a second terminal:

    streamlit run dashboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Make the project root importable when the file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from graph.workflow import format_workflow_report, run_workflow
from utils import config
from tools.data_tools import DataToolError
from tools.evaluation_tools import EvaluationToolError
from tools.model_tools import ModelToolError
from tools.optimization_tools import OptimizationToolError
from tools.persistence_tools import PersistenceError, artifact_paths
from tools.preprocessing_tools import PreprocessingError
from tools.training_tools import TrainingToolError
from utils.config import DATASETS_DIR
from utils.logger import setup_logger
from utils.state import AgentState, PersistenceStatus

logger = setup_logger("app")

DEFAULT_DATASET = DATASETS_DIR / "sample_classification.csv"
SEPARATOR = "=" * 78


def _section(title: str) -> str:
    """Return a formatted section header."""
    return f"\n{title}\n{'-' * len(title)}"


def _format_columns(state: AgentState) -> str:
    """Render the column list with its pandas dtype."""
    lines: List[str] = []
    for column in state.column_names:
        dtype = state.column_dtypes.get(column, "unknown")
        role = "TARGET" if column == state.target_column else ""
        lines.append(f"  {column:<18} {dtype:<12} {role}".rstrip())
    return "\n".join(lines) if lines else "  (no columns)"


def _format_numerical_statistics(state: AgentState) -> str:
    """Render the numerical statistics table."""
    if not state.numerical_statistics:
        return "  (none)"

    lines = [f"  {'column':<18}{'count':>8}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}"]
    for stats in state.numerical_statistics.values():
        def render(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        lines.append(
            f"  {stats.column:<18}{stats.count:>8}{render(stats.mean):>12}"
            f"{render(stats.std):>12}{render(stats.minimum):>12}{render(stats.maximum):>12}"
        )
    return "\n".join(lines)


def _format_categorical_statistics(state: AgentState) -> str:
    """Render unique counts and the most frequent values per categorical column."""
    if not state.categorical_statistics:
        return "  (none)"

    lines: List[str] = []
    for stats in state.categorical_statistics.values():
        top = ", ".join(
            f"{label} ({count}, {stats.top_values_percentage[label]}%)"
            for label, count in stats.top_values.items()
        )
        lines.append(f"  {stats.column:<18} {stats.unique_count:>3} unique -> {top}")
    return "\n".join(lines)


def format_analysis_report(state: AgentState) -> str:
    """Build the human readable analysis report from the shared state."""
    parts: List[str] = [SEPARATOR, "DATA ANALYSIS REPORT", SEPARATOR]

    summary = state.analysis_summary
    shape = state.dataset_shape
    parts.append(
        _section("1. Dataset")
        + f"\n  File      : {state.dataset_path}"
        + f"\n  Shape     : {shape.rows} rows x {shape.columns} columns"
        if shape
        else _section("1. Dataset") + f"\n  File      : {state.dataset_path}"
    )

    parts.append(_section("2. Columns and types") + "\n" + _format_columns(state))

    parts.append(
        _section("3. Feature types")
        + f"\n  Numerical   ({len(state.numerical_features)}): {', '.join(state.numerical_features) or '-'}"
        + f"\n  Categorical ({len(state.categorical_features)}): {', '.join(state.categorical_features) or '-'}"
        + f"\n  Unsupported ({len(state.unsupported_columns)}): {', '.join(state.unsupported_columns) or '-'}"
    )

    missing = state.missing_values
    if missing is not None:
        detail_lines = [
            f"  {column}: {count} missing ({missing.per_column_percentage[column]}%)"
            for column, count in missing.per_column.items()
        ]
        missing_text = (
            f"  Total missing values : {missing.total} ({missing.total_percentage}% of all cells)\n"
            f"  Affected columns     : {missing.affected_columns}\n"
            + ("\n".join(detail_lines) if detail_lines else "  No missing values found.")
        )
    else:
        missing_text = "  (not analysed)"
    parts.append(_section("4. Missing values") + "\n" + missing_text)
    parts.append(_section("5. Duplicate rows") + f"\n  Duplicate rows: {state.duplicate_count}")

    candidates = ", ".join(
        f"{candidate.column} ({candidate.score})" for candidate in state.target_candidates
    )
    parts.append(
        _section("6. Target column")
        + f"\n  Target column           : {state.target_column or 'not detected'}"
        + f"\n  Detection method        : {state.target_detection_method}"
        + f"\n  Detection confidence    : {state.target_detection_confidence}"
        + f"\n  Reason                  : {state.target_detection_reason}"
        + (f"\n  Ranked candidates       : {candidates}" if candidates else "")
    )

    parts.append(
        _section("7. Problem type")
        + f"\n  Problem type : {state.problem_type}"
        + f"\n  Reason       : {state.problem_type_reason}"
    )

    if state.class_distribution:
        distribution = "\n".join(
            f"  {label}: {entry.count} samples ({entry.percentage}%)"
            for label, entry in state.class_distribution.items()
        )
    else:
        distribution = "  (no class distribution available)"
    parts.append(_section("8. Class distribution") + "\n" + distribution)

    parts.append(_section("9. Numerical statistics") + "\n" + _format_numerical_statistics(state))
    parts.append(
        _section("10. Categorical statistics") + "\n" + _format_categorical_statistics(state)
    )

    if state.data_quality_issues:
        issue_lines = [
            f"  [{issue.severity.upper():<7}] {issue.kind}"
            + (f" ({issue.column})" if issue.column else "")
            + f": {issue.message}"
            for issue in state.data_quality_issues
        ]
        counts = state.issues_by_severity()
        issues_text = (
            f"  {len(state.data_quality_issues)} issue(s): "
            f"{counts['error']} error, {counts['warning']} warning, {counts['info']} info\n"
            "  (reported only - nothing was removed or modified)\n"
            + "\n".join(issue_lines)
        )
    else:
        issues_text = "  No data quality issues detected."
    parts.append(_section("11. Data quality issues") + "\n" + issues_text)

    if summary is not None:
        parts.append(_section("12. Summary") + "\n  " + summary.summary_text)

    parts.append(_section("13. Agent messages"))
    parts.append(
        "\n"
        + "\n".join(f"  [{message.level:<7}] {message.message}" for message in state.agent_messages)
    )
    parts.append(SEPARATOR)
    return "\n".join(parts)


def format_preprocessing_report(state: AgentState) -> str:
    """Build the human readable preprocessing report from the shared state.

    Phase 3 inspection only: it shows what the Preprocessing Agent decided. It
    never fits the pipeline and never trains a model.
    """
    summary = state.preprocessing_summary
    parts: List[str] = [SEPARATOR, "PREPROCESSING REPORT", SEPARATOR]

    parts.append(
        _section("1. Status")
        + f"\n  Status    : {state.preprocessing_status}"
        + f"\n  Target    : {summary.target_column if summary else state.target_column}"
        + f"\n  Pipeline  : {'built (not fitted)' if state.preprocessing_pipeline else 'not built'}"
        + f"\n  Fit scope : "
        + (
            state.preprocessing_config.fit_scope
            if state.preprocessing_config
            else "-"
        )
    )

    if summary is not None and summary.status != "completed" and summary.status_reason:
        parts.append(_section("2. Reason") + "\n  " + summary.status_reason)

    config = state.preprocessing_config
    if config is None:
        parts.append(SEPARATOR)
        return "\n".join(parts)

    parts.append(
        _section("3. Selected features")
        + f"\n  Numerical   ({len(config.numerical_features)}): "
        + (", ".join(config.numerical_features) or "-")
        + f"\n  Categorical ({len(config.categorical_features)}): "
        + (", ".join(config.categorical_features) or "-")
    )

    if config.excluded_features:
        excluded_lines = "\n".join(
            f"  {item.column:<18} {item.reason:<18} {item.detail}"
            for item in config.excluded_features
        )
        excluded_text = f"  {len(config.excluded_features)} column(s):\n{excluded_lines}"
    else:
        excluded_text = "  No column was excluded."
    parts.append(_section("4. Excluded features") + "\n" + excluded_text)

    parts.append(
        _section("5. Strategies")
        + f"\n  Numerical imputation   : SimpleImputer(strategy='{config.numerical_imputation}')"
        + f"\n  Categorical imputation : SimpleImputer(strategy='{config.categorical_imputation}')"
        + f"\n  Encoding              : OneHotEncoder(handle_unknown='{config.encoding_handle_unknown}')"
        + f"\n  Scaling               : "
        + (
            f"StandardScaler() (scale_numeric={str(config.scale_numeric).lower()})"
            if config.scale_numeric
            else "disabled"
        )
        + f"\n  Remainder             : {config.remainder}"
        + f"\n  Duplicate strategy    : {config.duplicate_handling.strategy} "
        + f"({config.duplicate_handling.duplicate_rows_detected} detected, "
        + f"{config.duplicate_handling.duplicate_rows_removed} removed so far)"
    )

    parts.append(
        _section("6. Pipeline architecture")
        + "\n"
        + "\n".join(
            f"  {step.name} ({step.applies_to}): {' -> '.join(step.transformers)}"
            for step in config.steps
        )
        + "\n  combined by ColumnTransformer(remainder='drop')"
    )

    if summary is not None:
        parts.append(
            _section("7. Summary")
            + f"\n  Candidate features : {summary.original_feature_count}"
            + f"\n  Used features      : {summary.used_feature_count} "
            + f"({summary.numerical_feature_count} numerical, "
            + f"{summary.categorical_feature_count} categorical)"
            + f"\n  Excluded features  : {summary.excluded_feature_count}"
            + f"\n  Excluded by reason : {summary.excluded_by_reason or '-'}"
            + f"\n  Estimated output   : {summary.estimated_output_feature_count} columns "
            + "(becomes exact after fitting)"
            + f"\n  {summary.summary_text}"
        )

    parts.append(SEPARATOR)
    return "\n".join(parts)


def format_model_selection_report(state: AgentState) -> str:
    """Build the human readable model selection report.

    Phase 4, step 1: only the decision and its reasons. No model is trained.
    Sections are numbered at render time so there are no gaps when an optional
    section is empty.
    """
    summary = state.model_selection_summary
    sections: List[tuple] = [
        (
            "Status",
            f"  Status       : {state.model_selection_status}"
            f"\n  Problem type : {state.problem_type}"
            f"\n  Target       : {state.target_column}",
        )
    ]

    if summary is not None and summary.status != "completed" and summary.status_reason:
        sections.append(("Reason", f"  {summary.status_reason}"))
        return _render_sections("MODEL SELECTION REPORT", sections)

    sections.append(
        (
            "Selected models",
            "\n"
            + "\n".join(
                f"  {model.name:<24}{model.model_type:<16}"
                f"scaling: {'yes' if model.scaling_required else 'no ':<4}{model.default_parameters}"
                for model in state.selected_models
            ),
        )
    )

    if state.selected_models:
        sections.append(
            (
                "Why these models",
                "\n"
                + "\n".join(
                    f"  {model.name}: {model.selection_reason}" for model in state.selected_models
                ),
            )
        )

    if summary is not None and summary.excluded_models:
        sections.append(
            (
                "Not selected",
                "\n"
                + "\n".join(
                    f"  {name}: {reason}" for name, reason in summary.excluded_models.items()
                ),
            )
        )

    if summary is not None:
        sections.append(
            (
                "Baseline parameters",
                "\n  No hyperparameter search is performed in Phase 4: these are the "
                "reproducible defaults.\n"
                + "\n".join(
                    f"  {name:<24}{parameters}"
                    for name, parameters in sorted(
                        (model.name, model.default_parameters)
                        for model in state.selected_models
                    )
                ),
            )
        )

    return _render_sections("MODEL SELECTION REPORT", sections)


def format_training_report(state: AgentState) -> str:
    """Build the human readable training and comparison report.

    Phase 4, step 2: the split, the per-model results and the best baseline
    model. All numbers come from the real run; nothing here is fabricated.
    """
    sections: List[tuple] = [
        (
            "Status",
            f"  Status          : {state.training_status}"
            f"\n  Experiments     : {state.experiment_count}"
            f"\n  Successful      : {len(state.successful_experiments)}"
            f"\n  Failed          : {len(state.failed_experiments)}"
            f"\n  Wall-clock time : {state.execution_time:.4f} s",
        )
    ]

    split = state.data_split
    if split is not None:
        sections.append(
            (
                "Train/test split",
                f"  Test size       : {split.test_size} (random_state={split.random_state})"
                f"\n  Stratified      : {split.stratified}"
                f"\n  Samples         : {split.total_samples} rows -> "
                f"{split.train_samples} train / {split.test_samples} test"
                f"\n  Duplicates      : {split.duplicate_rows_detected} detected, "
                f"{split.duplicate_rows_removed} removed ({split.duplicate_handling_strategy})"
                f"\n  Features        : {split.input_feature_count} input, "
                f"{split.excluded_feature_count} excluded by Phase 3"
                + (
                    f"\n  Note            : {split.stratification_reason}"
                    if split.stratification_reason
                    else ""
                )
                + "\n  Class balance   : "
                + ", ".join(
                    f"{label}={count}"
                    for label, count in split.class_distribution_before_split.items()
                ),
            )
        )

    cv_config = state.cross_validation_config
    if cv_config is not None:
        sections.append(
            (
                "Cross-validation",
                f"  Enabled    : {cv_config.enabled}"
                f"\n  Folds      : {cv_config.n_splits} "
                f"(shuffle={cv_config.shuffle}, random_state={cv_config.random_state})"
                f"\n  Scoring    : {cv_config.scoring or '-'}"
                + (
                    f"\n  Adjustment : {cv_config.adjusted_reason}"
                    if cv_config.adjusted_reason
                    else ""
                )
                + "\n  The Pipeline (preprocessor + model) is refitted inside every fold, "
                "so preprocessing never sees the validation rows.",
            )
        )

    if state.experiment_results:
        sections.append(
            (
                "Per-model results",
                "\n"
                + "\n".join(
                    f"  {experiment.model_name:<24}"
                    f"acc={_fmt(experiment.accuracy)}  prec={_fmt(experiment.precision)}  "
                    f"rec={_fmt(experiment.recall)}  f1={_fmt(experiment.f1_score)}  "
                    f"cv_f1={_fmt(experiment.cv_f1_mean)}  "
                    f"feat={experiment.transformed_feature_count}  "
                    f"time={experiment.total_time:.3f}s  [{experiment.status}]"
                    for experiment in state.experiment_results
                ),
            )
        )

        failed = state.failed_experiments
        if failed:
            sections.append(
                (
                    "Failures",
                    "\n"
                    + "\n".join(
                        f"  {experiment.model_name}: {experiment.error_message}"
                        for experiment in failed
                    )
                    + "\n  The other models still trained; failures are excluded from the ranking.",
                )
            )

    matrix_lines: List[str] = []
    for experiment in state.successful_experiments:
        matrix = experiment.confusion_matrix
        if matrix is None:
            continue
        matrix_lines.append(f"    {experiment.model_name} (labels: {matrix.labels})")
        matrix_lines.append("      " + "".join(f"{label:>7}" for label in matrix.labels))
        for label, row in zip(matrix.labels, matrix.matrix):
            matrix_lines.append(
                f"      {label:<5}" + "".join(f"{value:>7}" for value in row)
            )
    sections.append(
        ("Confusion matrices", "\n" + ("\n".join(matrix_lines) if matrix_lines else "  (none)"))
    )

    comparison = state.model_comparison
    if comparison is not None:
        sections.append(
            ("Model comparison", "\n" + "\n".join(comparison.table_text.splitlines()))
        )
        sections.append(("Ranking rule", f"  {comparison.ranking_rule}"))
        if comparison.notes:
            sections.append(
                ("Notes", "\n" + "\n".join(f"  - {note}" for note in comparison.notes))
            )

    best = state.best_baseline_model
    if best is not None:
        sections.append(
            (
                "Best baseline model",
                f"  Model          : {best.model_name}"
                f"\n  F1 score       : {_fmt(best.f1_score)} (primary metric)"
                f"\n  Accuracy       : {_fmt(best.accuracy)} (secondary metric)"
                f"\n  CV mean F1     : {_fmt(best.cv_f1_mean)}"
                f"\n  Scaling        : {'enabled' if best.preprocessing_scaling else 'disabled'}"
                f"\n  Parameters     : {best.parameters}"
                f"\n  Experiment id  : {best.experiment_id}"
                f"\n  Why            : {best.selection_reason}"
                "\n  Note: this is the best BASELINE model. Hyperparameter optimization "
                "(Phase 6) may change it.",
            )
        )
    else:
        sections.append(
            ("Best baseline model", "  None: no experiment produced usable metrics.")
        )

    return _render_sections("TRAINING REPORT", sections)


def format_evaluation_report(state: AgentState) -> str:
    """Build the human readable evaluation report.

    Phase 5: interpretation of the baseline experiments only. Nothing is
    retrained here - every number comes from the records the Training Agent
    produced. Each section states what was measured and, where a judgement is
    made, labels it as a reporting heuristic rather than a proof.
    """
    report = state.evaluation_report
    sections: List[tuple] = [
        (
            "Status",
            f"  Status             : {state.evaluation_status}"
            + (
                f"\n  Dataset           : {report.dataset_name}"
                if report and report.dataset_name
                else ""
            )
            + (
                f"\n  Problem type      : {report.problem_type}"
                if report and report.problem_type
                else ""
            )
            + (
                f"\n  Hold-out samples  : {report.test_samples}"
                if report and report.test_samples
                else ""
            )
            + "\n  No model was retrained by the Evaluation Agent.",
        )
    ]

    if report is None:
        sections.append(("Evaluation", "  (no evaluation report was produced)"))
        return _render_sections("EVALUATION REPORT", sections)

    best = report.best_model
    if best is not None:
        sections.append(
            (
                "Best baseline recommendation",
                f"  Model              : {best.best_baseline_model} ({best.model_type})"
                f"\n  Hold-out F1        : {_fmt(best.f1_score)}"
                f"\n  Hold-out accuracy  : {_fmt(best.accuracy)}"
                f"\n  CV mean F1         : {_fmt(best.cv_f1_mean)}"
                f"\n  F1 gap             : {_fmt(best.f1_gap)}"
                f"\n  Experiment time    : {best.total_time:.4f} s"
                f"\n  Scaling            : {'enabled' if best.preprocessing_scaling else 'disabled'}"
                f"\n  Parameters         : {best.parameters}"
                f"\n  Selection reason   : {report.selection_reason}"
                + (
                    f"\n  Error pattern      : {best.confusion_matrix_error_pattern}"
                    if best.confusion_matrix_error_pattern
                    else ""
                )
                + "\n  This is the best BASELINE model; hyperparameter optimization "
                "(Phase 6) may change it.",
            )
        )
    else:
        sections.append(
            ("Best baseline recommendation", "  None: no model produced usable metrics.")
        )

    sections.append(
        ("Performance summary", f"  {report.performance_summary}")
    )

    if report.model_evaluations:
        sections.append(
            (
                "Model comparison",
                "\n"
                + "\n".join(report.table_text.splitlines())
                + f"\n\n  Ranking rule: {report.ranking_rule}",
            )
        )

    gap = report.gap_analysis
    if gap is not None:
        gap_lines = [
            f"  {item.model_name:<24}hold-out f1={_fmt(item.f1_score)}  "
            f"cv f1={_fmt(item.cv_f1_mean)}  f1 gap={_fmt(item.f1_gap)} "
            f"({item.f1_gap_classification})  acc gap={_fmt(item.accuracy_gap)} "
            f"({item.accuracy_gap_classification})"
            for item in report.model_evaluations
        ]
        sections.append(
            (
                "Hold-out vs cross-validation",
                "\n".join(gap_lines)
                + f"\n\n  {gap.summary}"
                + f"\n  Rule       : {gap.rule}"
                + f"\n  Thresholds : close < {gap.close_threshold}, "
                f"substantial >= {gap.substantial_threshold}"
                + f"\n  Interpretation: {gap.interpretation}"
                + f"\n  Disclaimer : {gap.disclaimer}"
                + (
                    "\n  Notes      :\n" + "\n".join(f"    - {note}" for note in gap.notes)
                    if gap.notes
                    else ""
                ),
            )
        )

    if report.confusion_matrix_analysis:
        matrix_lines: List[str] = []
        for analysis in report.confusion_matrix_analysis:
            if not analysis.available:
                matrix_lines.append(f"  {analysis.model_name or 'unknown model'}:")
                matrix_lines.append(f"    {analysis.notes[0] if analysis.notes else 'not available'}")
                continue
            matrix_lines.append(f"  {analysis.model_name}:")
            matrix_lines.append(
                f"    {analysis.correct_predictions} correct / "
                f"{analysis.incorrect_predictions} incorrect of "
                f"{analysis.total_samples} (accuracy "
                f"{_fmt(analysis.accuracy_from_matrix)})"
            )
            if analysis.class_metrics:
                matrix_lines.append(
                    "    "
                    + f"{'class':<10}{'support':>8}{'tp':>5}{'fp':>5}{'fn':>5}{'tn':>5}"
                    f"{'prec':>9}{'rec':>9}{'f1':>9}"
                )
                for item in analysis.class_metrics:
                    matrix_lines.append(
                        f"    {item.label:<10}{item.support:>8}{item.true_positives:>5}"
                        f"{item.false_positives:>5}{item.false_negatives:>5}"
                        f"{item.true_negatives:>5}{_fmt(item.precision):>9}"
                        f"{_fmt(item.recall):>9}{_fmt(item.f1_score):>9}"
                    )
                matrix_lines.append(f"    {analysis.per_class_metrics_note}")
            if analysis.dominant_confusion_pairs:
                matrix_lines.append(
                    "    Dominant confusion pairs: "
                    + "; ".join(
                        f"{pair.count} '{pair.true_label}' predicted as "
                        f"'{pair.predicted_label}'"
                        for pair in analysis.dominant_confusion_pairs
                    )
                )
            matrix_lines.append(f"    Pattern: {analysis.error_pattern}")
        sections.append(("Confusion matrix analysis", "\n".join(matrix_lines)))

    timing = report.execution_time_analysis
    if timing is not None:
        sections.append(
            (
                "Execution time",
                f"  Total experiment time : {timing.total_experiment_time:.4f} s"
                f"\n  Average per model    : {timing.average_experiment_time:.4f} s"
                f"\n  Training / predict / CV : {timing.total_training_time:.4f} s / "
                f"{timing.total_prediction_time:.4f} s / "
                f"{timing.total_cross_validation_time:.4f} s"
                f"\n  Fastest model        : {timing.fastest_model or '-'} "
                f"({timing.fastest_time:.4f} s)"
                f"\n  Slowest model        : {timing.slowest_model or '-'} "
                f"({timing.slowest_time:.4f} s)"
                f"\n  Timed experiments    : {timing.timed_experiments}"
                + "\n"
                + "\n".join(f"  Note: {note}" for note in timing.notes),
            )
        )

    counts = report.experiment_count
    if counts is not None:
        sections.append(
            (
                "Experiment count",
                f"  Baseline models attempted : {counts.baseline_models_attempted}"
                f"\n  Successful experiments   : {counts.successful_experiments}"
                f"\n  Partial experiments      : {counts.partial_experiments}"
                f"\n  Failed experiments       : {counts.failed_experiments}"
                f"\n  Total experiments        : {counts.total_experiments}"
                f"\n  Hyperparameter experiments: {counts.hyperparameter_experiments}"
                + (
                    f"\n  Failed models            : {', '.join(counts.failed_model_names)}"
                    if counts.failed_model_names
                    else ""
                )
                + f"\n  {counts.note}",
            )
        )

    if report.observations:
        sections.append(
            ("Observations", "\n".join(f"  - {item}" for item in report.observations))
        )
    if report.limitations:
        sections.append(
            ("Limitations", "\n".join(f"  - {item}" for item in report.limitations))
        )
    if report.recommendations:
        sections.append(
            ("Recommendations", "\n".join(f"  - {item}" for item in report.recommendations))
        )
    if report.notes:
        sections.append(("Notes", "\n".join(f"  - {item}" for item in report.notes)))

    return _render_sections("EVALUATION REPORT", sections)


def format_optimization_report(state: AgentState) -> str:
    """Build the human readable hyperparameter optimization report.

    Phase 6: the search, the baseline-versus-optimized comparison and the
    recommendation. Every number comes from the records the Optimization Agent
    produced. Baseline and optimized scores are printed under separate headings
    so the two are never read as one measurement, and the report states plainly
    which split each number came from.
    """
    report = state.optimization_report
    sections: List[tuple] = [
        (
            "Status",
            f"  Status              : {state.optimization_status}"
            + (f"\n  Dataset             : {report.dataset_name}" if report else "")
            + (f"\n  Problem type        : {report.problem_type}" if report else "")
            + (f"\n  Target              : {report.target_column}" if report else "")
            + "\n  The search never saw the hold-out rows; they were scored once, "
            "after the parameters were fixed.",
        )
    ]

    if report is None:
        sections.append(("Optimization", "  (no optimization report was produced)"))
        return _render_sections("OPTIMIZATION REPORT", sections)

    sections.append(
        (
            "Models optimized",
            f"  Sampler            : {report.sampler} (seed={report.sampler_seed})"
            f"\n  Trials per model   : {report.n_trials_per_model}"
            f"\n  Candidate budget   : top {report.top_models} of the baseline ranking"
            f"\n  Cross-validation   : {report.cv_n_splits} fold(s)"
            + (
                f", adjusted: {report.cv_adjustment_reason}"
                if report.cv_adjustment_reason
                else ""
            )
            + "\n  Selected for search: "
            + (", ".join(report.models_selected_for_optimization) or "none")
            + "\n  Not optimized      : "
            + (", ".join(report.models_not_optimized) or "none")
            + f"\n  Candidate rule     : {report.candidate_rule}",
        )
    )

    for result in report.optimization_results:
        body = [
            f"  Status            : {result.status}",
            f"  Model type        : {result.model_type}",
            f"  Objective         : {result.objective_metric} "
            "(cross-validation on the training split)",
            f"  Trials            : {result.total_trials} total / "
            f"{result.completed_trials} completed / {result.failed_trials} failed / "
            f"{result.pruned_trials} pruned",
            f"  BASELINE cv f1    : {_fmt(result.baseline_cv_f1_mean)}",
            f"  BASELINE holdout  : f1={_fmt(result.baseline_holdout_f1_score)}  "
            f"acc={_fmt(result.baseline_holdout_accuracy)}",
            f"  BASELINE params   : {result.baseline_parameters or '-'}",
            f"  OPTIMIZED cv f1   : {_fmt(result.optimized_cv_f1_mean)}",
            f"  OPTIMIZED holdout : f1={_fmt(result.holdout_f1_score)}  "
            f"acc={_fmt(result.holdout_accuracy)}",
        ]
        if result.best_trial_number is not None:
            body.append(f"  Best trial        : #{result.best_trial_number}")
        body.append(f"  Best parameters   : {result.best_params or '-'}")
        if result.improvement is not None:
            body.append(
                f"  Improvement (cv)  : {_fmt(result.improvement)}"
                + (
                    f"  ({result.improvement_percentage}%)"
                    if result.improvement_percentage is not None
                    else ""
                )
                + f"  [{result.improvement_direction}]"
            )
        if result.holdout_f1_delta is not None:
            body.append(
                f"  Hold-out f1 delta : {_fmt(result.holdout_f1_delta)} "
                "(optimized - baseline, same split)"
            )
        body.append(f"  Search time       : {result.optimization_time:.4f} s")
        if result.cv_adjustment_reason:
            body.append(f"  Fold adjustment   : {result.cv_adjustment_reason}")
        if result.error_message:
            body.append(f"  Error             : {result.error_message}")
        if result.reason and result.status != "completed":
            body.append(f"  Reason            : {result.reason}")
        if result.improvement_note:
            body.append(f"  Note              : {result.improvement_note}")
        body.extend(f"  Note              : {note}" for note in result.notes)
        sections.append((f"Tuned: {result.model_name}", "\n".join(body)))

    if report.table_text:
        sections.append(("Baseline vs optimized", "\n".join(report.table_text.splitlines())))

    best = report.best_optimized_model
    if best is not None:
        sections.append(
            (
                "Best optimized recommendation",
                "\n".join(
                    [
                        f"  Model              : {best.best_optimized_model} "
                        f"({best.model_type})",
                        f"  BASELINE model     : {best.baseline_model_name}",
                        f"  BASELINE cv f1     : {_fmt(best.baseline_cv_f1_mean)}",
                        f"  OPTIMIZED cv f1    : {_fmt(best.optimized_cv_f1_mean)}",
                        f"  Improvement (cv)   : {_fmt(best.improvement)}"
                        + (
                            f"  ({best.improvement_percentage}%)"
                            if best.improvement_percentage is not None
                            else ""
                        )
                        + f"  [{best.improvement_direction}]",
                        f"  BASELINE holdout f1: {_fmt(best.baseline_holdout_f1_score)}",
                        f"  OPTIMIZED holdout : f1={_fmt(best.holdout_f1_score)}",
                        f"  Hold-out f1 delta : {_fmt(best.holdout_f1_delta)}",
                        f"  Best parameters   : {best.best_params}",
                        f"  Best trial        : {best.best_trial_number}",
                        f"  Trials / time     : {best.total_trials} trial(s) in "
                        f"{best.optimization_time:.4f} s over {best.cv_n_splits} CV fold(s)",
                        f"  Selection reason  : {best.selection_reason}",
                        f"  Performance       : {best.performance_summary}",
                        f"  Note              : {best.selection_note}",
                    ]
                ),
            )
        )
    else:
        sections.append(
            (
                "Best optimized recommendation",
                "  None: no optimized model produced usable hold-out metrics.\n"
                f"  {report.selection_note}",
            )
        )

    baseline = report.best_baseline_model
    if baseline is not None:
        sections.append(
            (
                "Best baseline model (Phase 5, unchanged)",
                f"  Model              : {baseline.best_baseline_model}"
                f"\n  Hold-out f1        : {_fmt(baseline.f1_score)}"
                f"\n  CV mean f1         : {_fmt(baseline.cv_f1_mean)}"
                f"\n  Parameters         : {baseline.parameters}"
                f"\n  Saved to           : {report.baseline_saved_to}"
                "\n  Phase 6 does not overwrite this result; both winners are kept.",
            )
        )
    else:
        sections.append(
            (
                "Best baseline model (Phase 5, unchanged)",
                f"  Not available ({report.baseline_saved_to}).",
            )
        )

    counts = report.counts
    if counts is not None:
        sections.append(
            (
                "Counts",
                f"  Baseline model experiments : {counts.baseline_models_attempted} "
                f"({counts.baseline_successful_experiments} successful, "
                f"{counts.baseline_failed_experiments} failed)"
                f"\n  Models optimized           : {counts.optimized_models_attempted} "
                f"({counts.successful_optimizations} successful, "
                f"{counts.failed_optimizations} failed)"
                f"\n  Optuna trials              : {counts.total_trials} total / "
                f"{counts.completed_trials} completed / {counts.failed_trials} failed / "
                f"{counts.pruned_trials} pruned"
                + (
                    f"\n  Models with failed trials  : {', '.join(counts.failed_trial_models)}"
                    if counts.failed_trial_models
                    else ""
                )
                + (
                    f"\n  Failed models              : {', '.join(counts.failed_model_names)}"
                    if counts.failed_model_names
                    else ""
                )
                + f"\n  {counts.note}",
            )
        )

    sections.append(
        (
            "How the search worked",
            f"  Objective rule    : {report.objective_rule}"
            f"\n  Selection note    : {report.selection_note}"
            f"\n  Ranking rule      : {report.ranking_rule}"
            f"\n  Hold-out note     : {report.holdout_note}",
        )
    )

    if report.observations:
        sections.append(("Observations", "\n".join(f"  - {item}" for item in report.observations)))
    if report.limitations:
        sections.append(("Limitations", "\n".join(f"  - {item}" for item in report.limitations)))
    if report.recommendations:
        sections.append(
            ("Recommendations", "\n".join(f"  - {item}" for item in report.recommendations))
        )
    if report.notes:
        sections.append(("Notes", "\n".join(f"  - {item}" for item in report.notes)))

    return _render_sections("OPTIMIZATION REPORT", sections)


def format_persistence_report(state: AgentState) -> str:
    """Build the human readable model persistence report.

    Phase 9, step 2: what was selected, why, how it was fitted and where it was
    written. Nothing here is measured - every number was recorded by an earlier
    phase, which is the whole point of the stage.
    """
    report = state.persistence_report
    if report is None:
        return _render_sections(
            "MODEL PERSISTENCE REPORT",
            [
                ("Status", f"  Persistence status : {state.persistence_status}"),
                (
                    "Artifacts",
                    f"  {config.DASHBOARD_NO_ARTIFACT}\n"
                    f"  {config.DASHBOARD_NO_ARTIFACT_GUIDANCE}",
                ),
            ],
        )

    selection = report.selection
    lines = [
        f"  Status              : {report.overall_status}",
        f"  Dataset             : {report.dataset_name}",
        f"  Target column       : {report.target_column}",
        f"  Task type           : {report.task_type}",
        f"  Final model         : {selection.model_name} ({selection.model_type})",
        f"  Selection source    : {selection.selection_source}",
        f"  Hold-out f1         : {_fmt(report.selected_f1_score)}",
        f"  Hold-out accuracy   : {_fmt(report.selected_accuracy)}",
        f"  BASELINE candidate  : {selection.baseline_model_name or '-'} "
        f"(hold-out f1 {_fmt(selection.baseline_f1_score)})",
        f"  OPTIMIZED candidate : {selection.optimized_model_name or '-'} "
        f"(hold-out f1 {_fmt(selection.optimized_f1_score)})",
        f"  Optimization used   : {selection.optimization_used} "
        f"(status: {selection.optimization_status})",
        f"  Final fit strategy  : {report.final_fit_strategy}",
        f"  {report.final_fit_note}",
        f"  Features            : {report.feature_count} "
        f"({len(state.preprocessing_numerical_features or [])} numerical, "
        f"{len(state.preprocessing_categorical_features or [])} categorical)",
        f"  Classes             : {report.class_count}",
        f"  Best parameters     : {selection.best_params or '-'}",
        f"  Model artifact      : {report.model_artifact}",
        f"  Metadata artifact   : {report.metadata_artifact}",
        f"  Results artifact    : {report.results_artifact}",
        "",
        f"  Selection reason: {selection.selection_reason}",
        f"  Ranking rule    : {selection.selection_rule}",
    ]
    sections = [("Status and selection", "\n".join(lines))]
    sections.append(("Artifacts", "\n".join(_artifact_lines(state))))

    if report.observations:
        sections.append(
            ("Observations", "\n".join(f"  - {item}" for item in report.observations))
        )
    if report.limitations:
        sections.append(
            ("Limitations", "\n".join(f"  - {item}" for item in report.limitations))
        )
    if report.notes:
        sections.append(("Notes", "\n".join(f"  - {item}" for item in report.notes)))
    return _render_sections("MODEL PERSISTENCE REPORT", sections)


def _artifact_lines(state: AgentState) -> List[str]:
    """Describe the three artifacts as they now exist on disk.

    Args:
        state: The finished workflow state.

    Returns:
        One line per artifact, marked present or missing.
    """
    paths = artifact_paths()
    rows = [
        ("model", paths.model, "the fitted preprocessing + estimator pipeline"),
        ("metadata", paths.metadata, "what the model is and how it scored"),
        ("results", paths.results, "the dashboard read model"),
    ]
    lines = [f"  Directory: {paths.directory}"]
    for label, path, description in rows:
        if path.exists():
            size = path.stat().st_size
            lines.append(
                f"  [{label:<8}] {path.name:<28} {size:>8} bytes  ({description})"
            )
        else:
            lines.append(f"  [{label:<8}] {path.name:<28} {'MISSING':>8}  ({description})")
    return lines


def _render_sections(title: str, sections: List[tuple]) -> str:
    """Render numbered report sections under a title banner.

    Numbering happens here so that an omitted (empty) section never leaves a
    gap in the report.
    """
    parts: List[str] = [SEPARATOR, title, SEPARATOR]
    for index, (section_title, body) in enumerate(sections, start=1):
        parts.append(_section(f"{index}. {section_title}") + "\n" + body)
    parts.append(SEPARATOR)
    return "\n".join(parts)


def _fmt(value: Optional[float]) -> str:
    """Format a metric for the CLI report, showing '-' when it is missing."""
    return "-" if value is None else f"{value:.4f}"


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent ML System - Phase 11 (controlled conventional-vs-autonomous benchmark).",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path to the CSV dataset (default: {DEFAULT_DATASET.name}).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. Omit it to let the agent detect it heuristically.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking for this run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the structured analysis summary as JSON.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Only run the Data Analysis Agent (Phase 2).",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Stop after the Model Selection Agent (no training).",
    )
    parser.add_argument(
        "--no-cross-validation",
        action="store_true",
        help="Skip baseline cross-validation and report the hold-out metrics only.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Stop after the Training Agent (no evaluation report).",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Stop after the Evaluation Agent (skip the Optuna hyperparameter search).",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help=(
            "Run the pipeline through the LangGraph orchestration layer instead of "
            "the direct sequential calls. The agents are identical either way; the "
            "graph only decides order, failure handling and whether to optimize."
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Enable the optional local LLM layer (Phase 8), which asks a local Ollama "
            "model to explain the results. Explanatory only: it cannot change a "
            "metric, a model or a routing decision, and the run completes normally "
            "when Ollama is not installed. Requires --graph."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help=(
            "Skip the Model Persistence Agent, so no artifact is written and the "
            "Streamlit dashboard has nothing to display. Persistence is on by "
            "default; it only writes files and cannot change a result."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run the Phase 11 controlled benchmark: the conventional workflow "
            "(tools only, manual grid over the five candidate families) versus "
            "the autonomous workflow (the existing graph, LLM off, Optuna on) on "
            "the same committed datasets, then write reports/phase11_comparison.*"
            " and the three comparison charts."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "With --benchmark: run both committed datasets explicitly. Without "
            "--dataset this is also the default behaviour -- the flag makes it "
            "unambiguous for scripts."
        ),
    )
    return parser.parse_args(argv)


def _print_tracking_notice(no_mlflow: bool) -> None:
    """Explain what happened to MLflow tracking for this run.

    The tracking result is deliberately not threaded through the workflow result:
    Phase 10 adds no field to ``WorkflowResult``, because the report and the
    dashboard are defined in terms of machine learning outcomes and the tracking
    status is an observability fact about the run rather than part of it. This
    function therefore asks the tracking module directly, which keeps the two
    concerns separate and costs one cheap store read.

    Args:
        no_mlflow: Whether ``--no-mlflow`` was passed.
    """
    if no_mlflow:
        print(
            "\nExperiment tracking skipped (--no-mlflow). Every result above was "
            "computed and reported exactly as it would have been."
        )
        return
    try:
        from tools import mlflow_tools  # noqa: PLC0415 - optional dependency

        summary = mlflow_tools.experiment_summary()
    except Exception:  # noqa: BLE001 - never let a notice break the report
        return
    if summary.get("runs"):
        latest = summary["runs"][0]
        print(
            f"\nRecorded in MLflow experiment '{summary['experiment_name']}' "
            f"(run {latest['run_id']}, status {latest['status']})."
        )
    elif not summary.get("enabled"):
        # Tracking was switched off by MLFLOW_ENABLED rather than by the flag.
        # Saying "not recorded" here would read as a failure that did not happen.
        print(
            "\nExperiment tracking disabled by configuration (MLFLOW_ENABLED). "
            "Every result above was computed and reported exactly as it would "
            "have been."
        )
    elif summary.get("installed"):
        print(
            "\nExperiment tracking is enabled but this run was not recorded; "
            "see the log for the reason."
        )
    else:
        print(
            "\nMLflow experiment tracking is unavailable (the package is not "
            "installed). Every result above is unaffected."
        )


def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Run the Phase 11 benchmark from the command line.

    ``--benchmark`` dispatches here before any agent is built: the benchmark is
    not another graph run, it is a controlled experiment that runs both sides
    itself.

    Dataset selection:
    - ``--all``: both committed datasets (explicit form of the default);
    - ``--dataset classification|multiclass``: exactly that committed dataset;
    - otherwise: both committed datasets.

    Args:
        args: The parsed command line arguments.

    Returns:
        0 on a complete benchmark (a per-dataset failure inside a workflow is
        reported on that dataset's record and still lets the rest complete with
        a non-zero indication; a hard, top-level failure returns 1).
    """
    from benchmarks import benchmark_config
    from benchmarks.benchmark_runner import format_comparison_text
    from benchmarks.benchmark_runner import run_benchmark

    accepted_keys = set(benchmark_config.ALL_DATASET_KEYS)
    if args.all:
        dataset_keys = list(benchmark_config.ALL_DATASET_KEYS)
    elif args.dataset in accepted_keys:
        dataset_keys = [args.dataset]
    else:
        dataset_keys = list(benchmark_config.ALL_DATASET_KEYS)

    print(
        "\nPhase 11 benchmark: conventional workflow (tools only, manual grid) vs "
        "autonomous workflow (graph + Optuna) on the committed datasets.\n"
        "Runs both workflows on the same split (random_state=42) and compares the "
        "same metrics (weighted average), then writes reports/phase11_comparison.\n"
    )
    try:
        suite = run_benchmark(
            dataset_keys,
            n_trials=config.OPTUNA_N_TRIALS,
            benchmark_mLflow=not args.no_mlflow,
        )
    except Exception as error:  # noqa: BLE001 - top-level guard for the CLI
        print(f"\nERROR: the benchmark could not run: {type(error).__name__}: {error}")
        return 1

    for comparison in suite.datasets:
        print(format_comparison_text(comparison))
        print()

    if suite.files_written:
        print("Written files:")
        for path in suite.files_written:
            print(f"  - {path}")
        print()

    failures = [d for d in suite.datasets if d.autonomous.status != "success"]
    if failures:
        print(
            "The benchmark completed but the following dataset(s) had a "
            "non-clean autonomous status: "
            + ", ".join(d.dataset_key for d in failures)
        )
        return 1
    print("Benchmark complete.")
    return 0


def run_graph_workflow(args: argparse.Namespace) -> int:
    """Run the whole pipeline through the LangGraph orchestration layer.

    The graph calls the same six agents the sequential path calls, so the machine
    learning results are identical. What the graph adds is the workflow status,
    the per-node outcome, the retry accounting and the recorded optimization
    decision.

    With ``--llm`` the Phase 8 interpretation nodes join the graph. That flag
    changes nothing measurable: the LLM layer only appends an explanation record
    per step, and every failure it can have becomes a printed status.

    Args:
        args: The parsed command line arguments.

    Returns:
        0 when the workflow completed or finished partially, 1 when a required
        node failed.
    """
    print(
        "\nRunning through the LangGraph orchestration layer "
        "(the six agents are unchanged)."
    )

    if args.llm:
        print(
            "\nOptional local LLM layer enabled (--llm). It explains results only; "
            "it cannot change a metric, a model or a routing decision, and the run "
            "completes normally if Ollama is not installed."
        )

    result = run_workflow(
        dataset_path=args.dataset,
        target_column=args.target,
        enable_optimization=not args.no_optimize,
        cross_validation=not args.no_cross_validation,
        enable_llm=bool(args.llm),
        enable_persistence=not args.no_save_model,
        enable_tracking=not args.no_mlflow,
    )

    print("\n" + format_workflow_report(result))
    _print_tracking_notice(args.no_mlflow)

    if args.json:
        print("\nStructured workflow summary (JSON):")
        print(json.dumps(result.to_dict(), indent=2))

    if not result.workflow_status.is_success:
        print(
            f"\nThe workflow ended with status '{result.workflow_status.value}'"
            + (f" at node '{result.failed_node}'." if result.failed_node else ".")
        )
        print(
            "The machine learning state was kept as it was when the failure "
            "occurred; nothing already computed was discarded."
        )
        return 1

    if args.llm:
        _print_llm_fallback_notice(result)

    print(
        "\nThe core machine learning pipeline does not require an LLM: every number "
        "above was produced by the deterministic agents."
    )
    return 0


def _print_llm_fallback_notice(result) -> None:
    """Print a short notice about the state of the optional LLM layer.

    Silent degradation is the failure mode to avoid here: the user asked for
    explanations, so it has to be obvious when they did not get them. The notice
    distinguishes "disabled" from "tried and could not be used", because the two
    call for different actions.

    Args:
        result: The finished :class:`~graph.state.WorkflowResult`.
    """
    if not result.llm_results:
        return

    unavailable = [entry for entry in result.llm_results if entry.failed]
    completed = [entry for entry in result.llm_results if entry.succeeded]

    print("\n" + SEPARATOR)
    print("Optional local LLM layer")
    print(SEPARATOR)
    print(
        f"  Steps attempted : {len(result.llm_results)}\n"
        f"  Generated       : {len(completed)}\n"
        f"  Unavailable     : {len(unavailable)}"
    )
    print(f"  {config.LLM_ROLE_STATEMENT}")

    if not unavailable:
        return

    print("\n  Fallback (the pipeline results above are unaffected):")
    for entry in unavailable:
        print(f"    - {entry.role.display_name}: {entry.status.value}")
        if entry.detail:
            print(f"      {entry.detail}")

    if config.LLM_FALLBACK_NOTE:
        print(f"\n  {config.LLM_FALLBACK_NOTE}")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the application."""
    args = parse_arguments(argv)

    print(SEPARATOR)
    print("Autonomous Multi-Agent ML System")
    print(SEPARATOR)
    print("\nProject Status: Phase 11 - controlled benchmark: conventional vs autonomous")
    print(
        "\nImplemented components:\n"
        "  - Data Analysis Agent (agents/data_analysis_agent.py)\n"
        "  - Preprocessing Agent (agents/preprocessing_agent.py)\n"
        "  - Model Selection Agent (agents/model_selection_agent.py)\n"
        "  - Training Agent (agents/training_agent.py)\n"
        "  - Evaluation Agent (agents/evaluation_agent.py)\n"
        "  - Optimization Agent (agents/optimization_agent.py)\n"
        "  - Model Persistence Agent (agents/persistence_agent.py)\n"
        "  - Reusable data tools (tools/data_tools.py)\n"
        "  - Reusable preprocessing tools (tools/preprocessing_tools.py)\n"
        "  - Controlled model registry (tools/model_tools.py)\n"
        "  - Reusable training tools (tools/training_tools.py)\n"
        "  - Reusable evaluation tools (tools/evaluation_tools.py)\n"
        "  - Reusable optimization tools (tools/optimization_tools.py)\n"
        "  - Reusable persistence tools (tools/persistence_tools.py)\n"
        "  - Reusable dashboard helpers (tools/dashboard_tools.py)\n"
        "  - Local experiment tracking (tools/mlflow_tools.py)\n"
        "  - LangGraph orchestration (graph/): state, nodes, routing, workflow\n"
        "  - Optional local LLM layer (llm/): client, prompts, parser, service\n"
        "  - Controlled conventional-vs-autonomous benchmark (benchmarks/)\n"
        "  - Typed shared state (utils/state.py)\n"
        "\nRun with --graph to use the LangGraph orchestration layer; without it\n"
        "the direct sequential path runs exactly as in Phase 6.\n"
        "\nThe local LLM layer is optional and OFF by default. Add --llm to\n"
        "--graph to ask a local Ollama model to explain the results; it explains\n"
        "only, and the run completes normally when Ollama is not installed.\n"
        "The core machine learning pipeline does not require an LLM.\n"
        "\nModel persistence is ON by default. Every run writes the fitted\n"
        "pipeline and its metadata, and 'streamlit run dashboard.py' then reads\n"
        "them. The dashboard never retrains a model. Use --no-save-model to skip\n"
        "the write.\n"
        "\nPhase 10 records each --graph run in a local MLflow tracking store under\n"
        "experiments/tracking. MLflow only records decisions the agents have already\n"
        "made; it never selects a model or changes a metric. Use --no-mlflow to skip\n"
        "it, and see the README for how to inspect the history.\n"
        "\nPhase 11 adds 'python app.py --benchmark': a controlled comparison of the\n"
        "conventional workflow (tools only, manual grid) against the autonomous\n"
        "workflow (graph + Optuna) on both committed datasets, writing JSON, CSV\n"
        "and PNG comparison charts under reports/.\n"
        "\nNot implemented yet (later phases):\n"
        "  - Model registry, deployment"
    )

    if args.benchmark:
        return run_benchmark_cli(args)

    state = AgentState()

    if args.graph:
        # The graph owns the whole run: it invokes the agents itself, decides the
        # order, handles failures and reports the workflow status. Only the
        # dataset banner above is shared with the sequential path.
        return run_graph_workflow(args)

    if args.llm:
        # The interpretation nodes are wired into the graph, so --llm without
        # --graph would be a flag that silently does nothing. Say so instead.
        print(
            "\nNOTE: --llm has no effect without --graph. The optional local LLM "
            "layer is wired into the LangGraph path, and the sequential path below "
            "is unchanged from Phase 6.\n"
            "Run 'python app.py --graph --llm' to use it."
        )

    try:
        state = DataAnalysisAgent().run(
            state, dataset_path=args.dataset, target_column=args.target
        )
    except DataToolError as error:
        print(f"\nERROR: {error}")
        print("The Data Analysis Agent stopped. No dataset was modified.")
        return 1

    print("\n" + format_analysis_report(state))

    if args.no_preprocess:
        print("\nStopped after the Data Analysis Agent (--no-preprocess).")
        return 0

    try:
        state = PreprocessingAgent().run(state)
    except (DataToolError, PreprocessingError) as error:
        print(f"\nERROR: {error}")
        print("The Preprocessing Agent stopped. No dataset was modified.")
        return 1
    print("\n" + format_preprocessing_report(state))

    try:
        state = ModelSelectionAgent().run(state)
    except (PreprocessingError, ModelToolError) as error:
        print(f"\nERROR: {error}")
        print("The Model Selection Agent stopped. No model was trained.")
        return 1
    print("\n" + format_model_selection_report(state))

    if args.no_train:
        print("\nStopped after the Model Selection Agent (--no-train).")
        return 0

    try:
        state = TrainingAgent().run(
            state, cross_validation=not args.no_cross_validation
        )
    except (PreprocessingError, ModelToolError, TrainingToolError) as error:
        print(f"\nERROR: {error}")
        print("The Training Agent stopped. No dataset was modified.")
        return 1
    print("\n" + format_training_report(state))

    if args.no_evaluate:
        print("\nStopped after the Training Agent (--no-evaluate).")
        if args.json:
            print("\nStructured summary (JSON):")
            print(json.dumps(state.to_dict(), indent=2))
        return 0

    try:
        state = EvaluationAgent().run(state)
    except (PreprocessingError, ModelToolError, TrainingToolError, EvaluationToolError) as error:
        print(f"\nERROR: {error}")
        print("The Evaluation Agent stopped. No model was retrained.")
        return 1
    print("\n" + format_evaluation_report(state))

    if args.no_optimize:
        print("\nStopped after the Evaluation Agent (--no-optimize).")
        if args.json:
            print("\nStructured summary (JSON):")
            print(json.dumps(state.to_dict(), indent=2))
        return 0

    try:
        state = OptimizationAgent().run(state)
    except (
        PreprocessingError,
        ModelToolError,
        TrainingToolError,
        EvaluationToolError,
        OptimizationToolError,
    ) as error:
        print(f"\nERROR: {error}")
        print("The Optimization Agent stopped. The Phase 5 baseline result is unchanged.")
        return 1
    print("\n" + format_optimization_report(state))

    return _persist_and_report(state, args)


def _persist_and_report(state: AgentState, args: argparse.Namespace) -> int:
    """Run the Model Persistence Agent and print the final section of the run.

    Extracted because both execution paths now end here, and the difference
    between them is only *where* the state came from - LangGraph or the direct
    sequential calls. The artifact is written the same way either way, so the
    dashboard does not care which path produced it.

    Args:
        state: The finished machine learning state.
        args: The parsed command line arguments.

    Returns:
        0 when the artifact was written or persistence was skipped, 1 when it
        failed. A failure here never changes the exit code of the machine learning
        work that already succeeded, so it is reported as its own outcome.
    """
    if args.no_save_model:
        state.persistence_status = PersistenceStatus.SKIPPED.value
        print(
            "\nSkipped model persistence (--no-save-model). No artifact was written, "
            "so the Streamlit dashboard will have nothing to display."
        )
        return 0

    try:
        state = PersistenceAgent().run(state)
    except PersistenceError as error:
        print(f"\nERROR: {error}")
        print(
            "The Model Persistence Agent could not write its artifact. The results "
            "above are unchanged and remain valid."
        )
        return 1

    print("\n" + format_persistence_report(state))

    if args.json:
        print("\nStructured summary (JSON):")
        print(json.dumps(state.to_dict(), indent=2))

    paths = artifact_paths()
    print(
        "\nThe dashboard reads these files and does not retrain anything:\n"
        f"  streamlit run dashboard.py      (after this run finished)\n"
        f"  artifacts: {paths.model.name}, {paths.metadata.name}, {paths.results.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())