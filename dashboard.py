"""Streamlit dashboard for the Autonomous Multi-Agent ML System (Phase 9).

Run it with::

    python app.py --graph          # writes the artifacts, in a separate terminal
    streamlit run dashboard.py     # displays them

What this file is: presentation. Every number it shows was produced by the
workflow run that wrote ``models/``, and this file does not import a single agent
or graph module, so opening the dashboard cannot start a workflow, cannot retrain
a model and cannot change a result. Re-running the app on every keystroke is
therefore free.

What this file is not: a second machine learning pipeline. It calls
``pipeline.predict`` on the artifact that was saved and nothing else - the
imputation, encoding and scaling all live inside that saved pipeline, so none of
them are repeated here.

Deliberately absent, and worth naming because it is a design decision rather than
an oversight:

* **no file uploader.** A ``.joblib``/``.pkl`` file is a pickle; loading one
  executes code inside the Streamlit process. The dashboard loads exactly one
  file - the artifact this project wrote, at a path taken from configuration - so
  there is no code path here that accepts a model from a user.
* **no training button.** Training is a command-line operation that writes
  artifacts. A dashboard that trained on refresh would re-run a search on every
  rerun and would make the displayed metrics depend on when the page was opened.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from tools import dashboard_tools as helpers
from utils import config


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
# ``set_page_config`` is called from ``main()`` rather than at import time. It has
# to be the first Streamlit command a run makes, and it is only a *run* that has
# a page to configure; importing this module - as the test suite does - must not
# talk to Streamlit at all.


@st.cache_data(show_spinner=False)
def cached_dashboard_state() -> helpers.DashboardState:
    """Load the artifacts once per session, not on every rerun.

    ``st.cache_data`` is keyed on the function's own source, so an interactive
    rerun reuses the loaded artifact. The pipeline object itself lives in the
    returned dataclass and is used only by ``run_prediction``; the cached *data*
    around it - metadata and report sections - is what the tables are built from.

    Returns:
        The loaded :class:`~tools.dashboard_tools.DashboardState`.
    """
    return helpers.load_dashboard_state()


def show_rows(rows: List[Dict[str, str]], key: str) -> None:
    """Render label/value rows as a two-column table.

    Args:
        rows: The rows to show.
        key: A unique key, so two tables with the same content do not collide in
            Streamlit's widget namespace.
    """
    if not rows:
        return
    st.dataframe(rows, width="stretch", hide_index=True, key=key)


# --------------------------------------------------------------------------- #
# Section 1: header. Sections 2 to 11 follow, numbered to match this one.
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """Show the title, the pipeline it reports on, and its own section index.

    The index is listed because the page numbers its panels 2 to 11: this title
    is section 1, and a reader arriving mid-page would otherwise have no way to
    tell what number they were looking at.
    """
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_SUBTITLE)
    st.info(
        "This dashboard reads the artifacts written by a workflow run. It never "
        "trains, tunes or re-evaluates a model, so opening it cannot change any "
        "result."
    )
    st.caption(
        "Sections: 1. Header (this page) - 2. System status - 3. Dataset "
        "overview - 4. Model comparison - 5. Evaluation - 6. Confusion matrix "
        "- 7. Optimization - 8. LLM interpretation - 9. Prediction - "
        "10. Model information - 11. Experiment tracking."
    )


# --------------------------------------------------------------------------- #
# Sections 2 to 11
# --------------------------------------------------------------------------- #
def render_system_status(state: helpers.DashboardState) -> None:
    """Show whether a model is available and which one."""
    """Show whether a model is available and which one."""
    st.subheader("2. System status")
    if not state.available:
        _render_no_artifact(state)
        return

    rows = helpers.build_status_section(state)
    show_rows(rows, key="status_rows")
    if state.results_available:
        st.caption(
            "Model available, dataset, evaluation and optimization sections are all "
            "read from the artifacts of the last run."
        )
    else:
        st.warning(
            f"{config.DASHBOARD_INVALID_ARTIFACT} The saved model loaded, but the "
            "report bundle could not be read, so only the model sections are shown."
        )


def _render_no_artifact(state: helpers.DashboardState) -> None:
    """Show the reason there is nothing to display.

    Two distinct situations produce this screen, and they need different actions,
    so they are never collapsed into one message.

    Args:
        state: The loaded dashboard state, whose ``problem`` distinguishes them.
    """
    st.warning(config.DASHBOARD_NO_ARTIFACT)
    if state.problem:
        st.error(state.problem)
    st.markdown(f"**What to do:** {config.DASHBOARD_NO_ARTIFACT_GUIDANCE}")
    st.markdown(
        "The dashboard starts and stays usable in this state on purpose: an absent "
        "artifact is a normal situation, not an error."
    )


def render_dataset(state: helpers.DashboardState) -> None:
    """Show the recorded dataset facts, without opening the CSV."""
    st.subheader("3. Dataset overview")
    section = helpers.build_dataset_section(state)
    if not section.get("available"):
        st.info(
            "No dataset summary was stored with this artifact. The dashboard does "
            "not read datasets, so it has nothing to show here."
        )
        return

    left, right = st.columns(2)
    with left:
        show_rows(section["metrics"], key="dataset_metrics")
    with right:
        if section["class_distribution"]:
            st.markdown("**Class distribution**")
            st.dataframe(
                section["class_distribution"],
                width="stretch",
                hide_index=True,
                key="class_distribution",
            )
        else:
            st.info("No class distribution was recorded.")

    with st.expander("Feature columns and preprocessing decisions"):
        st.markdown(
            f"**Numerical ({len(section['numerical_features'])})**: "
            + (", ".join(section["numerical_features"]) or "-")
        )
        st.markdown(
            f"**Categorical ({len(section['categorical_features'])})**: "
            + (", ".join(section["categorical_features"]) or "-")
        )
        if section["excluded_features"]:
            st.markdown("**Excluded columns**")
            st.dataframe(
                section["excluded_features"],
                width="stretch",
                hide_index=True,
                key="excluded_features",
            )
        if section["problem_type_reason"]:
            st.caption(f"Problem type: {section['problem_type_reason']}")


def render_comparison(state: helpers.DashboardState) -> None:
    """Show the baseline and optimized models side by side."""
    st.subheader("4. Model comparison")
    section = helpers.build_comparison_section(state)
    if not section.get("available"):
        st.info("No model comparison was stored with this artifact.")
        return

    st.dataframe(
        section["rows"], width="stretch", hide_index=True, key="comparison_table"
    )

    if section["f1_chart"]:
        st.markdown("**Hold-out F1**")
        # No ``key``: ``st.bar_chart`` does not accept one. Streamlit derives the
        # element identity from the chart's contents, so the two bars of a model
        # that appears as both baseline and tuned stay distinct anyway.
        st.bar_chart(section["f1_chart"], x="Model", y="F1", color="Kind")
        st.caption(
            "A model that was both a baseline and tuned has two bars at the same "
            "position, one per kind."
        )

    st.success(f"Persisted model: **{section['persisted_label']}**")
    st.caption(section["persisted_note"])
    st.caption(
        f"{section['baseline_count']} baseline model(s) and "
        f"{section['optimized_count']} tuned model(s). BASELINE and OPTIMIZED are "
        "kept apart on purpose: the two come from different phases and are not "
        "interchangeable measurements."
    )


def render_evaluation(state: helpers.DashboardState) -> None:
    """Show the recorded evaluation of the selected model."""
    st.subheader("5. Evaluation")
    section = helpers.build_evaluation_section(state)
    if not section.get("available"):
        st.info("No evaluation result was stored with this artifact.")
        return

    st.markdown(f"**Selected model:** {section['model_name']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Hold-out metrics**")
        show_rows(section["metrics"], key="evaluation_metrics")
    with right:
        st.markdown("**Cross-validation**")
        show_rows(section["cv_metrics"], key="evaluation_cv")

    source = section["metrics_source"]
    if source == "optimization_holdout":
        st.caption(
            "These two numbers were measured by Phase 6 after the hyperparameter "
            "search, on the same hold-out split. Precision, recall and a confusion "
            "matrix were not recorded for a tuned model, so they are shown as '-'"
            " rather than filled in with the untuned model's values."
        )
    elif source == "evaluation_holdout":
        st.caption("These numbers were measured by Phase 4 on the hold-out split.")

    gap = section.get("generalization_gap")
    if gap:
        st.markdown("**Generalization gap**")
        st.write(gap["text"])
        if section.get("interpretation"):
            st.caption(section["interpretation"])
        if section.get("gap_disclaimer"):
            st.caption(section["gap_disclaimer"])
    else:
        st.info(
            "No hold-out versus cross-validation comparison is available for the "
            "selected model, because one of the two numbers was never recorded."
        )

    if section.get("observations"):
        with st.expander("Evaluation observations"):
            for item in section["observations"]:
                st.markdown(f"- {item}")


def render_confusion_matrix(state: helpers.DashboardState) -> None:
    """Show the stored confusion matrix, or say clearly why there is none."""
    st.subheader("6. Confusion matrix")
    section = helpers.build_confusion_section(state)
    if not section.get("available"):
        st.info(section.get("note") or config.DASHBOARD_NO_ARTIFACT)
        return

    st.markdown(
        f"**{section['correct']} correct / {section['incorrect']} incorrect of "
        f"{section['total']}** hold-out samples "
        f"(accuracy {helpers.format_metric(section['accuracy_from_matrix'])})."
    )
    st.caption(
        "Binary task." if section["is_binary"] else f"{len(section['labels'])}-class task."
    )

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            section["frame"], width="stretch", hide_index=True, key="confusion_long"
        )
    with right:
        st.markdown("**Counts**")
        st.dataframe(
            section["matrix"], width="stretch", hide_index=True, key="confusion_counts"
        )

    if section.get("per_class"):
        st.markdown("**Per class**")
        st.dataframe(
            section["per_class"], width="stretch", hide_index=True, key="confusion_class"
        )
    if section.get("note"):
        st.caption(section["note"])


def render_optimization(state: helpers.DashboardState) -> None:
    """Show what the hyperparameter search did, without rerunning it."""
    st.subheader("7. Optimization")
    section = helpers.build_optimization_section(state)
    if not section.get("available"):
        st.info(section.get("note") or config.DASHBOARD_NO_ARTIFACT)
        return

    left, right = st.columns(2)
    with left:
        status_rows = [
            {"label": "Search performed", "value": "yes"},
            {"label": "Selected model", "value": str(section["selected_model"])},
            {"label": "Models searched", "value": str(section["models_searched"])},
            {
                "label": "Trials per model",
                "value": str(section["trials_per_model"]),
            },
            {
                "label": "Total trials (all models)",
                "value": str(section["total_trials"]),
            },
            {
                "label": "Trials for the winner",
                "value": str(section["winner_trials"]),
            },
            {"label": "Cross-validation folds", "value": str(section["cv_n_splits"])},
            {"label": "Objective metric", "value": str(section["objective_metric"])},
            {
                "label": "Sampler",
                "value": f"{section['sampler']} (seed={section['sampler_seed']})",
            },
        ]
        if section.get("best_trial_number") is not None:
            status_rows.append(
                {
                    "label": "Best trial",
                    "value": f"{section['best_trial_number']} of "
                    f"{section['winner_trials']}",
                }
            )
        show_rows(status_rows, key="optimization_status")
    with right:
        show_rows(
            [
                {
                    "label": "Baseline hold-out F1",
                    "value": helpers.format_metric(section["baseline_f1"]),
                },
                {
                    "label": "Optimized hold-out F1",
                    "value": helpers.format_metric(section["optimized_f1"]),
                },
                {"label": "Improvement", "value": section["improvement_text"]},
                {"label": "Best parameters", "value": section["best_params_text"]},
            ],
            key="optimization_scores",
        )

    st.dataframe(section["rows"], width="stretch", hide_index=True, key="optimization_rows")
    if section.get("note"):
        st.caption(section["note"])
    if section.get("holdout_note"):
        st.caption(section["holdout_note"])
    st.caption(
        "These numbers were recorded during the workflow run. The dashboard does "
        "not create or resume an Optuna study, so nothing here can trigger a search."
    )


def render_llm(state: helpers.DashboardState) -> None:
    """Show the stored local-model interpretations, if the run produced any."""
    st.subheader("8. LLM interpretation")
    section = helpers.build_llm_section(state)
    if not section.get("available"):
        st.info(section["note"])
        st.caption(section["statement"])
        return

    for entry in section["entries"]:
        with st.expander(f"{entry['role_display_name']} - {entry['status']}", expanded=True):
            st.caption(f"Model: {entry['model']}")
            if entry["interpretation"]:
                st.markdown(entry["interpretation"])
            if entry["observations"]:
                st.markdown("**Observations**")
                for item in entry["observations"]:
                    st.markdown(f"- {item}")
            if entry["recommendations"]:
                st.markdown("**Recommendations**")
                for item in entry["recommendations"]:
                    st.markdown(f"- {item}")
            if entry["confidence"] is not None:
                st.markdown(f"**Confidence:** {entry['confidence']}")
                if entry["confidence_note"]:
                    st.caption(entry["confidence_note"])
            if entry["detail"]:
                st.caption(entry["detail"])

    if section.get("failed"):
        st.caption(config.LLM_FALLBACK_NOTE or "")
    st.caption(section["statement"])


def render_prediction(state: helpers.DashboardState) -> None:
    """Predict one row using the saved pipeline.

    The form is built from the persisted metadata and the prediction is a single
    ``predict`` call on the artifact. Nothing in this section fits, transforms or
    encodes anything.
    """
    st.subheader("9. Prediction")
    section = helpers.build_prediction_section(state)
    if not section.get("available"):
        st.info(section.get("note") or config.DASHBOARD_NO_ARTIFACT)
        return

    st.caption(
        f"Enter values for the {len(section['fields'])} feature(s) the persisted "
        f"pipeline was fitted on, then press Predict. The model is "
        f"{section['model_name']}; the target column "
        f"'{section['target_column']}' is not an input."
    )

    with st.form("prediction_form", clear_on_submit=False):
        values: Dict[str, Any] = {}
        columns = st.columns(2)
        for index, field_spec in enumerate(section["fields"]):
            with columns[index % 2]:
                if field_spec.is_numeric:
                    values[field_spec.name] = st.number_input(
                        field_spec.name,
                        value=float(field_spec.default or 0.0),
                        key=f"num_{field_spec.name}",
                    )
                elif field_spec.options:
                    values[field_spec.name] = st.selectbox(
                        field_spec.name,
                        list(field_spec.options),
                        key=f"cat_{field_spec.name}",
                    )
                else:
                    values[field_spec.name] = st.text_input(
                        field_spec.name,
                        value=str(field_spec.default or ""),
                        key=f"cat_{field_spec.name}",
                    )
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.caption(config.DASHBOARD_NO_PREDICTION)
        return

    outcome = helpers.run_prediction(state, section["fields"], values)
    if not outcome.ok:
        st.error(outcome.message)
        if outcome.detail:
            st.caption(outcome.detail)
        return

    st.success(f"Predicted class: **{outcome.predicted_class}**")
    if outcome.supports_probability and outcome.probabilities:
        st.markdown("**Class probabilities**")
        st.dataframe(
            outcome.probabilities, width="stretch", hide_index=True, key="probabilities"
        )
        if outcome.confidence is not None:
            st.caption(
                f"Highest probability: {helpers.format_metric(outcome.confidence)}"
            )
    else:
        st.caption(
            "This estimator does not expose predict_proba, so no probabilities are "
            "shown. The predicted class comes from the saved pipeline unchanged."
        )


def render_model_information(state: helpers.DashboardState) -> None:
    """Show the artifact facts."""
    st.subheader("10. Model information")
    rows = helpers.build_model_information_section(state)
    if not rows:
        st.info(config.DASHBOARD_NO_ARTIFACT)
        return
    show_rows(rows, key="model_information")


@st.cache_data(show_spinner=False)
def cached_tracking_summary() -> Dict[str, Any]:
    """Return the MLflow tracking summary, or None when it cannot be obtained.

    Cached for the same reason :func:`cached_dashboard_state` is: the tracking
    store is read on every page render otherwise, and a read is a database
    query. The cache is what makes "the dashboard does not query MLflow on every
    rerun" true rather than merely intended.

    The import and the call are both inside a guard. This section is optional
    information, and a missing or broken mlflow installation must cost the reader
    this one section and nothing else - not an exception, and not a broken page.

    Returns:
        The summary mapping from ``mlflow_tools.experiment_summary``, or None when
        the tracking module could not be imported or the read failed.
    """
    try:
        from tools import mlflow_tools  # noqa: PLC0415 - optional dependency

        return mlflow_tools.experiment_summary()
    except Exception:  # noqa: BLE001 - an optional section must never break the page
        return None


def render_experiment_tracking() -> None:
    """Show what MLflow recorded, or that it recorded nothing.

    Section 11. Placed after the model information because it reports on runs
    that produced those artifacts, so it reads as the summary of the Phase 9
    output rather than as part of the pipeline narrative.
    """
    st.subheader("11. Experiment tracking")
    section = helpers.build_experiment_section(cached_tracking_summary())

    if not section["available"]:
        st.info(section["note"] or config.DASHBOARD_NO_EXPERIMENT_TRACKING)
        return

    latest = section["latest"]
    st.write(
        f"Experiment **{section['experiment_name']}** · latest run "
        f"`{latest['short_run_id']}` · MLflow status **{latest['status']}** · "
        f"workflow status **{latest['workflow_status']}** · "
        f"{latest['start_time']}"
    )
    st.caption(f"Tracking store: {section['tracking_uri']}")

    columns = st.columns(4)
    columns[0].metric("Selected model", latest["model"])
    columns[1].metric("Hold-out F1", latest["f1_score"])
    columns[2].metric("Hold-out accuracy", latest["accuracy"])
    columns[3].metric("Optimization", latest["optimization_status"])

    if section["runs"]:
        with st.expander(f"Run history (last {len(section['runs'])})"):
            st.dataframe(
                [
                    {
                        "Run": run["short_run_id"],
                        "Started": run["start_time"],
                        "Task": run["task_type"],
                        "Model": run["model"],
                        "F1": run["f1_score"],
                        "Accuracy": run["accuracy"],
                        "Optimization": run["optimization"],
                        "Status": run["status"],
                    }
                    for run in section["runs"]
                ],
                width="stretch",
            )
    st.caption(
        "This section only reads what a previous run recorded. Refreshing the page "
        "never trains a model and never starts an MLflow run."
    )


def render_benchmark() -> None:
    """Section 12: the controlled conventional-vs-autonomous comparison.

    Strictly read-only: the dashboard only reads ``reports/phase11_comparison.json``
    and never triggers a benchmark run, exactly like every other section never
    triggers the pipeline. When the file is missing the section shows the fixed
    "not available" message, so the dashboard does not depend on Phase 11 having
    been executed.
    """
    st.subheader("12. Conventional vs Autonomous")
    section = helpers.build_benchmark_section()

    if not section["available"]:
        st.info(config.DASHBOARD_NO_BENCHMARK)
        return

    st.caption(
        f"Generated {section['generated_at']} · conventional grid: "
        f"{section.get('conventional_grid_size')} configurations · autonomous "
        f"trials: {section.get('autonomous_n_trials')} per model"
    )
    for dataset in section["datasets"]:
        st.markdown(f"**{dataset['dataset']}**")
        st.dataframe(
            [
                {
                    "Workflow": row["workflow"],
                    "Best model": row["model"],
                    "Source": row["selection_source"],
                    "F1": row["f1_score"],
                    "Accuracy": row["accuracy"],
                    "CV F1": row["cv_f1_mean"],
                    "Runtime (s)": row["runtime_seconds"],
                    "Experiments": row["experiments"],
                }
                for row in dataset["rows"]
            ],
            width="stretch",
        )
        st.write(
            f"Δ F1 **{dataset['delta_f1']}** · Δ accuracy **{dataset['delta_accuracy']}** "
            f"· Δ runtime **{dataset['delta_runtime']}** s · "
            f"Δ experiments **{dataset['delta_experiments']}**"
        )
        st.markdown(f"*{dataset['verdict']}*")
        st.caption(" ".join(dataset["notes"]))


def render_security_note() -> None:
    """State the loading policy, because it is a real constraint."""
    with st.expander("Why can I not upload a model file?"):
        st.markdown(config.DASHBOARD_UNTRUSTED_ARTIFACT)
        st.markdown(config.PERSISTENCE_LIMITATION_NO_UPLOAD)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the dashboard.

    The order is fixed and matches the pipeline, so a reader can follow the same
    path the workflow took.
    """
    # First Streamlit command of the run, as the API requires.
    st.set_page_config(
        page_title=config.DASHBOARD_TITLE,
        page_icon="📊",
        layout="wide",
    )
    render_header()
    state = cached_dashboard_state()

    render_system_status(state)
    st.divider()
    render_dataset(state)
    st.divider()
    render_comparison(state)
    st.divider()
    render_evaluation(state)
    st.divider()
    render_confusion_matrix(state)
    st.divider()
    render_optimization(state)
    st.divider()
    render_llm(state)
    st.divider()
    render_prediction(state)
    st.divider()
    render_model_information(state)
    st.divider()
    render_experiment_tracking()
    st.divider()
    render_benchmark()
    st.divider()
    render_security_note()

    st.caption(config.PERSISTENCE_LIMITATION_NO_RETRAIN)


if __name__ == "__main__":
    # Streamlit executes the target file with ``__name__`` set to "__main__", so
    # this is the branch `streamlit run dashboard.py` takes. Guarding on it also
    # means `import dashboard` in a test does not attempt to render anything,
    # which is what lets the render functions be tested one at a time.
    main()
