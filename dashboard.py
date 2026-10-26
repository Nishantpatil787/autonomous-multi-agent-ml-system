"""Streamlit dashboard for the Autonomous ML System.

Run it with::

    streamlit run dashboard.py

The dashboard has four sections:

* **Dashboard** - the summary, selected model and model comparison of the last
  successful run, read from ``models/``;
* **Train a Model** - an explicit CSV upload, target choice and Start button that
  invokes the *existing* LangGraph workflow;
* **Predictions** - one prediction through the persisted pipeline;
* **Experiment History** - the MLflow runs and the detailed evaluation,
  optimization and benchmark records.

What this file is: presentation. Every number it shows was produced by a workflow
run that wrote ``models/``. It does not import an agent or a graph module, so
merely loading the page cannot train, tune or re-evaluate a model. The one place
that starts work is the Start ML Workflow button, which delegates to
:mod:`tools.dashboard_training_tools` and from there to ``graph.workflow.run_workflow`` -
the same entry point ``app.py`` uses. There is no second training implementation
here.

What this file is not: a second machine learning pipeline. A prediction calls
``pipeline.predict`` on the artifact that was saved, and nothing else - the
imputation, encoding and scaling all live inside that saved pipeline.

Deliberately absent: **no model-file upload.** A ``.joblib``/``.pkl`` file is a
pickle; loading one executes code inside the Streamlit process. The only model
the dashboard loads is the artifact this project wrote at the configured path.
The CSV uploader in the Train section is a different thing: it accepts a dataset
to train on, never a model, and the uploaded file is validated and saved under
``datasets/uploads/`` before the pipeline reads it.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from tools import dashboard_tools as helpers
from tools import dashboard_training_tools as training
from utils import config


# --------------------------------------------------------------------------- #
# Navigation and presentation constants
# --------------------------------------------------------------------------- #
NAV_DASHBOARD = "Dashboard"
NAV_TRAIN = "Train a Model"
NAV_PREDICT = "Predictions"
NAV_HISTORY = "Experiment History"
NAV_OPTIONS = (NAV_DASHBOARD, NAV_TRAIN, NAV_PREDICT, NAV_HISTORY)

#: Small, dependency-free styling on top of the light theme in
#: .streamlit/config.toml. The theme file does the palette; this only tightens
#: spacing and makes the native metric cards read as cards.
GLOBAL_CSS = """
<style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
    }
    /* This is a local dashboard, not a Streamlit Cloud app. */
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    [data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }
    h1 {
        letter-spacing: -0.02em;
    }
    .ams-eyebrow {
        color: #2563EB;
        font-weight: 700;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: -0.4rem;
    }
    .ams-muted {
        color: #64748B;
    }
</style>
"""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_dashboard_state() -> helpers.DashboardState:
    """Load the artifacts once per session, not on every rerun.

    Returns:
        The loaded :class:`~tools.dashboard_tools.DashboardState`.
    """
    return helpers.load_dashboard_state()


@st.cache_data(show_spinner=False, max_entries=4)
def cached_inspect(filename: str, payload: bytes) -> training.UploadedDataset:
    """Parse an uploaded CSV once per distinct file.

    Args:
        filename: The name the browser reported.
        payload: The raw file bytes.

    Returns:
        The validated :class:`~tools.dashboard_training_tools.UploadedDataset`.
    """
    return training.inspect_upload(filename, payload)


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


def inject_styles() -> None:
    """Apply the small amount of CSS the native theme does not cover."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header and navigation
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """Show the title, the tagline and the pipeline the page reports on."""
    st.markdown('<p class="ams-eyebrow">Multi-agent machine learning</p>', unsafe_allow_html=True)
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_TAGLINE)
    st.caption(
        "Upload a CSV dataset to train a new model, or explore the last successful "
        "run. The dashboard never changes a result unless you explicitly start a "
        "workflow."
    )
    st.caption(config.DASHBOARD_SUBTITLE)


def _apply_pending_navigation() -> None:
    """Apply a navigation request before the radio widget is created.

    Streamlit does not allow changing a widget's session value after the widget
    has been instantiated in the same run, so a button stores its destination in
    ``_pending_nav`` and this function applies it at the start of the next run.
    """
    pending = st.session_state.get("_pending_nav")
    if pending in NAV_OPTIONS:
        st.session_state["nav_section"] = pending
        del st.session_state["_pending_nav"]
    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = NAV_DASHBOARD


def render_navigation() -> str:
    """Render the sidebar and return the selected section.

    Returns:
        One of :data:`NAV_OPTIONS`.
    """
    with st.sidebar:
        st.markdown("## Autonomous ML")
        st.caption("Multi-agent machine learning")
        st.divider()
        section = st.radio(
            "Navigation",
            NAV_OPTIONS,
            key="nav_section",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("CSV datasets only. Model files are never accepted.")
        st.caption("Results are read from `models/`.")
        st.caption("Built with Python, scikit-learn, LangGraph and MLflow.")
    return str(section)


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #
def _humanise(value: Any) -> str:
    """Turn a recorded enum-like value into a readable label.

    Args:
        value: A value such as ``"binary_classification"``.

    Returns:
        A capitalised label, or ``"not recorded"`` when there is nothing to show.
    """
    text = str(value or "").replace("_", " ").strip()
    if not text or text == helpers.NOT_RECORDED:
        return "not recorded"
    return text[:1].upper() + text[1:]


def _dataset_name(state: helpers.DashboardState) -> str:
    """Return the recorded dataset name, or a clear fallback."""
    name = state.dataset.get("name") or state.metadata.get("dataset_name")
    return str(name) if name else "not recorded"


def _feature_count(state: helpers.DashboardState) -> str:
    """Return the number of feature columns the run recorded."""
    dataset = state.dataset
    numeric = dataset.get("numerical_feature_count")
    categorical = dataset.get("categorical_feature_count")
    if isinstance(numeric, int) and isinstance(categorical, int):
        return str(numeric + categorical)
    features = state.feature_columns()
    return str(len(features)) if features else "not recorded"


def _selected_metrics(state: helpers.DashboardState) -> Dict[str, Any]:
    """Return the metrics recorded for the selected model."""
    return dict(state.final_model.get("metrics") or {})


def _optimization_status(state: helpers.DashboardState) -> str:
    """Return the optimization outcome recorded for the selected model."""
    final = state.final_model
    return str(
        final.get("optimization_status")
        or state.optimization.get("overall_status")
        or "not run"
    )


def render_summary_cards(state: helpers.DashboardState) -> None:
    """Show the seven at-a-glance values for the last successful run.

    Every value is read from the artifacts. Nothing is computed from the CSV and
    nothing is filled in with a sample number: a missing value is shown as
    ``not recorded``.
    """
    dataset = state.dataset
    rows = dataset.get("rows")
    metrics = _selected_metrics(state)

    first = st.columns(4)
    first[0].metric("Dataset", _dataset_name(state), border=True)
    first[1].metric(
        "Rows",
        f"{int(rows):,}" if isinstance(rows, (int, float)) else "not recorded",
        border=True,
    )
    first[2].metric("Features", _feature_count(state), border=True)
    first[3].metric("Task", _humanise(state.task_type), border=True)

    second = st.columns(3)
    second[0].metric("Selected model", state.model_name, border=True)
    second[1].metric(
        "Hold-out F1",
        helpers.format_metric(metrics.get("f1_score")),
        border=True,
        help="The primary selection metric, measured on the hold-out split.",
    )
    second[2].metric("Optimization", _humanise(_optimization_status(state)), border=True)


def render_selected_model(state: helpers.DashboardState) -> None:
    """Show the persisted model, its provenance and its two recorded scores."""
    final = state.final_model
    metrics = _selected_metrics(state)
    source = str(final.get("selection_source") or "baseline")
    source_label = (
        "Optimized after hyperparameter search (Phase 6)"
        if source == "optimized"
        else "Baseline model (Phase 4)"
    )

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"#### {state.model_name}")
            st.caption(
                f"Target '{state.target_column}' | {source_label} | "
                "saved in models/"
            )
            st.caption(
                "Baseline hold-out F1: "
                f"{helpers.format_metric(final.get('baseline_f1_score'))}"
            )
        with right:
            st.metric(
                "Selection metric",
                helpers.format_metric(metrics.get("f1_score")),
                help="Primary hold-out F1, the project's selection metric.",
            )
            st.caption(
                "Accuracy " + helpers.format_metric(metrics.get("accuracy"))
            )
    st.caption(
        "The selection rule is the project's existing ranking: primary F1 "
        "(descending), secondary accuracy (descending), then model name. The "
        "artifact is persisted and is the one Predictions uses."
    )


def _comparison_rows(section: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a compact comparison table from the helper's rows.

    Baselines and optimized models are kept apart by a ``Kind`` column, and the
    persisted model is marked instead of being moved to the top or hidden.

    Args:
        section: The result of ``build_comparison_section``.

    Returns:
        Rows ready for ``st.dataframe``.
    """
    rows = list(section.get("rows") or [])
    metric_names = ("Accuracy", "Precision", "Recall", "F1", "CV F1")
    available = [
        metric
        for metric in metric_names
        if any(str(row.get(metric, helpers.NOT_RECORDED)) != helpers.NOT_RECORDED for row in rows)
    ]
    table: List[Dict[str, str]] = []
    for row in rows:
        item = {
            "Model": str(row.get("Model", "")),
            "Kind": str(row.get("Kind", "")).title(),
            "Selected": "yes" if row.get("Persisted") == "yes" else "",
        }
        for metric in available:
            item[metric] = str(row.get(metric, helpers.NOT_RECORDED))
        table.append(item)
    return table


# --------------------------------------------------------------------------- #
# Section: dashboard home
# --------------------------------------------------------------------------- #
def render_comparison(state: helpers.DashboardState) -> None:
    """Show the baseline and optimized models side by side.

    The chart comes first, so the shape of the comparison is visible before the
    exact numbers. The table keeps every row; the persisted model is marked with
    a ``Selected`` column rather than being isolated.

    Args:
        state: The loaded dashboard state.
    """
    section = helpers.build_comparison_section(state)
    if not section.get("available"):
        st.info("No model comparison was stored with this artifact.")
        return

    chart = [
        row
        for row in (section.get("f1_chart") or [])
        if isinstance(row.get("F1"), (int, float))
    ]
    if chart:
        st.markdown("**Hold-out F1 by model**")
        # No ``key``: st.bar_chart does not accept one. The element identity is
        # derived from the chart's contents.
        st.bar_chart(
            chart,
            x="Model",
            y="F1",
            color="Kind",
            x_label="Model",
            y_label="Hold-out F1",
        )
        st.caption(
            "A model that ran as both a baseline and a tuned model has two bars, "
            "one per kind. The two phases are different measurements and are not "
            "interchangeable."
        )
    else:
        st.info("No numeric hold-out F1 score was recorded, so there is no chart to draw.")

    table = _comparison_rows(section)
    if table:
        st.dataframe(table, width="stretch", hide_index=True, key="comparison_table")
    st.caption(section.get("persisted_note") or "")
    st.caption(
        f"Persisted model: {section.get('persisted_label')} | "
        f"{section.get('baseline_count')} baseline model(s) and "
        f"{section.get('optimized_count')} tuned model(s)."
    )


def render_system_status(state: helpers.DashboardState) -> None:
    """Show whether a model is available and which one."""
    if not state.available:
        _render_no_artifact(state)
        return

    rows = helpers.build_status_section(state)
    show_rows(rows, key="status_rows")
    if state.results_available:
        st.caption(
            "Model, dataset, evaluation and optimization details all come from the "
            "artifacts of the last successful run."
        )
    else:
        st.warning(
            f"{config.DASHBOARD_INVALID_ARTIFACT} The saved model loaded, but the "
            "report bundle could not be read, so only the model sections are shown."
        )


def _render_last_attempt_banner() -> None:
    """Warn that the last attempted run failed, so the results are older."""
    outcome = st.session_state.get("last_training")
    if outcome is not None and not outcome.ok:
        st.warning(config.DASHBOARD_FAILED_RUN_NOTE)
        st.caption(f"Last attempt: {outcome.message}")


def _render_no_artifact(state: helpers.DashboardState) -> None:
    """Show the reason there is nothing to display, and a way forward.

    Two distinct situations produce this screen, and they need different actions,
    so they are never collapsed into one message.

    Args:
        state: The loaded dashboard state, whose ``problem`` distinguishes them.
    """
    st.warning(config.DASHBOARD_NO_ARTIFACT)
    if state.problem:
        st.error(state.problem)
    st.markdown(f"**What to do:** {config.DASHBOARD_NO_ARTIFACT_GUIDANCE}")
    st.caption(
        "The fastest path is the button below: upload a CSV, choose the target "
        "column and start the existing workflow. The command-line route is "
        "`python app.py --graph`."
    )
    if st.button("Train a model", type="primary", key="empty_state_train"):
        st.session_state["_pending_nav"] = NAV_TRAIN
        st.rerun()


# --------------------------------------------------------------------------- #
# Section: train a model
# --------------------------------------------------------------------------- #
def _display_path(path: Path) -> str:
    """Return a project-relative path for display, never an absolute one."""
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def _render_last_training_outcome() -> None:
    """Show the result of the most recent run started from this session."""
    outcome = st.session_state.get("last_training")
    if outcome is None:
        return

    if outcome.ok:
        st.success(outcome.message)
    else:
        st.error(outcome.message)
        if outcome.detail:
            st.caption(outcome.detail)
        if outcome.failed_node:
            st.caption(f"Failed stage: {outcome.failed_node}")
        st.info(config.DASHBOARD_FAILED_RUN_NOTE)

    columns = st.columns(4)
    columns[0].metric("Workflow status", outcome.workflow_status or "unknown", border=True)
    columns[1].metric("Duration", f"{outcome.duration_seconds:.1f} s", border=True)
    columns[2].metric("Dataset", outcome.dataset_name or "-", border=True)
    columns[3].metric("Target", outcome.target or "-", border=True)
    if outcome.stages:
        with st.expander(f"Stages ({len(outcome.stages)})"):
            st.dataframe(outcome.stages, width="stretch", hide_index=True)
    st.caption(
        f"Started {outcome.started_at}. Uploaded datasets are kept under "
        "datasets/uploads/ so the run can be reproduced; they are ignored by Git."
    )


def render_train_page(state: helpers.DashboardState) -> None:
    """Upload a dataset, choose a target and start the existing workflow.

    The page validates the upload and the target before saving anything, then
    hands the saved path to :func:`tools.dashboard_training_tools.run_training_pipeline`.
    The run is synchronous, so the browser cannot submit a second copy while it
    is in progress.
    """
    del state  # The training page reads the upload, not the previous artifact.
    st.subheader("Train a Model")
    st.write(
        "Upload a CSV, choose the column the model should predict, and start the "
        "multi-agent workflow."
    )
    st.caption(config.DASHBOARD_TRAIN_NOTE)
    _render_last_training_outcome()

    upload = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        accept_multiple_files=False,
        key="dataset_upload",
        max_upload_size=training.MAX_UPLOAD_BYTES // (1024 * 1024),
        help=(
            "One .csv file, at most 20 MB. The file is previewed first; nothing is "
            "trained until you press Start ML Workflow."
        ),
    )
    if upload is None:
        st.info(
            "Choose a CSV file to see its shape, a five-row preview and the "
            "available columns. The current dashboard results stay unchanged "
            "until a new run succeeds."
        )
        return

    payload = upload.getvalue()
    try:
        dataset = cached_inspect(upload.name, payload)
    except training.UploadValidationError as error:
        st.error(str(error))
        return

    st.success(
        f"**{dataset.filename}** | {dataset.rows:,} rows | "
        f"{dataset.column_count} columns"
    )
    with st.expander("Preview the first five rows", expanded=True):
        st.dataframe(dataset.preview, width="stretch", hide_index=True)
    with st.expander("Columns and detected types"):
        st.dataframe(
            [
                {
                    "Column": column,
                    "Type": dataset.column_dtypes.get(column, ""),
                }
                for column in dataset.columns
            ],
            width="stretch",
            hide_index=True,
        )
    st.caption(config.DASHBOARD_UPLOAD_NOTE)
    for warning in dataset.warnings:
        st.warning(warning)

    st.markdown("### 1. Choose the target column")
    fingerprint = hashlib.sha256(payload).hexdigest()[:12]
    target = st.selectbox(
        "Target column",
        options=dataset.columns,
        index=None,
        placeholder="Select the column the model should predict",
        key=f"target_column_{fingerprint}",
        help="The target is the label to predict. It is excluded from the features.",
    )

    features = training.summarize_features(dataset.frame, target)
    validation = training.validate_target(dataset.frame, target) if target else None
    if validation is not None and not validation.ok:
        st.error(validation.message)
        return

    st.markdown("### 2. Review the dataset")
    first = st.columns(4)
    first[0].metric("Rows", f"{dataset.rows:,}", border=True)
    first[1].metric("Columns", dataset.column_count, border=True)
    first[2].metric("Missing values", f"{dataset.missing_values:,}", border=True)
    first[3].metric("Duplicate rows", f"{dataset.duplicate_rows:,}", border=True)

    if validation is not None:
        second = st.columns(4)
        second[0].metric("Features", features.feature_count, border=True)
        second[1].metric("Numerical", len(features.numerical), border=True)
        second[2].metric("Categorical", len(features.categorical), border=True)
        second[3].metric("Classes", validation.class_count, border=True)
        st.caption(
            f"Target '{target}' has {validation.class_count} classes; the "
            f"smallest has {validation.min_class_count} row(s)."
        )
        if validation.detail:
            st.warning(validation.detail)
        if not validation.cross_validation:
            st.info(
                "Cross-validation will be skipped because the smallest class has "
                "fewer than five rows. The workflow still trains and evaluates on "
                "a hold-out split."
            )
    if features.unsupported:
        st.caption(
            "Unsupported columns (excluded by the pipeline): "
            + ", ".join(features.unsupported)
        )

    if validation is None:
        st.info(
            "Select a target column to see the feature summary and enable the "
            "workflow. The target is excluded from the features."
        )
        return

    with st.expander("Advanced settings"):
        enable_optimization = st.checkbox(
            "Run hyperparameter optimization (Optuna)",
            value=True,
            key="train_optimization",
            help=(
                "Searches tuned hyperparameters for each candidate model. It is "
                "the slowest part of the run. Turn it off for a quick test."
            ),
        )
        n_trials = st.number_input(
            "Trials per model",
            min_value=1,
            max_value=200,
            value=int(config.OPTUNA_N_TRIALS),
            step=1,
            key="train_trials",
            disabled=not enable_optimization,
        )
        enable_llm = st.checkbox(
            "Request local LLM interpretation (optional, requires Ollama)",
            value=False,
            key="train_llm",
            help=(
                "Off by default. When Ollama is unavailable the workflow still "
                "completes without generated text."
            ),
        )

    st.markdown("### 3. Start the workflow")
    st.caption(
        "The run uses the existing agents and the existing LangGraph workflow. It "
        "can take a few minutes, especially with optimization enabled. If it "
        "fails, the previous dashboard results are kept."
    )
    with st.form("start_workflow_form", clear_on_submit=False):
        submitted = st.form_submit_button(
            "Start ML Workflow",
            type="primary",
            disabled=bool(st.session_state.get("workflow_running", False)),
        )

    if not submitted:
        return
    if validation is None or not target:
        st.error("Select the target column before starting the workflow.")
        return
    if not validation.ok:
        st.error(validation.message)
        return

    st.session_state["workflow_running"] = True
    try:
        try:
            saved_path = training.save_upload(dataset.filename, payload)
        except training.UploadValidationError as error:
            st.error(str(error))
            return

        with st.status("Running the ML workflow...", expanded=True) as status:
            status.write(f"Saved the upload to `{_display_path(saved_path)}`.")
            stages = (
                "Data analysis -> preprocessing -> model selection -> training "
                "-> evaluation"
            )
            if enable_optimization:
                stages += " -> optimization"
            status.write(stages + " -> model persistence.")
            status.write(
                "This can take a few minutes. The page refreshes when the run "
                "finishes."
            )
            outcome = training.run_training_pipeline(
                dataset_path=str(saved_path),
                target_column=str(target),
                enable_optimization=bool(enable_optimization),
                n_trials=int(n_trials) if enable_optimization else None,
                enable_tracking=True,
                enable_llm=bool(enable_llm),
            )
            if outcome.ok:
                status.update(
                    label="Workflow completed",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label="Workflow did not complete",
                    state="error",
                    expanded=True,
                )

        st.session_state["last_training"] = outcome
        if outcome.ok:
            st.cache_data.clear()
        st.rerun()
    finally:
        st.session_state["workflow_running"] = False


# --------------------------------------------------------------------------- #
# Section: predictions
# --------------------------------------------------------------------------- #
def _render_saved_model_summary(state: helpers.DashboardState) -> None:
    """Show which saved model the prediction form will use."""
    final = state.final_model
    metrics = _selected_metrics(state)
    columns = st.columns(4)
    columns[0].metric("Model", state.model_name, border=True)
    columns[1].metric("Target", state.target_column, border=True)
    columns[2].metric(
        "Hold-out F1",
        helpers.format_metric(metrics.get("f1_score")),
        border=True,
    )
    columns[3].metric(
        "Source",
        _humanise(final.get("selection_source") or "baseline"),
        border=True,
    )


def render_prediction(state: helpers.DashboardState) -> None:
    """Predict one row using the saved pipeline.

    The form is built from the persisted metadata and the prediction is a single
    ``predict`` call on the artifact. Nothing here fits, transforms or encodes
    anything.

    Args:
        state: The loaded dashboard state.
    """
    st.markdown("### Make a prediction")
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


# --------------------------------------------------------------------------- #
# Section: detailed history
# --------------------------------------------------------------------------- #
def render_dataset(state: helpers.DashboardState) -> None:
    """Show the recorded dataset facts, without opening the CSV."""
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


def render_evaluation(state: helpers.DashboardState) -> None:
    """Show the recorded evaluation of the selected model."""
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
            "These numbers were measured by Phase 6 after the hyperparameter "
            "search, on the same hold-out split. Precision, recall and a confusion "
            "matrix were not recorded for a tuned model, so they are shown as '-' "
            "rather than filled in with the untuned model's values."
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
        "These numbers were recorded during the workflow run. This section reads "
        "them and never starts a search itself."
    )


def render_llm(state: helpers.DashboardState) -> None:
    """Show the stored local-model interpretations, if the run produced any."""
    section = helpers.build_llm_section(state)
    if not section.get("available"):
        st.info(section["note"])
        st.caption(section["statement"])
        return

    for entry in section["entries"]:
        with st.expander(
            f"{entry['role_display_name']} - {entry['status']}", expanded=True
        ):
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


def render_model_information(state: helpers.DashboardState) -> None:
    """Show the artifact facts."""
    rows = helpers.build_model_information_section(state)
    if not rows:
        st.info(config.DASHBOARD_NO_ARTIFACT)
        return
    show_rows(rows, key="model_information")


@st.cache_data(show_spinner=False)
def cached_tracking_summary() -> Optional[Dict[str, Any]]:
    """Read the MLflow summary once per session.

    Returns:
        The summary mapping, or None when MLflow is not installed, not reachable
        or the store could not be read.
    """
    try:
        from tools import mlflow_tools  # noqa: PLC0415 - optional dependency

        summary = mlflow_tools.experiment_summary()
    except Exception:  # noqa: BLE001 - an optional section must never break the page
        return None

    # ``experiment_summary`` reports an empty run list both for a store with no
    # runs and for a store it could not read. Those are different situations, and
    # claiming "no run has been recorded" for an out-of-date database would be
    # wrong. Tracking is enabled and installed here, so say exactly what is known
    # and what to check.
    if (
        isinstance(summary, dict)
        and not summary.get("available")
        and summary.get("enabled")
        and summary.get("installed")
        and not summary.get("runs")
    ):
        summary["note"] = (
            "MLflow is enabled, but no readable run could be loaded. The store may "
            "be empty, or its schema may need upgrading (run "
            "'mlflow db upgrade <tracking-uri>'). Experiment tracking is optional: "
            "every other section is read from the saved artifacts and is unaffected."
        )
    return summary


def render_experiment_tracking() -> None:
    """Show what MLflow recorded, or that it recorded nothing."""
    st.subheader("Experiment tracking")
    section = helpers.build_experiment_section(cached_tracking_summary())

    if not section["available"]:
        st.info(section["note"] or config.DASHBOARD_NO_EXPERIMENT_TRACKING)
        return

    latest = section["latest"]
    st.write(
        f"Experiment **{section['experiment_name']}** | latest run "
        f"`{latest['short_run_id']}` | MLflow status **{latest['status']}** | "
        f"workflow status **{latest['workflow_status']}** | {latest['start_time']}"
    )
    st.caption(f"Tracking store: {section['tracking_uri']}")

    columns = st.columns(4)
    columns[0].metric("Selected model", latest["model"], border=True)
    columns[1].metric("Hold-out F1", latest["f1_score"], border=True)
    columns[2].metric("Hold-out accuracy", latest["accuracy"], border=True)
    columns[3].metric("Optimization", latest["optimization_status"], border=True)

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
                hide_index=True,
            )
    st.caption(
        "This section only reads what a previous run recorded. Refreshing the page "
        "never trains a model and never starts an MLflow run."
    )


def render_benchmark() -> None:
    """Show the controlled conventional-vs-autonomous comparison.

    Strictly read-only: the dashboard only reads
    ``reports/phase11_comparison.json`` and never triggers a benchmark run.
    """
    st.subheader("Conventional vs Autonomous")
    section = helpers.build_benchmark_section()

    if not section["available"]:
        st.info(config.DASHBOARD_NO_BENCHMARK)
        return

    st.caption(
        f"Generated {section['generated_at']} | conventional grid: "
        f"{section.get('conventional_grid_size')} configurations | autonomous "
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
            hide_index=True,
        )
        st.write(
            f"F1 change **{dataset['delta_f1']}** | accuracy change "
            f"**{dataset['delta_accuracy']}** | runtime change "
            f"**{dataset['delta_runtime']}** s | experiment change "
            f"**{dataset['delta_experiments']}**"
        )
        st.markdown(f"*{dataset['verdict']}*")
        st.caption(" ".join(dataset["notes"]))


def render_security_note() -> None:
    """State the loading policy, because it is a real constraint."""
    with st.expander("Can I upload a model file?"):
        st.markdown(config.DASHBOARD_UNTRUSTED_ARTIFACT)
        st.markdown(config.PERSISTENCE_LIMITATION_NO_UPLOAD)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the dashboard.

    The four sections are chosen in the sidebar. Only the selected section runs,
    so opening the page never parses an upload, never loads a model for training
    and never starts a workflow.
    """
    # First Streamlit command of the run, as the API requires.
    st.set_page_config(
        page_title=f"{config.DASHBOARD_TITLE} | Multi-Agent ML",
        page_icon="\U0001F4CA",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    _apply_pending_navigation()
    state = cached_dashboard_state()
    section = render_navigation()
    render_header()

    if section == NAV_TRAIN:
        render_train_page(state)
    elif section == NAV_PREDICT:
        st.subheader("Predictions")
        if state.available:
            _render_saved_model_summary(state)
        render_prediction(state)
        with st.expander("Model information"):
            render_model_information(state)
        render_security_note()
    elif section == NAV_HISTORY:
        render_experiment_tracking()
        with st.expander("System status"):
            render_system_status(state)
        with st.expander("Dataset overview"):
            render_dataset(state)
        with st.expander("Evaluation metrics"):
            render_evaluation(state)
        with st.expander("Confusion matrix"):
            render_confusion_matrix(state)
        with st.expander("Optimization results"):
            render_optimization(state)
        with st.expander("LLM interpretation"):
            render_llm(state)
        with st.expander("Model information"):
            render_model_information(state)
        st.divider()
        render_benchmark()
    else:
        _render_last_attempt_banner()
        if not state.available:
            _render_no_artifact(state)
        else:
            if not state.results_available:
                st.warning(config.DASHBOARD_INVALID_ARTIFACT)
            render_summary_cards(state)
            st.markdown("### Selected model")
            render_selected_model(state)
            st.markdown("### Model comparison")
            render_comparison(state)
            with st.expander("System status"):
                render_system_status(state)

    st.caption(config.PERSISTENCE_LIMITATION_NO_RETRAIN)


if __name__ == "__main__":
    # Streamlit executes the target file with ``__name__`` set to "__main__", so
    # this is the branch `streamlit run dashboard.py` takes. Guarding on it also
    # means `import dashboard` in a test does not attempt to render anything,
    # which is what lets the render functions be tested one at a time.
    main()
