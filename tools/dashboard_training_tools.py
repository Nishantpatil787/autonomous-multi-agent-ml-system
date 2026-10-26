"""CSV upload validation and workflow launching for the dashboard.

This module is the bridge between the interactive dashboard and the existing
multi-agent pipeline. It deliberately contains **no machine learning logic of its
own**:

* it parses and validates an uploaded CSV;
* it saves the file under a project-managed directory;
* it calls :func:`graph.workflow.run_workflow`, the same entry point ``app.py``
  uses, so data analysis, preprocessing, model selection, training, evaluation,
  Optuna optimization, persistence and MLflow tracking all run exactly once and
  in exactly one implementation;
* on success it publishes the three artifacts to ``models/`` so the dashboard
  can read the new run.

The module is separate from :mod:`tools.dashboard_tools` on purpose. The
dashboard's read layer is asserted to import no agent, no graph and to write no
file; this module does all three, but only when the user explicitly starts a
workflow. Keeping them apart means a page render can never train a model by
accident.

Security notes:

* a model file is never accepted. The only artifact loaded by the dashboard is
  the one this project wrote, through
  :func:`tools.persistence_tools.load_persisted_model`;
* an uploaded CSV is treated as untrusted text: its extension, size, header and
  parseability are checked before it is saved, and it is saved with a generated
  name so it cannot overwrite an existing dataset;
* the workflow is called as a Python function with arguments, never through a
  shell, so no uploaded filename reaches a command line.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from tools import data_tools
from tools.persistence_tools import artifact_paths
from utils import config

#: Hard ceiling for an uploaded file. Streamlit's own ``maxUploadSize`` in
#: .streamlit/config.toml is set to the same number, so the browser refuses the
#: file first and this check refuses it again before any parsing happens.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Where uploaded datasets are kept. ``datasets/*`` is ignored by Git, so a
#: demonstration upload never becomes a tracked file.
UPLOADS_DIR = config.DATASETS_DIR / "uploads"

#: Where a dashboard run writes its artifacts before they are published. Keeping
#: the staging directory means a failed run leaves the previous ``models/``
#: artifact untouched and still leaves its own files for inspection.
DASHBOARD_RUNS_DIR = config.EXPERIMENTS_DIR / "dashboard_runs"

#: Smallest number of rows accepted. The pipeline can split fewer rows, but a
#: model fitted on one or two rows is not a useful demonstration. Five rows
#: leaves a non-empty training and test side of an 80/20 split.
MIN_ROWS = 5

#: Extensions accepted for a dataset. Deliberately a one-item allow-list: the
#: dashboard must never accept a pickle or an archive.
ACCEPTED_DATASET_EXTENSIONS = (".csv",)


class UploadValidationError(Exception):
    """Raised when an upload cannot be used as a training dataset."""


class WorkflowLaunchError(Exception):
    """Raised when the workflow cannot be started or its artifacts published."""


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UploadedDataset:
    """A validated upload, ready to be previewed and configured.

    The frame is held in memory for the preview and the dataset overview. It is
    **not** modified: the pipeline does its own analysis and preprocessing.
    """

    filename: str
    frame: pd.DataFrame
    columns: List[str]
    rows: int
    column_count: int
    missing_values: int
    duplicate_rows: int
    column_dtypes: Dict[str, str]
    warnings: List[str] = field(default_factory=list)

    @property
    def preview(self) -> pd.DataFrame:
        """Return the first five rows, the only rows the UI displays."""
        return self.frame.head(5)


@dataclass(frozen=True)
class FeatureSummary:
    """How the project's own detector groups the columns."""

    numerical: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        """Return the number of usable feature columns."""
        return len(self.numerical) + len(self.categorical)


@dataclass(frozen=True)
class TargetValidation:
    """The outcome of checking a chosen target column before training."""

    ok: bool
    target: str = ""
    message: str = ""
    detail: str = ""
    class_count: int = 0
    min_class_count: int = 0
    cross_validation: bool = True


@dataclass(frozen=True)
class TrainingOutcome:
    """The result of one dashboard-initiated workflow run.

    ``ok`` is True only when the workflow completed, persistence completed and
    the artifacts were published (or written to the caller's output directory).
    A failed run keeps the previous dashboard artifacts intact and reports the
    reason here instead of raising into the Streamlit render.
    """

    ok: bool
    message: str
    detail: str = ""
    workflow_status: str = ""
    run_id: str = ""
    dataset_name: str = ""
    target: str = ""
    output_dir: Optional[Path] = None
    published: bool = False
    stages: List[Dict[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    failed_node: str = ""


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
def _check_filename(filename: str) -> str:
    """Return the bare filename when it has an accepted extension.

    Args:
        filename: The name the browser reported. It is untrusted.

    Returns:
        The final path component, with directories removed.

    Raises:
        UploadValidationError: The name is empty or the extension is not CSV.
    """
    bare = Path(str(filename or "")).name.strip()
    if not bare:
        raise UploadValidationError("The uploaded file has no name.")
    suffix = Path(bare).suffix.lower()
    if suffix not in ACCEPTED_DATASET_EXTENSIONS:
        raise UploadValidationError(
            f"'{suffix or bare}' is not accepted. Upload a CSV file (.csv); "
            "the dashboard never accepts a model file."
        )
    return bare


def _header_columns(payload: bytes) -> List[str]:
    """Return the CSV header names, before pandas renames duplicates.

    pandas silently renames a duplicate column to ``name.1``, which would hide a
    real formatting problem. Reading the first line directly lets the dashboard
    report it instead.

    Args:
        payload: The raw uploaded bytes.

    Returns:
        The header names in file order.

    Raises:
        UploadValidationError: The header is unreadable or empty.
    """
    try:
        first_line = payload.split(b"\n", 1)[0].decode("utf-8-sig").rstrip("\r")
    except UnicodeDecodeError as error:
        raise UploadValidationError(
            "The CSV file is not UTF-8 encoded, so its header cannot be read. "
            "Save it as UTF-8 CSV and upload it again."
        ) from error

    try:
        header = next(csv.reader([first_line]))
    except (csv.Error, StopIteration) as error:
        raise UploadValidationError(
            f"The CSV header could not be read: {error}"
        ) from error

    if not header or all(not str(name).strip() for name in header):
        raise UploadValidationError("The CSV file has no readable header row.")
    return [str(name) for name in header]


def inspect_upload(
    filename: str,
    payload: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> UploadedDataset:
    """Validate and parse an uploaded CSV without writing anything.

    Args:
        filename: The name the browser reported.
        payload: The raw bytes.
        max_bytes: Size ceiling, injectable so a test does not allocate 20 MB.

    Returns:
        An :class:`UploadedDataset` holding the parsed frame and quality counts.

    Raises:
        UploadValidationError: The file is empty, too large, not a CSV or not
            parseable. The message is written for the person who uploaded it.
    """
    bare = _check_filename(filename)

    if not payload:
        raise UploadValidationError("The uploaded file is empty.")
    if len(payload) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise UploadValidationError(
            f"The uploaded file is {len(payload) / (1024 * 1024):.1f} MB, above "
            f"the {limit_mb:.0f} MB limit."
        )

    header = _header_columns(payload)
    duplicates = sorted(
        {name for name in header if header.count(name) > 1}
    )
    if duplicates:
        raise UploadValidationError(
            "The CSV header contains duplicate column name(s): "
            + ", ".join(duplicates)
            + ". Rename them so every column is unique."
        )

    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except pd.errors.EmptyDataError as error:
        raise UploadValidationError("The CSV file is empty.") from error
    except pd.errors.ParserError as error:
        raise UploadValidationError(
            f"The CSV file could not be parsed: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise UploadValidationError(
            "The CSV file is not UTF-8 encoded. Save it as UTF-8 CSV and upload "
            "it again."
        ) from error
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        raise UploadValidationError(
            f"The CSV file could not be read ({type(error).__name__}: {error})."
        ) from error

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise UploadValidationError(
            f"The CSV file has no usable rows or columns (shape {frame.shape})."
        )
    if frame.shape[1] < 2:
        raise UploadValidationError(
            "The dataset needs at least one feature column and one target column; "
            f"this file has {frame.shape[1]} column(s)."
        )
    if frame.shape[0] < MIN_ROWS:
        raise UploadValidationError(
            f"The dataset has {frame.shape[0]} row(s); at least {MIN_ROWS} are "
            "needed for a train/test split."
        )

    missing_values = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    warnings: List[str] = []
    if missing_values:
        warnings.append(
            f"{missing_values} missing value(s) were found. The pipeline's "
            "preprocessing step will impute them; the dashboard does not alter "
            "the file."
        )
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} exact duplicate row(s) were found. The pipeline "
            "removes them in memory only; the uploaded file is unchanged."
        )
    unsupported = data_tools.detect_feature_types(frame).unsupported_columns
    if unsupported:
        warnings.append(
            "Column(s) with unsupported types will be excluded by the pipeline: "
            + ", ".join(unsupported)
        )

    return UploadedDataset(
        filename=bare,
        frame=frame,
        columns=[str(column) for column in frame.columns],
        rows=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        column_dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        warnings=warnings,
    )


def summarize_features(
    frame: pd.DataFrame,
    target: Optional[str] = None,
) -> FeatureSummary:
    """Group the columns exactly as the project's own detector does.

    Args:
        frame: The uploaded frame.
        target: The selected target, excluded from the feature groups.

    Returns:
        A :class:`FeatureSummary`.
    """
    excluded = [target] if target else []
    types = data_tools.detect_feature_types(frame, exclude_columns=excluded)
    return FeatureSummary(
        numerical=list(types.numerical_features),
        categorical=list(types.categorical_features),
        unsupported=list(types.unsupported_columns),
    )


def validate_target(frame: pd.DataFrame, target: str) -> TargetValidation:
    """Check that a chosen target can be used by the existing pipeline.

    The checks mirror the pipeline's own guards, so a problem is reported before
    a long run instead of halfway through it. Nothing is changed.

    Args:
        frame: The uploaded frame.
        target: The column the user selected.

    Returns:
        A :class:`TargetValidation`; ``ok`` is False with a message when the
        column cannot be trained on.
    """
    name = str(target or "").strip()
    if not name:
        return TargetValidation(
            ok=False,
            message="Select the column the model should predict.",
        )
    if name not in [str(column) for column in frame.columns]:
        return TargetValidation(
            ok=False,
            target=name,
            message=(
                f"The selected target column '{name}' is not present in the "
                "dataset. Choose one of the listed columns."
            ),
        )

    series = frame[name]
    if series.isna().any():
        return TargetValidation(
            ok=False,
            target=name,
            message=(
                f"The target column '{name}' contains "
                f"{int(series.isna().sum())} missing value(s). Rows without a "
                "target cannot be used for supervised training."
            ),
        )

    class_count = int(series.nunique(dropna=True))
    if class_count < 2:
        return TargetValidation(
            ok=False,
            target=name,
            message=(
                f"The target column '{name}' has a single distinct value; there "
                "is nothing to learn. Choose a column with at least two classes."
            ),
        )
    if class_count > config.MAX_CLASSES_FOR_CLASSIFICATION:
        return TargetValidation(
            ok=False,
            target=name,
            message=(
                f"The target column '{name}' has {class_count} distinct values. "
                "This project supports classification targets with at most "
                f"{config.MAX_CLASSES_FOR_CLASSIFICATION} classes."
            ),
        )

    counts = series.value_counts(dropna=True)
    min_class_count = int(counts.min()) if not counts.empty else 0
    if min_class_count < config.CV_MIN_FOLDS:
        return TargetValidation(
            ok=False,
            target=name,
            class_count=class_count,
            min_class_count=min_class_count,
            message=(
                f"The smallest class in '{name}' has {min_class_count} sample(s). "
                f"At least {config.CV_MIN_FOLDS} per class are needed for "
                "cross-validation."
            ),
        )

    features = summarize_features(frame, target=name)
    if features.feature_count == 0:
        return TargetValidation(
            ok=False,
            target=name,
            class_count=class_count,
            min_class_count=min_class_count,
            message=(
                "No usable feature column is left after removing the target. "
                "Upload a dataset with at least one numerical or categorical "
                "feature."
            ),
        )

    detail = ""
    if features.unsupported:
        detail = (
            "Unsupported feature column(s) will be excluded by the pipeline: "
            + ", ".join(features.unsupported)
        )

    return TargetValidation(
        ok=True,
        target=name,
        detail=detail,
        class_count=class_count,
        min_class_count=min_class_count,
        cross_validation=min_class_count >= config.CV_FOLDS,
    )


# --------------------------------------------------------------------------- #
# Saving the upload
# --------------------------------------------------------------------------- #
def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe stem derived from an untrusted filename."""
    stem = Path(str(filename or "")).stem
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in stem
    ).strip("_")
    return (cleaned or "dataset")[:60]


def save_upload(
    filename: str,
    payload: bytes,
    *,
    uploads_dir: Optional[Path] = None,
) -> Path:
    """Save an uploaded CSV under a generated, non-overwriting name.

    Args:
        filename: The name the browser reported.
        payload: The validated raw bytes.
        uploads_dir: Destination directory, injectable for tests.

    Returns:
        The path that was written.

    Raises:
        UploadValidationError: The name is not a CSV or the file cannot be
            written. Nothing is overwritten: the file is opened in exclusive
            creation mode.
    """
    bare = _check_filename(filename)
    # Re-validate the content here as well. The view calls ``inspect_upload``
    # first, but this function is also public, and "validated" must not depend on
    # which caller remembered to do it. A rejected file is never written.
    inspect_upload(bare, payload)
    directory = Path(uploads_dir) if uploads_dir is not None else UPLOADS_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise UploadValidationError(
            f"The upload directory could not be created: {error}"
        ) from error

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{_safe_stem(bare)}__{stamp}__{uuid.uuid4().hex[:8]}.csv"
    path = directory / name
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:  # practically impossible, still explicit
        raise UploadValidationError(
            "A file with the generated upload name already exists; nothing was "
            "overwritten. Try the upload again."
        ) from error
    except OSError as error:
        raise UploadValidationError(
            f"The uploaded file could not be saved: {error}"
        ) from error
    return path


# --------------------------------------------------------------------------- #
# Publishing artifacts
# --------------------------------------------------------------------------- #
def publish_artifacts(source_dir: Path, destination_dir: Optional[Path] = None) -> Path:
    """Copy the three artifacts from a staging directory into ``models/``.

    Each file is copied to a temporary name in the destination directory and
    then moved into place with :func:`os.replace`, so a reader never sees a
    half-written file. The previous artifact stays untouched until every source
    file has been verified and copied.

    Args:
        source_dir: Directory the workflow wrote.
        destination_dir: Directory to publish into, or None for ``models/``.

    Returns:
        The destination directory.

    Raises:
        WorkflowLaunchError: One of the three artifacts is missing or a copy
            failed.
    """
    source = artifact_paths(source_dir)
    destination = (
        Path(destination_dir).resolve()
        if destination_dir is not None
        else Path(config.MODELS_DIR).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    targets = artifact_paths(destination)

    sources = [source.model, source.metadata, source.results]
    for path in sources:
        if not path.is_file():
            raise WorkflowLaunchError(
                f"The workflow finished but '{path.name}' was not written, so "
                "the previous dashboard artifact was kept."
            )

    temporary: List[Path] = []
    backups: List[tuple[Path, Path]] = []
    replaced: List[Path] = []
    try:
        # Copy every file first: if any copy fails, the live artifacts are still
        # the previous complete set.
        for path, target in zip(
            sources, [targets.model, targets.metadata, targets.results]
        ):
            temporary_path = target.with_name(
                f".{target.name}.tmp-{uuid.uuid4().hex[:8]}"
            )
            shutil.copy2(path, temporary_path)
            temporary.append(temporary_path)

        # Keep a copy of the live set before touching it. A replacement that
        # fails halfway can then be rolled back instead of leaving one new file
        # beside two old ones, which the dashboard would load as a mismatched
        # artifact set.
        for target in [targets.model, targets.metadata, targets.results]:
            if target.exists():
                backup = target.with_name(f".{target.name}.bak-{uuid.uuid4().hex[:8]}")
                shutil.copy2(target, backup)
                backups.append((backup, target))

        for temporary_path, target in zip(
            temporary, [targets.model, targets.metadata, targets.results]
        ):
            os.replace(temporary_path, target)
            replaced.append(target)
        temporary.clear()
    except OSError as error:
        for leftover in temporary:
            try:
                leftover.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        # Restore every file that was actually replaced. A replaced target that
        # did not exist before is removed, returning the directory to its
        # previous state. Targets the failure never reached are left untouched.
        restored = {target: backup for backup, target in backups}
        for target in replaced:
            if target in restored:
                try:
                    os.replace(restored[target], target)
                except OSError:  # pragma: no cover - best-effort rollback
                    pass
            else:
                try:
                    target.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best-effort rollback
                    pass
        raise WorkflowLaunchError(
            f"The new artifacts could not be published to '{destination.name}': "
            f"{error}. The previous artifact was restored."
        ) from error
    finally:
        # A successful publish no longer needs its backups; remove them either
        # way. A failure to remove a dot-file is not a reason to fail the run.
        for backup, _ in backups:
            try:
                backup.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass

    return destination


# --------------------------------------------------------------------------- #
# Running the workflow
# --------------------------------------------------------------------------- #
def _relative(path: Optional[Path]) -> str:
    """Return a path relative to the project root, or its name."""
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(config.PROJECT_ROOT)).replace(
            "\\", "/"
        )
    except ValueError:
        return Path(path).name


def _stage_records(result: Any) -> List[Dict[str, str]]:
    """Turn the workflow's node records into simple display rows."""
    rows: List[Dict[str, str]] = []
    for record in getattr(result, "node_records", []) or []:
        status = getattr(record, "status", "")
        rows.append(
            {
                "stage": str(getattr(record, "display_name", "") or getattr(record, "node", "")),
                "status": str(getattr(status, "value", status)),
            }
        )
    return rows


def run_training_pipeline(
    *,
    dataset_path: str,
    target_column: str,
    enable_optimization: bool = True,
    n_trials: Optional[int] = None,
    enable_tracking: bool = True,
    enable_llm: bool = False,
    output_dir: Optional[Path] = None,
    staging_root: Optional[Path] = None,
) -> TrainingOutcome:
    """Run the existing workflow on a chosen dataset and target.

    The browser-facing call uses ``output_dir=None``. The run writes to a
    staging directory under ``experiments/dashboard_runs/`` and, only when the
    workflow and persistence both complete, publishes the three artifacts to
    ``models/``. A failed run therefore leaves the previous dashboard results
    intact instead of replacing them with a half-finished one.

    Args:
        dataset_path: CSV to train on. The dashboard passes the saved upload.
        target_column: Column the model should predict.
        enable_optimization: Whether the Optuna node may run.
        n_trials: Trials per model, or None for the configured default.
        enable_tracking: Whether MLflow may record the run.
        enable_llm: Whether the optional local interpretation layer may run.
        output_dir: Write directly here instead of staging and publishing. Used
            by tests and by any caller that manages its own destination.
        staging_root: Parent directory for staged runs, injectable for tests.

    Returns:
        A :class:`TrainingOutcome`. Never raises for a workflow failure; the
        failure is reported in the returned record so the page stays usable.
    """
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    target = str(target_column or "").strip()
    path = Path(dataset_path)
    run_id = f"{started.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def finish(
        *,
        ok: bool,
        message: str,
        detail: str = "",
        workflow_status: str = "",
        published: bool = False,
        output: Optional[Path] = None,
        stages: Optional[List[Dict[str, str]]] = None,
        failed_node: str = "",
    ) -> TrainingOutcome:
        """Build the outcome with a consistent timestamp and duration."""
        ended = datetime.now(timezone.utc)
        return TrainingOutcome(
            ok=ok,
            message=message,
            detail=detail,
            workflow_status=workflow_status,
            run_id=run_id,
            dataset_name=path.name,
            target=target,
            output_dir=output,
            published=published,
            stages=list(stages or []),
            duration_seconds=round(time.perf_counter() - clock, 2),
            started_at=started.isoformat(timespec="seconds"),
            finished_at=ended.isoformat(timespec="seconds"),
            failed_node=failed_node,
        )

    # Defensive re-validation. The view has already checked the upload, but this
    # function is also callable directly, and a long run should not start on a
    # path or target that cannot work.
    if not path.is_file():
        return finish(
            ok=False,
            message="The dataset file is no longer available. Upload it again.",
        )
    try:
        frame = data_tools.load_csv(path)
    except Exception as error:  # noqa: BLE001 - reported to the caller
        return finish(
            ok=False,
            message="The uploaded dataset could not be read.",
            detail=f"{type(error).__name__}: {error}",
        )

    validation = validate_target(frame, target)
    if not validation.ok:
        return finish(
            ok=False,
            message=validation.message,
            detail=validation.detail,
        )

    trials = config.OPTUNA_N_TRIALS if n_trials is None else int(n_trials)
    if not 1 <= trials <= 200:
        return finish(
            ok=False,
            message="The number of optimization trials must be between 1 and 200.",
        )

    direct_output = Path(output_dir).resolve() if output_dir is not None else None
    if direct_output is not None:
        staging = direct_output
    else:
        parent = Path(staging_root) if staging_root is not None else DASHBOARD_RUNS_DIR
        staging = (parent / run_id).resolve()
    try:
        staging.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return finish(
            ok=False,
            message="The run directory could not be created.",
            detail=f"{type(error).__name__}: {error}",
        )

    try:
        # Imported here, not at module import: the dashboard must start even if
        # the optional graph dependency is missing, and importing the graph on a
        # simple page view would contradict the read-only design.
        from graph.workflow import run_workflow
    except Exception as error:  # noqa: BLE001 - optional dependency boundary
        return finish(
            ok=False,
            message=(
                "The multi-agent workflow could not be imported. Install the "
                "project requirements (langgraph and optuna) and try again."
            ),
            detail=f"{type(error).__name__}: {error}",
        )

    try:
        result = run_workflow(
            dataset_path=str(path),
            target_column=target,
            enable_optimization=bool(enable_optimization),
            cross_validation=bool(validation.cross_validation),
            n_trials=trials,
            enable_persistence=True,
            persistence_output_dir=str(staging),
            enable_tracking=bool(enable_tracking),
            enable_llm=bool(enable_llm),
        )
    except Exception as error:  # noqa: BLE001 - shown in the page, never hidden
        return finish(
            ok=False,
            message="The workflow stopped with an unexpected error.",
            detail=f"{type(error).__name__}: {error}",
        )

    workflow_status = str(getattr(result.workflow_status, "value", result.workflow_status))
    stages = _stage_records(result)
    final_state = getattr(result, "final_agent_state", None)
    persistence_status = str(
        getattr(final_state, "persistence_status", "") if final_state else ""
    )
    failed_node = str(getattr(result, "failed_node", "") or "")
    last_error = getattr(result, "last_error", None)
    error_detail = ""
    if last_error is not None:
        error_detail = (
            f"{getattr(last_error, 'error_type', '')}: "
            f"{getattr(last_error, 'message', '')}"
        )
        failed_node = failed_node or str(getattr(last_error, "node", ""))

    if workflow_status != "completed":
        return finish(
            ok=False,
            message=(
                f"The workflow did not complete (status: {workflow_status or 'unknown'}). "
                "The previous dashboard results were kept."
            ),
            detail=error_detail,
            workflow_status=workflow_status,
            stages=stages,
            failed_node=failed_node,
        )

    if persistence_status != "completed":
        return finish(
            ok=False,
            message=(
                "The workflow completed but the model was not saved "
                f"(persistence status: {persistence_status or 'unknown'}). The "
                "previous dashboard results were kept."
            ),
            workflow_status=workflow_status,
            stages=stages,
            failed_node=failed_node,
        )

    if direct_output is not None:
        return finish(
            ok=True,
            message="Workflow completed and the artifacts were written.",
            workflow_status=workflow_status,
            published=True,
            output=direct_output,
            stages=stages,
        )

    try:
        published_dir = publish_artifacts(staging, None)
    except WorkflowLaunchError as error:
        return finish(
            ok=False,
            message=str(error),
            detail=f"The staged run was kept under {_relative(staging)}.",
            workflow_status=workflow_status,
            stages=stages,
            failed_node=failed_node,
        )

    return finish(
        ok=True,
        message="Workflow completed. The dashboard now shows the new run.",
        workflow_status=workflow_status,
        published=True,
        output=published_dir,
        stages=stages,
    )
