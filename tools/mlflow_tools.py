"""MLflow experiment tracking integration for Phase 10.

This module provides a safe, opt-in wrapper around the MLflow tracking API.
It is an OBSERVABILITY layer only — it never makes ML decisions, controls
routing, or affects model selection.

Design principles:
- All public methods catch every exception and log a warning; the ML
  workflow is NEVER interrupted by a tracking failure.
- MLflow is imported lazily so the rest of the system works when mlflow
  is not installed.
- Tracking is disabled by default (MLFLOW_ENABLED=false).
- No raw dataset rows, no API keys, no credentials are logged.
"""

from __future__ import annotations

import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from utils.logger import get_logger

if TYPE_CHECKING:
    # Avoid hard runtime dependency on AgentState for import-order safety
    from utils.state import AgentState

logger = get_logger("MLflowTracker")


# ---------------------------------------------------------------------------
# Helper: safe metric value (MLflow rejects NaN / Inf)
# ---------------------------------------------------------------------------

def _safe_metric(value: Any) -> Optional[float]:
    """Return a finite float suitable for MLflow, or None if unusable."""
    try:
        v = float(value)
        if v != v or v == float("inf") or v == float("-inf"):  # NaN or Inf
            return None
        return v
    except (TypeError, ValueError):
        return None


def _log_metrics_safe(client: Any, run_id: str, metrics: Dict[str, Any]) -> None:
    """Log a dict of metrics, silently skipping non-finite values."""
    for key, value in metrics.items():
        v = _safe_metric(value)
        if v is not None:
            try:
                client.log_metric(run_id, key, v)
            except Exception:  # noqa: BLE001
                pass


def _log_params_safe(client: Any, run_id: str, params: Dict[str, Any]) -> None:
    """Log a dict of params, casting all values to str (MLflow requirement)."""
    for key, value in params.items():
        try:
            client.log_param(run_id, key, str(value)[:250])
        except Exception:  # noqa: BLE001
            pass


def _log_tags_safe(client: Any, run_id: str, tags: Dict[str, str]) -> None:
    """Log a dict of tags."""
    for key, value in tags.items():
        try:
            client.set_tag(run_id, key, str(value)[:250])
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------


class MLflowTracker:
    """Thin, failure-isolated wrapper around the MLflow tracking client.

    All public methods return ``None`` silently on any failure — the ML
    pipeline should never depend on or be interrupted by this class.

    Args:
        enabled: Whether MLflow logging is active.
        tracking_uri: Local directory or remote URI for the tracking store.
        experiment_name: MLflow experiment to use (created if absent).
    """

    def __init__(
        self,
        enabled: bool,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        self._enabled = enabled
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id: Optional[str] = None
        self._client: Any = None  # mlflow.MlflowClient — lazy import

        if self._enabled:
            self._setup_client()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_client(self) -> None:
        """Import mlflow and initialise tracking URI + experiment."""
        try:
            # In MLflow 3.x+, local filesystem store requires MLFLOW_ALLOW_FILE_STORE=true
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

            import mlflow  # noqa: PLC0415 (lazy import by design)

            from pathlib import Path  # noqa: PLC0415

            # Resolve relative URI relative to CWD; format Windows paths safely as file://
            raw_uri = self._tracking_uri
            if not raw_uri.startswith(("file://", "http://", "https://", "sqlite://", "postgresql://", "mysql://")):
                p = Path(raw_uri).resolve()
                p.mkdir(parents=True, exist_ok=True)
                uri = p.as_uri()
            else:
                uri = raw_uri

            mlflow.set_tracking_uri(uri)

            # Create experiment if it does not exist
            exp = mlflow.get_experiment_by_name(self._experiment_name)
            if exp is None:
                mlflow.create_experiment(self._experiment_name)

            self._client = mlflow.MlflowClient(tracking_uri=uri)
            logger.debug("[MLflow] Client initialised. URI=%s", uri)
        except ImportError:
            warnings.warn(
                "mlflow is not installed — experiment tracking is disabled. "
                "Install it with: pip install mlflow",
                stacklevel=2,
            )
            self._enabled = False
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[MLflow] Tracker setup failed (tracking disabled): {exc}",
                stacklevel=2,
            )
            self._enabled = False

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_workflow_run(
        self,
        dataset_path: str,
        enable_optimization: bool = True,
        enable_llm: bool = False,
    ) -> Optional[str]:
        """Start a new MLflow run for the entire ML workflow.

        Returns the run_id string, or None if disabled / failed.
        """
        if not self._enabled or self._client is None:
            return None
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(self._experiment_name)
            if exp is None:
                logger.warning("[MLflow] Experiment not found; skipping run.")
                return None

            run = self._client.create_run(
                experiment_id=exp.experiment_id,
                tags={
                    "mlflow.runName": os.path.basename(dataset_path),
                    "phase": "10",
                    "dataset": os.path.basename(dataset_path),
                    "enable_optimization": str(enable_optimization),
                    "enable_llm": str(enable_llm),
                },
            )
            self._run_id = run.info.run_id
            logger.info("[MLflow] Run started. run_id=%s", self._run_id)
            return self._run_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] start_workflow_run failed: %s", exc)
            return None

    def end_run(self, status: str = "FINISHED") -> None:
        """Finalise the active MLflow run.

        Args:
            status: One of ``"FINISHED"``, ``"FAILED"``, ``"KILLED"``.
        """
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            self._client.set_terminated(self._run_id, status=status)
            logger.info("[MLflow] Run ended. run_id=%s  status=%s", self._run_id, status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] end_run failed: %s", exc)

    # ------------------------------------------------------------------
    # Phase-specific logging helpers
    # ------------------------------------------------------------------

    def log_data_analysis(self, state: "AgentState") -> None:
        """Log dataset metadata and analysis outputs."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            params: Dict[str, Any] = {
                "dataset_name": state.dataset_name or "",
                "problem_type": state.problem_type or "",
                "target_column": state.target_column or "",
            }
            if state.dataset_shape:
                params["dataset_rows"] = state.dataset_shape[0]
                params["dataset_cols"] = state.dataset_shape[1]

            metrics: Dict[str, Any] = {
                "duplicate_count": state.duplicate_rows_count,
                "numerical_features": len(state.numerical_features),
                "categorical_features": len(state.categorical_features),
            }

            _log_params_safe(self._client, self._run_id, params)
            _log_metrics_safe(self._client, self._run_id, metrics)
            _log_tags_safe(
                self._client,
                self._run_id,
                {"analysis_status": "completed"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_data_analysis failed: %s", exc)

    def log_preprocessing(self, state: "AgentState") -> None:
        """Log preprocessing decisions."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            params: Dict[str, Any] = {
                "numerical_imputation": state.numerical_imputation_strategy,
                "categorical_imputation": state.categorical_imputation_strategy,
                "encoding_strategy": state.encoding_strategy,
                "scaling_strategy": state.scaling_strategy,
                "scale_numeric": state.scale_numeric,
                "duplicate_handling": state.duplicate_handling_strategy,
            }
            metrics: Dict[str, Any] = {
                "selected_numerical_features": len(state.selected_numerical_features),
                "selected_categorical_features": len(state.selected_categorical_features),
                "excluded_features": len(state.excluded_features),
                "duplicate_rows_removed": state.duplicate_rows_removed,
            }
            _log_params_safe(self._client, self._run_id, params)
            _log_metrics_safe(self._client, self._run_id, metrics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_preprocessing failed: %s", exc)

    def log_training_results(self, state: "AgentState") -> None:
        """Log baseline training results for each model."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            _log_params_safe(
                self._client,
                self._run_id,
                {
                    "selected_models": ",".join(state.selected_models),
                    "experiment_count": state.experiment_count,
                    "best_baseline_model": state.best_baseline_model or "",
                },
            )
            # Per-model metrics with model prefix
            for result in state.experiment_results:
                if not isinstance(result, dict):
                    continue
                model = result.get("model_name", "unknown").replace(" ", "_").lower()
                for metric_key in ("accuracy", "precision", "recall", "f1_score",
                                   "cv_f1_mean", "cv_accuracy_mean", "training_time"):
                    value = result.get(metric_key)
                    if value is not None:
                        v = _safe_metric(value)
                        if v is not None:
                            try:
                                self._client.log_metric(
                                    self._run_id, f"baseline_{model}_{metric_key}", v
                                )
                            except Exception:  # noqa: BLE001
                                pass

            # Best baseline summary metrics
            if state.best_baseline_metrics:
                _log_metrics_safe(
                    self._client,
                    self._run_id,
                    {f"best_baseline_{k}": v for k, v in state.best_baseline_metrics.items()},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_training_results failed: %s", exc)

    def log_evaluation(self, state: "AgentState") -> None:
        """Log evaluation agent outputs."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            _log_tags_safe(
                self._client,
                self._run_id,
                {"evaluation_status": state.evaluation_status},
            )
            # Per-model evaluation summary
            for eval_record in state.model_evaluations:
                if not isinstance(eval_record, dict):
                    continue
                model = eval_record.get("model_name", "unknown").replace(" ", "_").lower()
                for metric_key in ("accuracy", "precision", "recall", "f1_score",
                                   "cv_f1_mean", "f1_gap"):
                    value = eval_record.get(metric_key)
                    if value is not None:
                        v = _safe_metric(value)
                        if v is not None:
                            try:
                                self._client.log_metric(
                                    self._run_id, f"eval_{model}_{metric_key}", v
                                )
                            except Exception:  # noqa: BLE001
                                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_evaluation failed: %s", exc)

    def log_optimization(self, state: "AgentState") -> None:
        """Log Optuna optimization results."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            _log_params_safe(
                self._client,
                self._run_id,
                {
                    "optimization_status": state.optimization_status,
                    "best_optimized_model": state.best_optimized_model or "none",
                    "hyperparameter_experiments": state.hyperparameter_experiments,
                    "successful_trials": state.successful_trials,
                    "failed_trials": state.failed_trials,
                },
            )
            if state.best_optimized_metrics:
                _log_metrics_safe(
                    self._client,
                    self._run_id,
                    {f"best_optimized_{k}": v for k, v in state.best_optimized_metrics.items()},
                )

            # Best hyperparameters per model
            for model_name, hp_dict in state.best_hyperparameters.items():
                safe_name = model_name.replace(" ", "_").lower()
                for hp_key, hp_val in hp_dict.items():
                    try:
                        self._client.log_param(
                            self._run_id,
                            f"hp_{safe_name}_{hp_key}",
                            str(hp_val)[:250],
                        )
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_optimization failed: %s", exc)

    def log_persistence(self, state: "AgentState") -> None:
        """Log model persistence information."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            _log_params_safe(
                self._client,
                self._run_id,
                {
                    "persistence_status": state.persistence_status,
                    "persisted_model_path": state.persisted_model_path or "",
                    "persisted_metadata_path": state.persisted_metadata_path or "",
                },
            )
            _log_tags_safe(
                self._client,
                self._run_id,
                {"persistence_status": state.persistence_status},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_persistence failed: %s", exc)

    def log_reproducibility(self) -> None:
        """Log software environment metadata for reproducibility."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            tags: Dict[str, str] = {
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "python_executable": sys.executable,
            }

            # Safely record key library versions
            for lib_name in ("sklearn", "pandas", "numpy", "optuna", "mlflow", "langgraph"):
                try:
                    import importlib  # noqa: PLC0415
                    mod = importlib.import_module(lib_name if lib_name != "sklearn" else "sklearn")
                    version = getattr(mod, "__version__", "unknown")
                    tags[f"lib_{lib_name}_version"] = version
                except ImportError:
                    tags[f"lib_{lib_name}_version"] = "not_installed"

            _log_tags_safe(self._client, self._run_id, tags)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_reproducibility failed: %s", exc)

    def log_workflow_summary(self, state: "AgentState", elapsed_seconds: float) -> None:
        """Log final workflow telemetry."""
        if not self._enabled or self._client is None or self._run_id is None:
            return
        try:
            _log_metrics_safe(
                self._client,
                self._run_id,
                {"workflow_elapsed_seconds": elapsed_seconds},
            )
            _log_tags_safe(
                self._client,
                self._run_id,
                {
                    "workflow_status": str(
                        state.workflow_status.value
                        if hasattr(state.workflow_status, "value")
                        else state.workflow_status
                    ),
                    "llm_enabled": str(state.llm_enabled),
                    "llm_status": str(state.llm_status),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] log_workflow_summary failed: %s", exc)

    # ------------------------------------------------------------------
    # Read-side helpers (for dashboard)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        """Return the active run_id, or None."""
        return self._run_id

    @property
    def enabled(self) -> bool:
        """Return whether MLflow logging is actually active."""
        return self._enabled

    def get_recent_runs(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Return recent runs as a list of dicts (for dashboard display).

        Never raises — returns empty list on any failure.
        """
        if not self._enabled or self._client is None:
            return []
        try:
            import mlflow  # noqa: PLC0415

            exp = mlflow.get_experiment_by_name(self._experiment_name)
            if exp is None:
                return []
            runs = self._client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            results = []
            for run in runs:
                results.append(
                    {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "end_time": run.info.end_time,
                        "dataset": run.data.tags.get("dataset", ""),
                        "best_baseline_model": run.data.params.get("best_baseline_model", ""),
                        "best_optimized_model": run.data.params.get("best_optimized_model", ""),
                        "optimization_status": run.data.params.get("optimization_status", ""),
                        "workflow_status": run.data.tags.get("workflow_status", ""),
                        "metrics": dict(run.data.metrics),
                        "params": dict(run.data.params),
                    }
                )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MLflow] get_recent_runs failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Factory — creates a tracker from Config values
# ---------------------------------------------------------------------------


def create_tracker_from_config() -> MLflowTracker:
    """Build an MLflowTracker using the global AppConfig.

    Returns a disabled (no-op) tracker when MLFLOW_ENABLED is false.
    """
    from utils.config import Config  # noqa: PLC0415

    return MLflowTracker(
        enabled=Config.MLFLOW_ENABLED,
        tracking_uri=Config.MLFLOW_TRACKING_URI,
        experiment_name=Config.MLFLOW_EXPERIMENT_NAME,
    )
