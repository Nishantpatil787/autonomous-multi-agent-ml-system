"""Hyperparameter Optimization Agent using Optuna.

Consumes Phase 4/5 results (trained baseline models, evaluation rankings)
and autonomously optimizes the top-N candidate models via Optuna TPE search.
Results are stored in AgentState without retaining live sklearn or Optuna
objects (all state fields are JSON-serializable dicts).

Leakage guarantee:
  - The Optuna objective function uses ONLY X_train / y_train.
  - Holdout evaluation (X_test / y_test) is performed ONCE per model AFTER
    the study completes, and only when evaluate_holdout=True.
"""

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from tools import optimization_tools, training_tools
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState, WorkflowStatus

logger = get_logger("OptimizationAgent")


class OptimizationAgent(BaseAgent):
    """Specialized agent responsible for Optuna-based hyperparameter optimization.

    Workflow:
    1. Validate that the Evaluation Agent has completed (EVALUATION_COMPLETED).
    2. Select top-N models from state.model_evaluations by rank.
    3. Re-load and re-split the dataset deterministically (same seed as Phase 4).
    4. For each candidate, run an Optuna study (train data only).
    5. Optionally evaluate the best params on the held-out test set.
    6. Persist JSON-safe results to AgentState.
    7. Preserve state.best_baseline_model; set state.best_optimized_model.
    """

    def __init__(self, name: str = "OptimizationAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        n_trials: Optional[int] = None,
        top_n: Optional[int] = None,
        evaluate_holdout: bool = True,
        cv_folds: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Execute hyperparameter optimization for the top-N baseline models.

        Args:
            state: Shared workflow state (must have EVALUATION_COMPLETED status).
            n_trials: Number of Optuna trials per model (default: Config.OPTUNA_N_TRIALS).
            top_n: Number of top baseline models to optimize (default: Config.OPTUNA_TOP_MODELS).
            evaluate_holdout: If True, evaluate best params on held-out test set once.
            cv_folds: Override for CV folds inside Optuna objective.
            timeout: Wall-clock timeout (seconds) per Optuna study.

        Returns:
            Updated AgentState with optimization results.
        """
        # ------------------------------------------------------------------
        # 0. Resolve runtime parameters
        # ------------------------------------------------------------------
        n_trials = n_trials if n_trials is not None else Config.OPTUNA_N_TRIALS
        top_n = top_n if top_n is not None else Config.OPTUNA_TOP_MODELS
        folds = cv_folds if cv_folds is not None else Config.CV_FOLDS
        opt_timeout = timeout if timeout is not None else Config.OPTUNA_TIMEOUT_SECONDS
        seed = Config.OPTUNA_RANDOM_SEED

        # ------------------------------------------------------------------
        # 1. Validate prerequisites
        # ------------------------------------------------------------------
        missing = self._check_prerequisites(state)
        if missing:
            return self._record_failure(state, missing)

        self.logger.info(
            f"Starting OptimizationAgent: top_n={top_n}, n_trials={n_trials}, seed={seed}"
        )
        state.workflow_status = WorkflowStatus.OPTIMIZING
        state.add_log(
            self.name,
            f"Hyperparameter optimization started (top_n={top_n}, n_trials={n_trials}).",
        )

        # ------------------------------------------------------------------
        # 2. Select candidate models from Phase 5 ranking
        # ------------------------------------------------------------------
        candidates = optimization_tools.select_candidate_models(
            state.model_evaluations, top_n=top_n
        )
        if not candidates:
            return self._record_failure(
                state, "No ranked models found in state.model_evaluations. Run EvaluationAgent first."
            )
        self.logger.info(f"Candidate models selected for optimization: {candidates}")

        # ------------------------------------------------------------------
        # 3. Re-load dataset and prepare features (same as TrainingAgent)
        # ------------------------------------------------------------------
        try:
            df, _ = training_tools.load_training_dataframe(
                state.dataset_path,
                duplicate_strategy=state.duplicate_handling_strategy,
            )
        except Exception as exc:
            return self._record_failure(state, f"Failed to load dataset: {exc}")

        try:
            X, y = training_tools.prepare_features_and_target(
                df,
                target_column=state.target_column,
                excluded_features=state.excluded_features,
            )
        except Exception as exc:
            return self._record_failure(state, f"Failed to prepare feature matrix: {exc}")

        # ------------------------------------------------------------------
        # 4. Reproduce the exact same train/test split as Phase 4
        # ------------------------------------------------------------------
        X_train, X_test, y_train, y_test, _, _ = training_tools.split_dataset(
            X,
            y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
        )
        self.logger.info(
            f"Dataset split reproduced: {len(X_train)} train / {len(X_test)} test samples."
        )

        # Build a baseline CV F1 lookup from experiment_results for reporting
        baseline_cv_map: Dict[str, float] = {}
        for exp in state.experiment_results:
            mn = exp.get("model_name", "")
            cv_f1 = exp.get("cv_f1_mean")
            if mn and cv_f1 is not None:
                baseline_cv_map[mn] = float(cv_f1)

        # ------------------------------------------------------------------
        # 5. Run Optuna optimization for each candidate model
        # ------------------------------------------------------------------
        optimization_results: List[Dict[str, Any]] = []
        total_hyperparameter_experiments = 0
        total_successful = 0
        total_failed = 0

        for model_key in candidates:
            self.logger.info(f"Optimizing model: {model_key} ({n_trials} trials)...")

            try:
                _, trial_records, result_dict = optimization_tools.run_optimization(
                    model_key=model_key,
                    X_train=X_train,
                    y_train=y_train,
                    pipeline_config=state.pipeline_config,
                    n_trials=n_trials,
                    seed=seed,
                    cv_folds=folds,
                    timeout=opt_timeout,
                )
            except Exception as exc:
                err_msg = f"Optimization of '{model_key}' failed unexpectedly: {exc}"
                self.logger.error(err_msg)
                result_dict = {
                    "model_name": model_key,
                    "best_params": {},
                    "optimized_cv_f1": 0.0,
                    "n_trials": n_trials,
                    "successful_trials": 0,
                    "failed_trials": n_trials,
                    "optimization_time": 0.0,
                    "status": "failed",
                    "error_message": err_msg,
                    "holdout_metrics": {},
                    "baseline_cv_f1": baseline_cv_map.get(model_key, 0.0),
                    "improvement": 0.0,
                    "improvement_pct": None,
                }
                optimization_results.append(result_dict)
                total_failed += n_trials
                continue

            # Attach baseline CV F1 for comparison
            baseline_cv = baseline_cv_map.get(model_key, 0.0)
            result_dict["baseline_cv_f1"] = baseline_cv

            # Calculate improvement
            improvement, improvement_pct = optimization_tools.calculate_improvement(
                baseline_cv_f1=baseline_cv,
                optimized_cv_f1=result_dict["optimized_cv_f1"],
            )
            result_dict["improvement"] = improvement
            result_dict["improvement_pct"] = improvement_pct

            # Accumulate trial counts
            total_hyperparameter_experiments += result_dict.get("n_trials", n_trials)
            total_successful += result_dict.get("successful_trials", 0)
            total_failed += result_dict.get("failed_trials", 0)

            # ------------------------------------------------------------------
            # 5b. Optional: one-shot holdout evaluation with best params
            # ------------------------------------------------------------------
            if evaluate_holdout and result_dict["status"] == "completed":
                try:
                    holdout_metrics = optimization_tools.evaluate_optimized_on_holdout(
                        model_key=model_key,
                        best_params=result_dict["best_params"],
                        pipeline_config=state.pipeline_config,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )
                    result_dict["holdout_metrics"] = holdout_metrics
                    self.logger.info(
                        f"Holdout evaluation for '{model_key}': "
                        f"F1={holdout_metrics.get('f1_score', 0.0):.4f}, "
                        f"Acc={holdout_metrics.get('accuracy', 0.0):.4f}"
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"Holdout evaluation failed for '{model_key}': {exc}. "
                        "Continuing without holdout metrics."
                    )
                    result_dict["holdout_metrics"] = {}

            self.logger.info(
                f"'{model_key}' optimization done: "
                f"baseline_cv_f1={baseline_cv:.4f}, "
                f"optimized_cv_f1={result_dict['optimized_cv_f1']:.4f}, "
                f"improvement={improvement:+.4f}"
            )
            optimization_results.append(result_dict)

        # ------------------------------------------------------------------
        # 6. Determine best optimized model (by optimized_cv_f1 among completed)
        # ------------------------------------------------------------------
        completed_results = [
            r for r in optimization_results
            if r.get("status") == "completed"
        ]
        if completed_results:
            best_result = max(
                completed_results, key=lambda r: r.get("optimized_cv_f1", 0.0)
            )
            best_optimized_model = best_result["model_name"]
            best_optimized_metrics = {
                "cv_f1": best_result.get("optimized_cv_f1", 0.0),
                **best_result.get("holdout_metrics", {}),
            }
        else:
            best_optimized_model = None
            best_optimized_metrics = {}

        # ------------------------------------------------------------------
        # 7. Build comparison table
        # ------------------------------------------------------------------
        comparison = optimization_tools.build_optimization_comparison(
            model_evaluations=state.model_evaluations,
            optimization_results=optimization_results,
        )

        # ------------------------------------------------------------------
        # 8. Build optimization summary
        # ------------------------------------------------------------------
        opt_summary: Dict[str, Any] = {
            "candidates_optimized": candidates,
            "n_trials_per_model": n_trials,
            "total_hyperparameter_experiments": total_hyperparameter_experiments,
            "total_successful_trials": total_successful,
            "total_failed_trials": total_failed,
            "best_optimized_model": best_optimized_model,
            "best_baseline_model": state.best_baseline_model,
            "holdout_evaluation_performed": evaluate_holdout,
            "seed": seed,
            "cv_folds": folds,
        }

        # ------------------------------------------------------------------
        # 9. Persist to AgentState (JSON-serializable dicts only)
        # ------------------------------------------------------------------
        # Merge into existing dict-based optimization_results field
        state.optimization_results = {
            r["model_name"]: r for r in optimization_results
        }
        state.best_hyperparameters = {
            r["model_name"]: r.get("best_params", {})
            for r in optimization_results
        }

        # Phase 6 specific fields (added to state via update)
        state.optimization_status = "completed" if completed_results else "failed"
        state.optimization_comparison = comparison
        state.best_optimized_model = best_optimized_model
        state.best_optimized_metrics = best_optimized_metrics
        state.optimization_summary = opt_summary
        state.hyperparameter_experiments = total_hyperparameter_experiments
        state.successful_trials = total_successful
        state.failed_trials = total_failed

        state.workflow_status = WorkflowStatus.OPTIMIZATION_COMPLETED
        state.add_log(
            self.name,
            f"Optimization completed. Best optimized model: {best_optimized_model}. "
            f"Total trials: {total_hyperparameter_experiments} "
            f"({total_successful} succeeded, {total_failed} failed).",
            status="success",
            details={
                "best_optimized_model": best_optimized_model,
                "best_optimized_metrics": best_optimized_metrics,
                "candidates": candidates,
                "total_trials": total_hyperparameter_experiments,
            },
        )
        self.logger.info(
            f"OptimizationAgent complete. Best optimized model: {best_optimized_model}. "
            f"Baseline best: {state.best_baseline_model}."
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_prerequisites(self, state: AgentState) -> Optional[str]:
        """Return an error string if prerequisites are not met, else None."""
        if not state.dataset_path:
            return "dataset_path is missing from state."
        if not state.target_column:
            return "target_column is missing from state."
        if not state.pipeline_config:
            return "pipeline_config is missing from state. Run PreprocessingAgent first."
        if not state.model_evaluations:
            return "model_evaluations is empty. Run EvaluationAgent first."
        if state.workflow_status not in (
            WorkflowStatus.EVALUATION_COMPLETED,
            WorkflowStatus.OPTIMIZATION_COMPLETED,
        ):
            return (
                f"Expected EVALUATION_COMPLETED or OPTIMIZATION_COMPLETED status, "
                f"got '{state.workflow_status}'. Complete Phase 5 before Phase 6."
            )
        return None

    def _record_failure(self, state: AgentState, message: str) -> AgentState:
        """Record a failure in the state and logger."""
        self.logger.error(message)
        state.workflow_status = WorkflowStatus.FAILED
        state.error_message = message
        state.add_log(self.name, message, status="error")
        return state
