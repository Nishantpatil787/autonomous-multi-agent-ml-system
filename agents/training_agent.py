"""Training Agent for autonomous leakage-free model training, cross-validation, and benchmarking."""

import time
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent
from tools import training_tools
from utils.config import Config
from utils.state import AgentState, ExperimentResult, WorkflowStatus


class TrainingAgent(BaseAgent):
    """Specialized agent responsible for training candidate models, executing CV, and ranking performance."""

    def __init__(self, name: str = "TrainingAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        cv_folds: Optional[int] = None,
        test_size: Optional[float] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Execute model training on train split, evaluate on test split, run CV, and rank experiments.

        Args:
            state: The shared workflow state.
            cv_folds: Optional override for cross-validation folds.
            test_size: Optional override for test split ratio.

        Returns:
            Updated AgentState with experiment results, model comparisons, and best baseline model.
        """
        # 1. Validate prerequisites
        if not state.dataset_path:
            err_msg = "Cannot execute training: dataset_path is missing from state."
            return self._record_failure(state, err_msg)

        if not state.target_column:
            err_msg = "Cannot execute training: target_column is missing from state."
            return self._record_failure(state, err_msg)

        if not state.selected_models:
            err_msg = "Cannot execute training: No models selected in state. Run ModelSelectionAgent first."
            return self._record_failure(state, err_msg)

        if not state.pipeline_config:
            err_msg = "Cannot execute training: Preprocessing pipeline configuration missing from state."
            return self._record_failure(state, err_msg)

        self.logger.info(f"Initiating training experiments for {len(state.selected_models)} models.")
        state.workflow_status = WorkflowStatus.TRAINING
        state.add_log(self.name, f"Starting training experiments on '{state.dataset_name}'")

        # 2. Load dataset and handle duplicate rows in-memory
        try:
            df, dups_removed = training_tools.load_training_dataframe(
                state.dataset_path,
                duplicate_strategy=state.duplicate_handling_strategy,
            )
        except Exception as exc:
            return self._record_failure(state, f"Failed to load dataset: {str(exc)}")

        # 3. Separate features and target (strictly preventing target leakage)
        try:
            X, y = training_tools.prepare_features_and_target(
                df,
                target_column=state.target_column,
                excluded_features=state.excluded_features,
            )
        except Exception as exc:
            return self._record_failure(state, f"Failed to prepare feature matrix: {str(exc)}")

        # 4. Perform train/test split
        split_ratio = test_size if test_size is not None else Config.TEST_SIZE
        X_train, X_test, y_train, y_test, is_stratified, split_reason = training_tools.split_dataset(
            X,
            y,
            test_size=split_ratio,
            random_state=Config.RANDOM_STATE,
        )

        state.split_info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": split_ratio,
            "stratified": is_stratified,
            "split_reason": split_reason,
            "random_state": Config.RANDOM_STATE,
        }
        self.logger.info(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples ({split_reason})")

        folds = cv_folds if cv_folds is not None else Config.CV_FOLDS
        experiment_records: List[Dict[str, Any]] = []

        # 5. Train each candidate model within a leakage-free pipeline
        for idx, model_key in enumerate(state.selected_models, start=1):
            exp_id = f"EXP_{idx:03d}_{model_key}"
            self.logger.info(f"[{idx}/{len(state.selected_models)}] Training model: {model_key} ({exp_id})")

            try:
                # Assemble pipeline with model-specific scaling
                pipeline, scaling_used = training_tools.build_model_pipeline(
                    state.pipeline_config,
                    model_key=model_key,
                )

                # Train model strictly on X_train (Zero Data Leakage)
                t_fit_start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                t_fit = round(time.perf_counter() - t_fit_start, 4)

                # Predict on held-out test data
                t_pred_start = time.perf_counter()
                y_pred = pipeline.predict(X_test)
                t_pred = round(time.perf_counter() - t_pred_start, 4)

                # Evaluate test metrics
                metrics = training_tools.evaluate_classifier(y_test, y_pred)
                cm_data = training_tools.calculate_confusion_matrix(y_test, y_pred)

                # Run cross-validation on X_train (using fresh pipeline instance)
                cv_pipeline, _ = training_tools.build_model_pipeline(
                    state.pipeline_config,
                    model_key=model_key,
                )
                cv_res = training_tools.run_cross_validation(
                    cv_pipeline,
                    X_train,
                    y_train,
                    n_splits=folds,
                    random_state=Config.RANDOM_STATE,
                )

                exp = ExperimentResult(
                    experiment_id=exp_id,
                    model_name=model_key,
                    preprocessing_scaling=scaling_used,
                    parameters=pipeline.named_steps["model"].get_params(),
                    train_samples=len(X_train),
                    test_samples=len(X_test),
                    feature_count_raw=X_train.shape[1],
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    cv_accuracy_mean=cv_res.get("cv_accuracy_mean"),
                    cv_accuracy_std=cv_res.get("cv_accuracy_std"),
                    cv_f1_mean=cv_res.get("cv_f1_mean"),
                    cv_folds=cv_res.get("cv_folds", 0),
                    training_time=t_fit,
                    prediction_time=t_pred,
                    total_time=round(t_fit + t_pred, 4),
                    confusion_matrix=cm_data["matrix"],
                    confusion_matrix_labels=cm_data["labels"],
                    status="success",
                )
                exp_dict = exp.__dict__
                experiment_records.append(exp_dict)
                self.logger.info(
                    f"Completed {model_key}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
                    f"CV F1={cv_res.get('cv_f1_mean')} (in {t_fit:.3f}s)"
                )
            except Exception as exc:
                err = f"Model '{model_key}' training failed: {str(exc)}"
                self.logger.error(err)
                exp_dict = {
                    "experiment_id": exp_id,
                    "model_name": model_key,
                    "preprocessing_scaling": False,
                    "status": "failed",
                    "error_message": str(exc),
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "training_time": 0.0,
                }
                experiment_records.append(exp_dict)

        # 6. Rank experiments and determine best baseline model
        ranked_comparison, best_exp = training_tools.rank_experiments(experiment_records)

        state.experiment_results = experiment_records
        state.model_comparison = ranked_comparison
        state.experiment_count = len(experiment_records)

        if best_exp:
            state.best_baseline_model = best_exp["model_name"]
            state.best_model_name = best_exp["model_name"]
            state.best_baseline_metrics = {
                "f1_score": best_exp["f1_score"],
                "accuracy": best_exp["accuracy"],
                "precision": best_exp["precision"],
                "recall": best_exp["recall"],
            }
            self.logger.info(
                f"Best baseline model identified: {state.best_baseline_model} "
                f"(F1: {best_exp['f1_score']:.4f}, Acc: {best_exp['accuracy']:.4f})"
            )
        else:
            self.logger.warning("No experiment succeeded during training.")

        state.workflow_status = WorkflowStatus.TRAINING_COMPLETED
        state.add_log(
            self.name,
            f"Training completed for {len(experiment_records)} models. Best baseline: {state.best_baseline_model}.",
            details={
                "best_model": state.best_baseline_model,
                "best_metrics": state.best_baseline_metrics,
                "total_experiments": len(experiment_records),
            },
        )
        return state

    def _record_failure(self, state: AgentState, message: str) -> AgentState:
        """Helper to record failure in state and logger."""
        self.logger.error(message)
        state.workflow_status = WorkflowStatus.FAILED
        state.error_message = message
        state.add_log(self.name, message, status="error")
        return state
