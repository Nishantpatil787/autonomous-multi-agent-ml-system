"""Persistence Agent for autonomous final model selection and safe artifact export."""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from tools import persistence_tools
from utils.config import Config
from utils.logger import get_logger
from utils.state import AgentState, WorkflowStatus

logger = get_logger("PersistenceAgent")


class PersistenceAgent(BaseAgent):
    """Specialized agent responsible for final model selection and joblib artifact persistence."""

    def __init__(self, name: str = "PersistenceAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        model_filename: Optional[str] = None,
        metadata_filename: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Select best model (optimized vs baseline), fit deployable pipeline, and persist artifacts.

        Args:
            state: Shared workflow state.
            model_filename: Optional filename override for model joblib file.
            metadata_filename: Optional filename override for metadata JSON file.

        Returns:
            Updated AgentState containing persisted model paths and metadata.
        """
        # 1. Verify prerequisites
        if not state.dataset_path or not state.target_column or not state.pipeline_config:
            err_msg = "Cannot persist model: dataset_path, target_column, or pipeline_config missing."
            self.logger.error(err_msg)
            state.persistence_status = "failed"
            state.add_log(self.name, err_msg, status="error")
            return state

        if not state.best_baseline_model and not state.model_evaluations:
            err_msg = "Cannot persist model: No trained or evaluated models found in state."
            self.logger.error(err_msg)
            state.persistence_status = "failed"
            state.add_log(self.name, err_msg, status="error")
            return state

        self.logger.info("Initiating model persistence workflow.")
        state.add_log(self.name, "Starting final model selection and persistence.")

        # 2. Select final model according to policy
        selection = persistence_tools.select_final_model_for_persistence(state)
        selected_model_name = selection["model_name"]
        selection_source = selection["selection_source"]
        best_params = selection.get("best_params", {})

        # 3. Fit the complete final pipeline on train split
        try:
            pipeline, feature_cols = persistence_tools.fit_final_pipeline_for_persistence(
                state=state,
                model_name=selected_model_name,
                best_params=best_params,
            )
        except Exception as exc:
            err_msg = f"Failed to fit final pipeline for '{selected_model_name}': {str(exc)}"
            self.logger.error(err_msg)
            state.persistence_status = "failed"
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        # 4. Construct clean, serializable metadata
        # Extract all experiments from state if present
        all_exps = []
        if state.experiment_results and isinstance(state.experiment_results, list):
            for exp in state.experiment_results:
                if isinstance(exp, dict):
                    all_exps.append({
                        "model_name": str(exp.get("model_name", "")).replace("_", " ").title(),
                        "f1_score": round(float(exp.get("f1_score", 0.0) or 0.0), 4),
                        "accuracy": round(float(exp.get("accuracy", 0.0) or 0.0), 4),
                        "precision": round(float(exp.get("precision", 0.0) or 0.0), 4),
                        "recall": round(float(exp.get("recall", 0.0) or 0.0), 4),
                        "cv_f1_mean": round(float(exp.get("cv_f1_mean", 0.0) or 0.0), 4),
                        "training_time": round(float(exp.get("training_time", 0.0) or 0.0), 3),
                    })

        # Extract confusion matrix from state
        cm_dict = None
        if state.confusion_matrices and isinstance(state.confusion_matrices, dict):
            cm_obj = state.confusion_matrices.get(selected_model_name, {})
            if isinstance(cm_obj, dict) and "matrix" in cm_obj:
                cm_dict = cm_obj
        if not cm_dict and state.evaluation_report and isinstance(state.evaluation_report, dict):
            if "confusion_matrix" in state.evaluation_report:
                cm_dict = state.evaluation_report["confusion_matrix"]
        if not cm_dict and state.confusion_matrix_analyses and isinstance(state.confusion_matrix_analyses, dict):
            cm_dict = state.confusion_matrix_analyses.get(selected_model_name)

        # Extract optimization comparison from state
        opt_comp = []
        if state.optimization_comparison and isinstance(state.optimization_comparison, list):
            opt_comp = state.optimization_comparison
        elif state.optimization_results:
            results_iter = (
                state.optimization_results.values()
                if isinstance(state.optimization_results, dict)
                else state.optimization_results
            )
            for r in results_iter:
                if isinstance(r, dict):
                    opt_comp.append({
                        "model": str(r.get("model_name", "")).replace("_", " ").title(),
                        "baseline_f1": round(float(r.get("baseline_cv_f1", 0.0) or 0.0), 4),
                        "optimized_f1": round(float(r.get("optimized_cv_f1", 0.0) or 0.0), 4),
                        "improvement": round(float(r.get("cv_improvement", 0.0) or 0.0), 4),
                        "improvement_pct": round(float(r.get("cv_improvement_pct", 0.0) or 0.0), 2),
                    })

        sel_metrics = selection.get("metrics", {})
        metadata: Dict[str, Any] = {
            "model_name": selected_model_name,
            "selection_source": selection_source,
            "optimization_used": selection.get("optimization_used", False),
            "target_column": state.target_column,
            "feature_columns": feature_cols,
            "task_type": state.problem_type or "classification",
            "accuracy": sel_metrics.get("accuracy"),
            "metrics": sel_metrics,
            "cv_metrics": selection.get("cv_metrics", {}),
            "best_params": best_params,
            "all_experiments": all_exps,
            "confusion_matrix": cm_dict,
            "optimization_comparison": opt_comp,
            "numerical_features": state.selected_numerical_features or state.numerical_features,
            "categorical_features": state.selected_categorical_features or state.categorical_features,
            "dataset_name": state.dataset_name or Path(state.dataset_path).name,
            "persisted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_stages": ["preprocessor", "model"],
            "preprocessor_config": {
                "numerical_imputation": state.pipeline_config.get("numerical_imputation_strategy"),
                "categorical_imputation": state.pipeline_config.get("categorical_imputation_strategy"),
                "scaling": state.pipeline_config.get("scaling_strategy"),
            },
        }

        # 5. Persist artifacts
        m_filename = model_filename or persistence_tools.DEFAULT_MODEL_FILENAME
        meta_filename = metadata_filename or persistence_tools.DEFAULT_METADATA_FILENAME

        m_path = Config.MODELS_DIR / m_filename
        meta_path = Config.MODELS_DIR / meta_filename

        try:
            persisted_m, persisted_meta = persistence_tools.save_persisted_model(
                pipeline=pipeline,
                metadata=metadata,
                model_path=m_path,
                metadata_path=meta_path,
            )
            state.persisted_model_path = str(persisted_m)
            state.persisted_metadata_path = str(persisted_meta)
            state.persisted_model_metadata = metadata
            state.persistence_status = "completed"

            self.logger.info(
                f"Persistence complete for model '{selected_model_name}' ({selection_source}). "
                f"Saved to: {persisted_m}"
            )
            state.add_log(
                self.name,
                f"Successfully persisted final model '{selected_model_name}' ({selection_source}).",
                status="success",
                details={"model_path": str(persisted_m), "metadata_path": str(persisted_meta)},
            )
        except Exception as exc:
            err_msg = f"Failed to persist model artifacts: {str(exc)}"
            self.logger.error(err_msg)
            state.persistence_status = "failed"
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")

        return state
