"""Evaluation Agent for diagnostic assessment, hold-out vs CV gap analysis, and final synthesis."""

from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent
from tools import evaluation_tools
from utils.config import Config
from utils.state import AgentState, GapAnalysis, WorkflowStatus


class EvaluationAgent(BaseAgent):
    """Specialized agent responsible for in-depth diagnostic evaluation of baseline experiment results."""

    def __init__(self, name: str = "EvaluationAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        close_threshold: Optional[float] = None,
        moderate_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Analyze existing experiment results, compute gap diagnostics, and produce final evaluation report.

        Args:
            state: The shared workflow state populated with Phase 4 training results.
            close_threshold: Optional threshold for close agreement in gap analysis.
            moderate_threshold: Optional threshold for moderate difference in gap analysis.

        Returns:
            Updated AgentState containing comprehensive diagnostic report and findings.
        """
        # 1. Verify prerequisites
        if not state.experiment_results:
            err_msg = (
                "Cannot execute EvaluationAgent: No experiment results found in state. "
                "TrainingAgent must be executed first."
            )
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.evaluation_status = "failed"
            state.add_log(self.name, err_msg, status="error")
            return state

        self.logger.info(f"Beginning diagnostic evaluation of {len(state.experiment_results)} experiment results.")
        state.workflow_status = WorkflowStatus.EVALUATING
        state.add_log(self.name, "Starting comprehensive model evaluation and gap analysis.")

        close_thresh = close_threshold if close_threshold is not None else Config.GAP_THRESHOLD_CLOSE
        mod_thresh = moderate_threshold if moderate_threshold is not None else Config.GAP_THRESHOLD_MODERATE

        # 2. Perform Hold-out vs Cross-Validation Gap Analysis
        gap_records: List[Dict[str, Any]] = []
        for exp in state.experiment_results:
            model_name = exp.get("model_name", "unknown")
            if exp.get("status") != "success":
                continue

            holdout_f1 = exp.get("f1_score", 0.0)
            cv_f1 = exp.get("cv_f1_mean")
            f1_gap, f1_gap_abs = evaluation_tools.calculate_gap(holdout_f1, cv_f1)
            f1_cls = evaluation_tools.classify_performance_gap(f1_gap_abs, close_thresh, mod_thresh)

            holdout_acc = exp.get("accuracy", 0.0)
            cv_acc = exp.get("cv_accuracy_mean")
            acc_gap, acc_gap_abs = evaluation_tools.calculate_gap(holdout_acc, cv_acc)
            acc_cls = evaluation_tools.classify_performance_gap(acc_gap_abs, close_thresh, mod_thresh)

            interpretation = evaluation_tools.interpret_gap(
                model_name=model_name,
                metric_name="F1-score",
                holdout_val=holdout_f1,
                cv_val=cv_f1,
                gap=f1_gap,
                classification=f1_cls,
            )

            gap_item = GapAnalysis(
                model_name=model_name,
                holdout_f1=holdout_f1,
                cv_f1=cv_f1,
                f1_gap=f1_gap,
                f1_gap_abs=f1_gap_abs,
                f1_classification=f1_cls,
                holdout_accuracy=holdout_acc,
                cv_accuracy=cv_acc,
                accuracy_gap=acc_gap,
                accuracy_gap_abs=acc_gap_abs,
                accuracy_classification=acc_cls,
                interpretation=interpretation,
            )
            gap_records.append(gap_item.__dict__)

        # 3. Perform Confusion Matrix Diagnostics
        cm_analyses: Dict[str, Dict[str, Any]] = {}
        for exp in state.experiment_results:
            model_name = exp.get("model_name", "unknown")
            cm_data = exp.get("confusion_matrix", [])
            labels = exp.get("confusion_matrix_labels", [])
            cm_analyses[model_name] = evaluation_tools.analyze_confusion_matrix(
                cm_data=cm_data,
                labels=labels,
                model_name=model_name,
            )

        # 4. Analyze Execution Times & Experiment Counts
        time_analysis = evaluation_tools.analyze_execution_time(state.experiment_results)

        # 5. Build Model Comparison and Ranking
        ranked_evaluations = evaluation_tools.compare_and_rank_models(
            experiments=state.experiment_results,
            close_threshold=close_thresh,
            moderate_threshold=mod_thresh,
        )

        # 6. Generate Complete Structured Evaluation Report
        evaluation_report = evaluation_tools.generate_evaluation_report(
            ranked_models=ranked_evaluations,
            gap_analyses=gap_records,
            cm_analyses=cm_analyses,
            time_analysis=time_analysis,
            problem_type=state.problem_type or "classification",
        )

        # 7. Update AgentState
        state.model_evaluations = ranked_evaluations
        state.gap_analyses = gap_records
        state.confusion_matrix_analyses = cm_analyses
        state.execution_time_analysis = time_analysis
        state.evaluation_report = evaluation_report
        state.final_summary_report = evaluation_report
        state.evaluation_status = "completed"
        state.workflow_status = WorkflowStatus.EVALUATION_COMPLETED

        best_model = evaluation_report.get("best_model")
        state.best_baseline_model = best_model
        state.best_model_name = best_model

        state.add_log(
            self.name,
            f"Diagnostic evaluation completed. Best baseline model: '{best_model}'.",
            details={
                "best_model": best_model,
                "selection_reason": evaluation_report.get("selection_reason"),
                "total_evaluated": len(ranked_evaluations),
            },
        )
        self.logger.info(f"Evaluation Agent completed. Top model: {best_model}")
        return state
