"""Model Selection Agent for autonomous candidate algorithm screening and registry configuration."""

from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent
from tools import model_tools
from utils.config import Config
from utils.state import AgentState, ProblemType, WorkflowStatus


class ModelSelectionAgent(BaseAgent):
    """Specialized agent responsible for screening and recommending classification models."""

    def __init__(self, name: str = "ModelSelectionAgent") -> None:
        super().__init__(name=name)

    def run(
        self,
        state: AgentState,
        candidate_models: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AgentState:
        """Screen and select candidate models based on task metadata and preprocessing state.

        Args:
            state: The shared workflow state.
            candidate_models: Optional explicit list of model keys to screen.

        Returns:
            Updated AgentState containing selected models, reasoning, and metadata.
        """
        # 1. Validate prerequisites
        if not state.target_column:
            err_msg = "Cannot select models: Target column not identified in state."
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        if state.problem_type not in Config.SUPPORTED_PROBLEM_TYPES:
            err_msg = (
                f"Cannot select models: Problem type '{state.problem_type}' is unsupported. "
                f"Supported types: {Config.SUPPORTED_PROBLEM_TYPES}"
            )
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        if not state.pipeline_config and not state.selected_numerical_features and not state.selected_categorical_features:
            err_msg = "Cannot select models: Preprocessing pipeline configuration missing from state."
            self.logger.error(err_msg)
            state.workflow_status = WorkflowStatus.FAILED
            state.error_message = err_msg
            state.add_log(self.name, err_msg, status="error")
            return state

        self.logger.info(f"Screening candidate models for problem type: {state.problem_type}")
        state.workflow_status = WorkflowStatus.SELECTING_MODELS
        state.add_log(self.name, f"Screening models for {state.problem_type}")

        # 2. Determine models to select
        all_supported = model_tools.get_supported_models(state.problem_type)
        if candidate_models:
            # Validate explicit overrides against registry
            selected_models: List[str] = []
            for m in candidate_models:
                valid, msg = model_tools.validate_model_configuration(m, state.problem_type)
                if valid:
                    selected_models.append(m.lower().strip())
                else:
                    self.logger.warning(f"Excluding invalid requested model '{m}': {msg}")
            if not selected_models:
                selected_models = all_supported
        else:
            selected_models = all_supported

        # 3. Formulate selection reasoning and metadata
        reasoning: Dict[str, str] = {}
        model_configs: Dict[str, Dict[str, Any]] = {}

        for m in selected_models:
            definition = model_tools.get_model_definition(m)
            scaling_req = definition["scaling_required"]
            scaling_note = "requires standard scaling" if scaling_req else "invariant to scaling"
            reasoning[m] = (
                f"Selected as baseline candidate for {state.problem_type}; {scaling_note}. "
                f"{definition['description']}"
            )
            model_configs[m] = {
                "display_name": definition["display_name"],
                "scaling_required": scaling_req,
                "default_params": definition["default_params"],
                "supported_problem_types": definition["supported_problem_types"],
                "description": definition["description"],
            }

        # 4. Update state
        state.selected_models = selected_models
        state.model_selection_reasoning = reasoning
        state.selected_model_configs = model_configs
        summary: Dict[str, Any] = {
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "total_candidates": len(selected_models),
            "selected_models": selected_models,
            "models_requiring_scaling": [
                m for m, cfg in model_configs.items() if cfg["scaling_required"]
            ],
            "tree_based_models": [
                m for m, cfg in model_configs.items() if not cfg["scaling_required"]
            ],
        }
        state.model_selection_summary = summary
        state.workflow_status = WorkflowStatus.MODELS_SELECTED
        state.add_log(
            self.name,
            f"Model selection complete: {len(selected_models)} candidate models selected.",
            details=summary,
        )
        self.logger.info(f"Model Selection Agent selected {len(selected_models)} models: {selected_models}")
        return state
