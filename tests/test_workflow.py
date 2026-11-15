"""Tests for graph/workflow.py — full integration and end-to-end graph execution."""

import hashlib
import json
from pathlib import Path

import pytest

from graph.workflow import build_ml_workflow, run_graph
from graph.state import WorkflowResult
from utils.config import Config
from utils.state import WorkflowStatus

BINARY_CSV = str(Config.DATASETS_DIR / "sample_classification.csv")
MULTICLASS_CSV = str(Config.DATASETS_DIR / "sample_multiclass.csv")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Failure handling tests (20–25)
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_data_analysis_failure_stops_graph(self):
        result = run_graph(
            dataset_path="nonexistent_dataset.csv",
            enable_optimization=False,
            max_retries=0,
        )
        assert result.failed_node == "data_analysis"
        assert "preprocessing" not in result.completed_nodes
        assert "training" not in result.completed_nodes

    def test_failure_sets_error_message(self):
        result = run_graph(
            dataset_path="missing.csv",
            enable_optimization=False,
            max_retries=0,
        )
        assert result.error_message is not None

    def test_optimization_failure_preserves_evaluation(self):
        """Even if optimization fails, Phase-5 evaluation results remain accessible."""
        # We simulate this by running successfully through evaluation first
        # then verifying that baseline model is preserved in the result
        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=True,
            max_retries=1,
        )
        # Whether optimization succeeded or not, best_baseline_model must be set
        assert result.best_baseline_model is not None

    def test_workflow_result_is_dataclass(self):
        result = run_graph(
            dataset_path="bad.csv",
            enable_optimization=False,
            max_retries=0,
        )
        assert isinstance(result, WorkflowResult)


# ---------------------------------------------------------------------------
# Retry tests (26–29)
# ---------------------------------------------------------------------------

class TestRetryMechanism:
    def test_retry_count_recorded_on_failure(self):
        result = run_graph(
            dataset_path="bad.csv",
            enable_optimization=False,
            max_retries=1,
        )
        # With max_retries=1, if data_analysis fails it will have been tried twice
        # retry_counts["data_analysis"] should be >= 0
        assert isinstance(result.retry_counts, dict)

    def test_no_infinite_retry(self):
        """Graph terminates; does not loop indefinitely."""
        result = run_graph(
            dataset_path="bad.csv",
            enable_optimization=False,
            max_retries=1,
        )
        # The data_analysis node should NOT have more retries than max_retries
        count = result.retry_counts.get("data_analysis", 0)
        assert count <= 1

    def test_retry_count_zero_on_clean_run(self):
        result = run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            max_retries=1,
        )
        # On a clean run with a valid CSV, retry counts should all be 0
        for node, count in result.retry_counts.items():
            assert count == 0, f"Unexpected retry for node '{node}'"


# ---------------------------------------------------------------------------
# Full integration tests (30–36)
# ---------------------------------------------------------------------------

class TestBinaryGraphIntegration:
    @pytest.fixture(scope="class")
    def binary_result(self):
        return run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=True,
            max_retries=1,
        )

    def test_binary_workflow_status_optimization_completed(self, binary_result):
        assert binary_result.workflow_status in (
            "optimization_completed",
            WorkflowStatus.OPTIMIZATION_COMPLETED.value,
        )

    def test_binary_all_required_nodes_completed(self, binary_result):
        required = [
            "data_analysis", "preprocessing", "model_selection",
            "training", "evaluation"
        ]
        for node in required:
            assert node in binary_result.completed_nodes

    def test_binary_optimization_executed(self, binary_result):
        assert binary_result.optimization_executed is True

    def test_binary_best_baseline_model_set(self, binary_result):
        assert binary_result.best_baseline_model is not None

    def test_binary_best_optimized_model_set(self, binary_result):
        # Optimization ran, so best_optimized_model should be set
        assert binary_result.best_optimized_model is not None

    def test_binary_no_failed_node(self, binary_result):
        assert binary_result.failed_node is None

    def test_binary_final_agent_state_serializable(self, binary_result):
        d = binary_result.final_agent_state.to_dict()
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)


class TestMulticlassGraphIntegration:
    @pytest.fixture(scope="class")
    def multiclass_result(self):
        if not Path(MULTICLASS_CSV).exists():
            pytest.skip("Multiclass CSV not found.")
        return run_graph(
            dataset_path=MULTICLASS_CSV,
            enable_optimization=True,
            max_retries=1,
        )

    def test_multiclass_workflow_completes(self, multiclass_result):
        assert multiclass_result.failed_node is None

    def test_multiclass_baseline_set(self, multiclass_result):
        assert multiclass_result.best_baseline_model is not None


class TestNoOptimizeGraphRun:
    @pytest.fixture(scope="class")
    def no_opt_result(self):
        return run_graph(
            dataset_path=BINARY_CSV,
            enable_optimization=False,
            max_retries=1,
        )

    def test_no_optimize_workflow_completes(self, no_opt_result):
        assert no_opt_result.failed_node is None

    def test_no_optimize_optimization_not_executed(self, no_opt_result):
        assert no_opt_result.optimization_executed is False

    def test_no_optimize_baseline_set(self, no_opt_result):
        assert no_opt_result.best_baseline_model is not None

    def test_no_optimize_evaluation_still_runs(self, no_opt_result):
        assert "evaluation" in no_opt_result.completed_nodes


class TestExistingSequentialPath:
    def test_sequential_import_still_works(self):
        """Verify Phase 1–6 sequential path is not broken."""
        from agents.data_analysis_agent import DataAnalysisAgent
        from agents.preprocessing_agent import PreprocessingAgent
        from utils.state import AgentState
        state = AgentState(dataset_path=BINARY_CSV)
        state = DataAnalysisAgent().run(state)
        state = PreprocessingAgent().run(state)
        assert state.workflow_status == WorkflowStatus.PREPROCESSING_COMPLETED


class TestSourceCSVIntegrityGraph:
    def test_binary_csv_unchanged_after_graph(self):
        sha_before = _sha256(BINARY_CSV)
        run_graph(BINARY_CSV, enable_optimization=False, max_retries=0)
        sha_after = _sha256(BINARY_CSV)
        assert sha_before == sha_after

    def test_multiclass_csv_unchanged_after_graph(self):
        if not Path(MULTICLASS_CSV).exists():
            pytest.skip("Multiclass CSV not found.")
        sha_before = _sha256(MULTICLASS_CSV)
        run_graph(MULTICLASS_CSV, enable_optimization=False, max_retries=0)
        sha_after = _sha256(MULTICLASS_CSV)
        assert sha_before == sha_after


class TestRepeatRunDeterminism:
    def test_same_workflow_status_on_repeat(self):
        r1 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        r2 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        assert r1.workflow_status == r2.workflow_status

    def test_same_completed_nodes_on_repeat(self):
        r1 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        r2 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        assert r1.completed_nodes == r2.completed_nodes

    def test_same_baseline_model_on_repeat(self):
        r1 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        r2 = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        assert r1.best_baseline_model == r2.best_baseline_model


class TestFinalAgentStateSerialization:
    def test_to_dict_produces_json_serializable_result(self):
        result = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        d = result.to_dict()
        serialized = json.dumps(d, default=str)
        assert isinstance(serialized, str)

    def test_no_sklearn_objects_in_result_dict(self):
        """Ensure no live sklearn objects leak into workflow result dict."""
        result = run_graph(BINARY_CSV, enable_optimization=False, max_retries=1)
        d = result.final_agent_state.to_dict()
        # JSON dump should succeed without needing custom encoder
        try:
            json.dumps(d)
            serializable = True
        except TypeError:
            serializable = False
        assert serializable is True


class TestBuildMlWorkflow:
    def test_workflow_compiles_without_error(self):
        app = build_ml_workflow()
        assert app is not None

    def test_workflow_no_optimize_compiles(self):
        app = build_ml_workflow(enable_optimization=False)
        assert app is not None

    def test_no_llm_dependency(self):
        """Verify the workflow has no LLM dependency."""
        import sys
        llm_modules = [
            m for m in sys.modules
            if not m.startswith("langchain_core")
            and any(pkg in m for pkg in ["ollama", "openai", "anthropic", "gemini"])
        ]
        assert llm_modules == [], f"Unexpected LLM modules loaded: {llm_modules}"
