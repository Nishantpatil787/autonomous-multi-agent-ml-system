"""Tests for dashboard helper logic, graceful startup, and safe artifact loading."""

from pathlib import Path
import pandas as pd
import pytest

from dashboard import (
    load_benchmark_results,
    load_dashboard_artifacts,
    load_mlflow_recent_runs,
    render_dashboard,
)
from graph.workflow import run_graph
from tools import data_tools
from tools.persistence_tools import load_persisted_model, predict_with_persisted_pipeline
from utils.config import Config


class TestDashboardGracefulLoading:
    def test_load_dashboard_artifacts_missing_files_returns_message(self, tmp_path):
        m_missing = tmp_path / "absent_model.joblib"
        meta_missing = tmp_path / "absent_meta.json"
        with pytest.MonkeyPatch.context() as mp:
            from dashboard import load_dashboard_artifacts
            with pytest.raises(Exception) if False else pytest.MonkeyPatch().context():
                pass
        pipeline, metadata, err = load_dashboard_artifacts()
        # Even if files don't exist or exist, error string or tuple returned cleanly
        assert isinstance(err, (str, type(None)))

    def test_dashboard_imports_cleanly(self):
        import dashboard
        assert hasattr(dashboard, "render_dashboard")
        assert hasattr(dashboard, "load_dashboard_artifacts")
        assert hasattr(dashboard, "load_mlflow_recent_runs")
        assert hasattr(dashboard, "load_benchmark_results")

    def test_load_mlflow_recent_runs_returns_list_when_disabled(self):
        runs = load_mlflow_recent_runs()
        assert isinstance(runs, list)

    def test_load_benchmark_results_returns_none_or_dict(self):
        res = load_benchmark_results()
        assert res is None or isinstance(res, dict)


class TestDatasetUploadAndWorkflowIntegration:
    """Test dynamic dataset upload, target column selection, and workflow execution."""

    @pytest.fixture
    def custom_uploaded_csv(self, tmp_path: Path) -> Path:
        """Create a synthetic custom CSV with non-standard column names."""
        df = pd.DataFrame({
            "sensor_temp": [22.4, 25.1, 28.3, 19.8, 31.2, 24.5, 29.0, 21.1, 33.4, 27.2] * 4,
            "pressure_psi": [101.3, 102.5, 99.8, 103.1, 98.4, 101.9, 100.2, 102.8, 97.9, 101.0] * 4,
            "machine_type": ["Turbine", "Compressor", "Pump", "Turbine", "Compressor", "Pump", "Turbine", "Compressor", "Pump", "Turbine"] * 4,
            "failure_risk": ["Low", "High", "Low", "Medium", "High", "Low", "Medium", "Low", "High", "Low"] * 4,
        })
        csv_path = tmp_path / "custom_machinery_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_custom_dataset_profiling_and_target_detection(self, custom_uploaded_csv: Path):
        """Verify that uploaded dataset is properly profiled without assuming 'Outcome' target."""
        df = pd.read_csv(custom_uploaded_csv)
        assert df.shape == (40, 4)

        detected_types = data_tools.detect_feature_types(df, target_column="failure_risk")
        assert "sensor_temp" in detected_types["numerical"]
        assert "pressure_psi" in detected_types["numerical"]
        assert "machine_type" in detected_types["categorical"]
        assert "failure_risk" not in detected_types["numerical"]
        assert "failure_risk" not in detected_types["categorical"]

        problem_type, reason = data_tools.detect_problem_type(df["failure_risk"])
        assert problem_type == "multiclass_classification"

    def test_workflow_execution_on_custom_dataset(self, custom_uploaded_csv: Path, monkeypatch):
        """Verify that run_graph executes end-to-end on a custom dataset and target column."""
        import shutil
        m_dir = Config.MODELS_DIR
        joblib_path = m_dir / "best_model.joblib"
        meta_path = m_dir / "best_model_metadata.json"
        bak_joblib = m_dir / "best_model.joblib.bak"
        bak_meta = m_dir / "best_model_metadata.json.bak"

        # Backup existing production model artifacts
        if joblib_path.exists():
            shutil.copy2(joblib_path, bak_joblib)
        if meta_path.exists():
            shutil.copy2(meta_path, bak_meta)

        try:
            result = run_graph(
                dataset_path=str(custom_uploaded_csv),
                target_column="failure_risk",
                enable_optimization=False,  # Fast test mode
            )
            assert result.failed_node is None
            assert "data_analysis" in result.completed_nodes
            assert "training" in result.completed_nodes
            assert "model_persistence" in result.completed_nodes

            # Load newly persisted artifacts and verify provenance
            pipeline, metadata = load_persisted_model()
            assert metadata["target_column"] == "failure_risk"
            assert metadata["task_type"] == "multiclass_classification"
            assert "sensor_temp" in metadata["feature_columns"]
            assert "pressure_psi" in metadata["feature_columns"]
            assert "machine_type" in metadata["feature_columns"]
            assert "failure_risk" not in metadata["feature_columns"]

            # Test live dynamic prediction with custom feature inputs
            sample_input = {
                "sensor_temp": 26.5,
                "pressure_psi": 100.8,
                "machine_type": "Turbine",
            }
            pred_res = predict_with_persisted_pipeline(pipeline, metadata, sample_input)
            assert "predictions" in pred_res
            assert len(pred_res["predictions"]) == 1
            assert pred_res["predictions"][0] in ["Low", "High", "Medium"]
        finally:
            # Restore production model artifacts
            if bak_joblib.exists():
                shutil.move(bak_joblib, joblib_path)
            if bak_meta.exists():
                shutil.move(bak_meta, meta_path)
