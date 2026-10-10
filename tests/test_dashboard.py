"""Tests for the dashboard layer (Phase 9, checks 11-15).

The dashboard is split in two on purpose, and most of these tests exist to keep
that split honest:

* ``tools/dashboard_tools.py`` holds every piece of logic and imports neither
  Streamlit nor an agent, so it can be tested directly and re-used without a
  browser;
* ``dashboard.py`` is the Streamlit presentation layer and holds no logic.

What is checked here: that the logic layer loads a persisted artifact without
training anything, that it reports the "no artifact" situation instead of
failing, that every one of the ten sections produces usable data for both a
binary and a multiclass run, that a prediction goes through the *saved* pipeline
rather than through a second copy of the preprocessing, and that no code path
anywhere offers to load a caller-supplied model file.

The last class of tests renders ``dashboard.py`` for real with Streamlit's own
``AppTest``. Every other test here exercises the logic layer, which cannot catch
a mistake in a Streamlit call - a keyword that does not exist in the installed
version only fails at render time, in a browser. Those tests exist so that a
``TypeError`` in the view is a red test rather than a blank page a user finds.
"""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pandas as pd
from sklearn.pipeline import Pipeline

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tests.helpers import BINARY_DATASET, MULTICLASS_DATASET, file_hash
from tools import dashboard_tools as helpers
from tools.persistence_tools import artifact_paths, load_persisted_model
from utils import config
from utils.state import AgentState

#: Hashes of the committed fixtures. The dashboard must never write to a dataset.
EXPECTED_BINARY_HASH = "49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c"
EXPECTED_MULTICLASS_HASH = (
    "0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8"
)

#: Root of the project, used by the source-inspection tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The module holding the presentation layer.
DASHBOARD_MODULE = PROJECT_ROOT / "dashboard.py"


def trained_state(dataset: Path) -> AgentState:
    """Run the real pipeline and persist it into a temporary directory.

    Optimization is deliberately left off: the dashboard only reads the recorded
    results, so the point of these tests is that a *saved* artifact is enough for
    every section to have something to show.

    Args:
        dataset: The CSV to run on.

    Returns:
        The directory the artifacts were written to.
    """
    out = Path(tempfile.mkdtemp())
    state = DataAnalysisAgent().run(AgentState(dataset_path=str(dataset)))
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    state = EvaluationAgent().run(state)
    PersistenceAgent().run(state, output_dir=out)
    return out


def optimized_artifact_dir() -> Path:
    """Run the full workflow, search included, and return the artifact folder.

    This is the slower path: two models go through 20 Optuna trials each. Only
    the sections that differ between the two selection sources need it, and it
    runs once for the whole class.

    Returns:
        The directory the artifacts were written to.
    """
    from graph.workflow import run_workflow

    out = Path(tempfile.mkdtemp())
    result = run_workflow(
        dataset_path=str(BINARY_DATASET),
        enable_optimization=True,
        persistence_output_dir=str(out),
        enable_tracking=False,
    )
    if result.final_agent_state.persistence_status != "completed":
        shutil.rmtree(out, ignore_errors=True)
        raise AssertionError("the fixture run did not persist an artifact")
    return out


# --------------------------------------------------------------------------- #
# 11. Module boundaries
# --------------------------------------------------------------------------- #
class TestModuleBoundaries(unittest.TestCase):
    """Check 11: the logic layer is separable, and stays that way."""

    def _imports(self, path: Path) -> List[str]:
        """Return the module names a file imports.

        Args:
            path: The Python file to inspect.

        Returns:
            Every imported module or package name.
        """
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names

    def test_the_helper_module_imports_no_presentation_library(self):
        """Streamlit needs a server; the logic must not need one."""
        names = self._imports(Path(helpers.__file__))
        for banned in ("streamlit", "matplotlib", "plotly", "seaborn", "altair"):
            with self.subTest(module=banned):
                self.assertFalse(
                    any(name == banned or name.startswith(banned) for name in names)
                )

    def test_the_helper_module_imports_no_agent_and_no_graph(self):
        """The dashboard reads artifacts; it must not be able to start a run."""
        names = self._imports(Path(helpers.__file__))
        for banned in ("agents", "graph", "llm"):
            with self.subTest(module=banned):
                self.assertFalse(
                    any(name == banned or name.startswith(banned + ".") for name in names)
                )

    def test_the_helper_module_does_import_the_persistence_layer(self):
        names = self._imports(Path(helpers.__file__))
        self.assertTrue(
            any(
                name == "tools.persistence_tools" or name == "persistence_tools"
                for name in names
            )
        )

    def test_the_view_module_does_not_import_the_helpers_by_agents(self):
        """It may use the helper layer, but must not reach past it into agents."""
        names = self._imports(DASHBOARD_MODULE)
        for banned in ("agents", "graph"):
            with self.subTest(module=banned):
                self.assertFalse(
                    any(name == banned or name.startswith(banned + ".") for name in names)
                )

    def test_the_view_module_never_calls_joblib_load(self):
        """The view must go through the trusted loader, not unpickle anything."""
        tree = ast.parse(DASHBOARD_MODULE.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and "joblib" in ast.dump(node.func):
                self.fail(f"dashboard.py calls {ast.dump(node.func)} at line {node.lineno}")

    def test_the_view_module_offers_no_file_uploader(self):
        """An uploaded pickle executes arbitrary code; there must be no door."""
        tree = ast.parse(DASHBOARD_MODULE.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "file_uploader",
                "upload_file",
                "unpickle",
            }:
                self.fail(
                    f"dashboard.py uses {node.attr} at line {node.lineno}"
                )

    def test_the_view_module_renders_only_under_the_streamlit_entry_point(self):
        """``import dashboard`` must not build the page as a side effect.

        Streamlit executes the file with ``__name__`` set to ``"__main__"``, so
        this guard is what lets the test suite import the module at all.
        """
        tree = ast.parse(DASHBOARD_MODULE.read_text(encoding="utf-8-sig"))
        calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(
            calls, [], "top-level statements run on import, so the page builds eagerly"
        )
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
        ]
        self.assertTrue(
            any(
                isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                for node in guards
            ),
            "dashboard.py has no `if __name__ == '__main__':` guard",
        )


# --------------------------------------------------------------------------- #
# 12. Loading: with an artifact, and without one
# --------------------------------------------------------------------------- #
class TestLoadingWithoutAnArtifact(unittest.TestCase):
    """Check 12a: the dashboard must start before anything has been persisted."""

    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        self.state = helpers.load_dashboard_state(self.out)

    def test_loading_an_empty_directory_does_not_raise(self):
        self.assertIsNotNone(self.state)

    def test_the_state_reports_the_artifact_as_unavailable(self):
        self.assertFalse(self.state.available)

    def test_an_absent_artifact_is_not_reported_as_an_error(self):
        """Nothing on disk is a normal starting state, not a failure."""
        self.assertEqual(self.state.problem, "")
        self.assertIsNone(self.state.model)
        self.assertEqual(self.state.metadata, {})
        self.assertFalse(self.state.results_available)

    def test_the_prediction_section_states_that_nothing_is_persisted(self):
        """The exact wording the acceptance criterion asks for."""
        section = helpers.build_prediction_section(self.state)
        self.assertFalse(section["available"])
        self.assertEqual(section["note"], "No persisted model is available yet.")

    def test_every_section_says_it_has_nothing_to_show(self):
        for name in (
            "build_dataset_section",
            "build_evaluation_section",
            "build_comparison_section",
            "build_confusion_section",
            "build_optimization_section",
        ):
            with self.subTest(section=name):
                section = getattr(helpers, name)(self.state)
                self.assertFalse(section["available"])

    def test_a_section_with_no_data_carries_no_chart_rows(self):
        """An empty chart would render as a blank panel with no explanation."""
        comparison = helpers.build_comparison_section(self.state)
        self.assertEqual(comparison["rows"], [])
        self.assertEqual(comparison["f1_chart"], [])

    def test_the_list_sections_are_empty(self):
        self.assertEqual(helpers.build_status_section(self.state), [])
        self.assertEqual(helpers.build_model_information_section(self.state), [])

    def test_the_llm_section_explains_itself_without_ollama(self):
        """No stored interpretation is still an answer, not a blank panel."""
        section = helpers.build_llm_section(self.state)
        self.assertEqual(section["entries"], [])
        self.assertTrue(section["statement"])
        self.assertIn("optional", section["note"])

    def test_the_prediction_form_has_no_fields_to_fill_in(self):
        self.assertEqual(helpers.build_prediction_fields(self.state), [])

    def test_a_prediction_without_an_artifact_is_refused_not_raised(self):
        outcome = helpers.run_prediction(self.state, [], {})
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.message)

    def test_a_damaged_artifact_is_reported_distinctly_from_an_absent_one(self):
        """Two different problems deserve two different messages."""
        paths = artifact_paths(self.out)
        paths.model.write_text("not a pickle", encoding="utf-8")
        paths.metadata.write_text(
            json.dumps({"schema_version": 1}), encoding="utf-8"
        )
        damaged = helpers.load_dashboard_state(self.out)
        self.assertFalse(damaged.available)
        self.assertTrue(damaged.problem)
        self.assertNotEqual(damaged.problem, self.state.problem)


class TestLoadingWithAnArtifact(unittest.TestCase):
    """Check 12b: the same code path, with artifacts on disk."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = trained_state(BINARY_DATASET)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def setUp(self) -> None:
        self.state = helpers.load_dashboard_state(self.out)

    def test_the_state_is_available(self):
        self.assertTrue(self.state.available)
        self.assertEqual(self.state.problem, "")

    def test_the_pipeline_is_the_saved_object(self):
        self.assertIsNotNone(self.state.model)
        self.assertEqual(
            list(self.state.model.pipeline.named_steps), ["preprocessor", "model"]
        )

    def test_the_metadata_travels_with_it(self):
        self.assertEqual(self.state.metadata["model_name"], self.state.model_name)
        self.assertEqual(self.state.metadata["target_column"], "Outcome")

    def test_the_recorded_results_are_loaded_verbatim(self):
        """Read, not recomputed: the counts come from the saved document."""
        self.assertEqual(self.state.results["dataset"]["rows"], 123)
        self.assertEqual(
            self.state.results["dataset"]["target_column"], "Outcome"
        )

    def test_loading_does_not_refit_anything(self):
        """The decisive check: no ``fit`` runs while an artifact is loaded.

        Counting calls is used rather than comparing estimator state, because a
        refit on the same data with the same seed would produce a byte-identical
        estimator and the comparison would prove nothing. A call to ``fit`` is
        the only unambiguous evidence of training.
        """
        from sklearn.pipeline import Pipeline
        from unittest import mock

        original = Pipeline.fit
        with mock.patch.object(
            Pipeline, "fit", side_effect=AssertionError("the dashboard fitted")
        ) as fit:
            state = helpers.load_dashboard_state(self.out)
        fit.assert_not_called()
        self.assertTrue(state.available)
        self.assertIs(Pipeline.fit, original)

    def test_predicting_does_not_refit_anything(self):
        """A rerun of the prediction form must be as cheap as the first one."""
        from sklearn.pipeline import Pipeline
        from unittest import mock

        state = helpers.load_dashboard_state(self.out)
        fields = helpers.build_prediction_fields(state)
        values = {f.name: f.default for f in fields}
        with mock.patch.object(
            Pipeline, "fit", side_effect=AssertionError("the dashboard fitted")
        ) as fit:
            outcome = helpers.run_prediction(state, fields, values)
        fit.assert_not_called()
        self.assertTrue(outcome.ok, outcome.message)

    def test_the_saved_estimator_is_already_fitted(self):
        """If it were not, the first prediction would be meaningless."""
        state = helpers.load_dashboard_state(self.out)
        estimator = state.model.pipeline.named_steps["model"]
        self.assertTrue(
            hasattr(estimator, "coef_") or hasattr(estimator, "n_features_in_")
        )


# --------------------------------------------------------------------------- #
# 13. The ten sections
# --------------------------------------------------------------------------- #
class TestSectionsWithAnArtifact(unittest.TestCase):
    """Check 13: each section has real content for a persisted run.

    The artifact is produced with optimization *enabled* on the binary dataset,
    which is the case that exercises the most code: an optimized selection, a
    skipped precision and recall, and no confusion matrix.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = optimized_artifact_dir()
        cls.state = helpers.load_dashboard_state(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_status_section_reports_everything_the_header_promises(self):
        labels = {row["label"] for row in helpers.build_status_section(self.state)}
        for expected in (
            "Persisted model available",
            "Selected model",
            "Selection source",
            "Task type",
            "Target column",
            "Optimization",
            "LLM interpretation",
        ):
            with self.subTest(label=expected):
                self.assertIn(expected, labels)

    def test_the_status_section_does_not_claim_a_language_model_was_used(self):
        """Ollama is not installed here, so the dashboard must not imply it was."""
        rows = helpers.build_status_section(self.state)
        value = next(r["value"] for r in rows if r["label"] == "LLM interpretation")
        self.assertIn("not", value.lower())

    def test_the_status_section_names_the_optimized_selection(self):
        rows = helpers.build_status_section(self.state)
        source = next(r["value"] for r in rows if r["label"] == "Selection source")
        self.assertIn("optimized", source)

    def test_the_dataset_section_reads_the_phase_two_record(self):
        section = helpers.build_dataset_section(self.state)
        self.assertTrue(section["available"])
        values = {row["label"]: row["value"] for row in section["metrics"]}
        self.assertEqual(values["Rows"], "123")
        self.assertEqual(values["Target column"], "Outcome")
        self.assertEqual(values["Missing values"], "17")
        self.assertEqual(values["Duplicate rows"], "3")

    def test_the_dataset_section_shows_the_issue_breakdown_as_text(self):
        """A raw dict repr in a table cell would be unreadable."""
        section = helpers.build_dataset_section(self.state)
        values = {row["label"]: row["value"] for row in section["metrics"]}
        value = values["Data quality issues"]
        self.assertNotIn("{", value)
        self.assertNotIn("'", value)
        self.assertIn("warning", value)

    def test_the_dataset_section_splits_features_by_kind(self):
        section = helpers.build_dataset_section(self.state)
        self.assertTrue(section["numerical_features"])
        self.assertTrue(section["categorical_features"])
        # Phase 2 counts every non-target column, including the record id. The
        # dashboard shows the exclusion and the reason rather than quietly
        # dropping the column.
        excluded = {item["column"]: item["reason"] for item in section["excluded_features"]}
        self.assertIn("record_id", excluded)
        self.assertEqual(excluded["record_id"], "identifier_like")

    def test_the_dataset_section_lists_more_features_than_the_model_uses(self):
        """The difference is the identifier, and both numbers are shown."""
        section = helpers.build_dataset_section(self.state)
        dataset_total = len(section["numerical_features"]) + len(
            section["categorical_features"]
        )
        self.assertEqual(dataset_total, 8)
        self.assertEqual(len(self.state.feature_columns()), 7)

    def test_the_dataset_section_shows_the_class_distribution(self):
        section = helpers.build_dataset_section(self.state)
        self.assertEqual(len(section["class_distribution"]), 2)
        total = sum(row["count"] for row in section["class_distribution"])
        self.assertEqual(total, 123)

    def test_the_comparison_table_keeps_baseline_and_optimized_apart(self):
        section = helpers.build_comparison_section(self.state)
        kinds = {row["Kind"] for row in section["rows"]}
        self.assertEqual(kinds, {"BASELINE", "OPTIMIZED"})

    def test_the_comparison_table_marks_exactly_one_persisted_row(self):
        """Support Vector Machine appears twice; only the tuned one was saved."""
        section = helpers.build_comparison_section(self.state)
        marked = [row for row in section["rows"] if row["Persisted"] == "yes"]
        self.assertEqual(len(marked), 1)
        self.assertEqual(marked[0]["Kind"], "OPTIMIZED")
        self.assertEqual(
            section["persisted_label"],
            f"{marked[0]['Model']} (OPTIMIZED)",
        )

    def test_the_comparison_chart_groups_the_two_rows_of_one_model(self):
        """Same name twice on one axis needs a grouping column to stay readable."""
        section = helpers.build_comparison_section(self.state)
        names = [row["Model"] for row in section["f1_chart"]]
        self.assertGreater(len(names), len(set(names)))
        for row in section["f1_chart"]:
            with self.subTest(model=row["Model"]):
                self.assertIn(row["Kind"], ("Baseline (Phase 4)", "Optimized (Phase 6)"))
                self.assertIsInstance(row["F1"], float)

    def test_the_evaluation_section_reports_the_selected_model_and_metrics(self):
        section = helpers.build_evaluation_section(self.state)
        self.assertTrue(section["available"])
        self.assertEqual(section["model_name"], self.state.model_name)
        self.assertTrue(section["metrics"])
        self.assertTrue(section["interpretation"])

    def test_the_evaluation_section_says_where_the_metrics_came_from(self):
        """An optimized score is a Phase 6 score and must not claim to be Phase 4."""
        section = helpers.build_evaluation_section(self.state)
        self.assertEqual(section["metrics_source"], "optimization_holdout")

    def test_the_evaluation_section_shows_a_dash_for_unrecorded_metrics(self):
        section = helpers.build_evaluation_section(self.state)
        values = {row["label"]: row["value"] for row in section["metrics"]}
        self.assertEqual(values["Precision"], helpers.NOT_RECORDED)
        self.assertEqual(values["Recall"], helpers.NOT_RECORDED)
        self.assertNotEqual(values["F1"], helpers.NOT_RECORDED)

    def test_the_evaluation_section_reads_the_generalisation_gap(self):
        section = helpers.build_evaluation_section(self.state)
        gap = section["generalization_gap"]
        self.assertIn("Hold-out F1", gap["text"])
        self.assertIn("cross-validated F1", gap["text"])
        self.assertTrue(section["gap_disclaimer"])

    def test_the_confusion_section_explains_its_absence_for_a_tuned_model(self):
        """It must be clear rather than blank, and must never re-score."""
        section = helpers.build_confusion_section(self.state)
        self.assertFalse(section["available"])
        self.assertNotIn("frame", section)
        self.assertIn("Phase 6", section["note"])

    def test_the_optimization_section_reports_the_search_that_ran(self):
        section = helpers.build_optimization_section(self.state)
        self.assertTrue(section["performed"])
        self.assertEqual(section["models_searched"], 2)
        self.assertEqual(section["total_trials"], section["trials_per_model"] * 2)
        self.assertEqual(section["winner_trials"], section["trials_per_model"])

    def test_the_optimization_section_reports_the_improvement(self):
        section = helpers.build_optimization_section(self.state)
        self.assertIn("improved", str(section["improvement_text"]).lower())
        self.assertTrue(section["best_params_text"])

    def test_the_llm_section_is_readable_without_ollama(self):
        section = helpers.build_llm_section(self.state)
        self.assertEqual(section["entries"], [])
        self.assertTrue(section["statement"])
        self.assertTrue(section["note"])

    def test_the_model_information_section_is_a_label_value_list(self):
        rows = helpers.build_model_information_section(self.state)
        self.assertIsInstance(rows, list)
        labels = {row["label"] for row in rows}
        for expected in (
            "Artifact",
            "Metadata file",
            "Model type",
            "Selection source",
            "Feature count",
            "Target column",
            "Artifact size",
        ):
            with self.subTest(label=expected):
                self.assertIn(expected, labels)

    def test_the_model_information_section_shows_no_absolute_path(self):
        """An absolute path would leak the user's home directory on screen."""
        rows = helpers.build_model_information_section(self.state)
        value = next(r["value"] for r in rows if r["label"] == "Artifact directory")
        self.assertNotIn(":", value)
        self.assertNotIn("\\", value)
        self.assertNotIn(str(Path.home()), value)

    def test_the_project_artifact_directory_renders_as_models(self):
        """The real location is short and portable when it is inside the project."""
        from utils import config

        self.assertEqual(
            helpers._relative_to_project(config.MODELS_DIR), "models"
        )

    def test_a_directory_outside_the_project_falls_back_to_its_name(self):
        outside = Path(tempfile.gettempdir()) / "some_other_place"
        self.assertEqual(helpers._relative_to_project(outside), "some_other_place")

    def test_the_model_information_section_holds_no_secret(self):
        text = json.dumps(helpers.build_model_information_section(self.state)).lower()
        for word in ("api_key", "password", "secret", "token"):
            with self.subTest(word=word):
                self.assertNotIn(word, text)


class TestBaselineArtifactSections(unittest.TestCase):
    """Check 13b: the other selection source, where Phase 4 measured everything."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = trained_state(BINARY_DATASET)
        cls.state = helpers.load_dashboard_state(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_selection_source_is_the_baseline(self):
        rows = helpers.build_status_section(self.state)
        source = next(r["value"] for r in rows if r["label"] == "Selection source")
        self.assertIn("baseline", source)

    def test_the_metrics_come_from_the_evaluation_phase(self):
        section = helpers.build_evaluation_section(self.state)
        self.assertEqual(section["metrics_source"], "evaluation_holdout")
        values = {row["label"]: row["value"] for row in section["metrics"]}
        for label in ("Accuracy", "Precision", "Recall", "F1"):
            with self.subTest(metric=label):
                self.assertNotEqual(values[label], helpers.NOT_RECORDED)

    def test_the_confusion_matrix_is_stored_and_summarised(self):
        section = helpers.build_confusion_section(self.state)
        self.assertTrue(section["available"])
        self.assertEqual(section["labels"], ["0", "1"])
        self.assertEqual(section["total"], 24)
        self.assertEqual(section["correct"] + section["incorrect"], section["total"])

    def test_the_chart_frame_is_tidy_for_streamlit(self):
        """``st.bar_chart`` needs long form: one row per matrix cell."""
        section = helpers.build_confusion_section(self.state)
        frame = section["frame"]
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 4)
        self.assertEqual(
            list(frame.columns), ["True label", "Predicted label", "Count"]
        )
        self.assertEqual(int(frame["Count"].sum()), section["total"])

    def test_the_optimization_section_says_the_search_did_not_run(self):
        section = helpers.build_optimization_section(self.state)
        self.assertFalse(section["available"])
        self.assertIn("not performed", str(section["note"]).lower())
        # Nothing about a search that did not happen is reported as if it had.
        self.assertNotIn("best_params", section)
        self.assertNotIn("improvement_text", section)

    def test_the_comparison_table_has_no_optimized_rows(self):
        section = helpers.build_comparison_section(self.state)
        self.assertEqual({row["Kind"] for row in section["rows"]}, {"BASELINE"})


# --------------------------------------------------------------------------- #
# 14. Prediction, through the saved pipeline
# --------------------------------------------------------------------------- #
class TestPrediction(unittest.TestCase):
    """Check 14: the form, the frame and the prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        # The baseline artifact is used here on purpose: the Logistic Regression
        # it selects exposes ``predict_proba``, so the probability branch is
        # exercised too. The tuned-SVM case is covered separately.
        cls.out = trained_state(BINARY_DATASET)
        cls.state = helpers.load_dashboard_state(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def _defaults(self) -> Dict[str, Any]:
        """Return one value per field, each at its recorded default."""
        return {f.name: f.default for f in helpers.build_prediction_fields(self.state)}

    def test_the_form_has_one_field_per_feature(self):
        fields = helpers.build_prediction_fields(self.state)
        self.assertEqual([field.name for field in fields], self.state.feature_columns())
        self.assertEqual(len(fields), 7)

    def test_numeric_and_categorical_fields_are_told_apart(self):
        fields = helpers.build_prediction_fields(self.state)
        kinds = {field.kind for field in fields}
        self.assertEqual(kinds, {"number", "text"})

    def test_a_categorical_field_lists_the_options_the_data_has(self):
        """A select box needs its choices recorded, not guessed."""
        fields = helpers.build_prediction_fields(self.state)
        text_fields = [f for f in fields if f.kind == "text"]
        self.assertTrue(text_fields)
        for field in text_fields:
            with self.subTest(field=field.name):
                self.assertTrue(field.options)
                self.assertIn(field.default, field.options)

    def test_the_defaults_come_from_the_dataset_not_from_hardcoded_values(self):
        """The saved document records the mean; the field must show that mean."""
        expected = self.state.results["dataset"]["numerical_statistics"]
        for field in helpers.build_prediction_fields(self.state):
            if field.kind != "number":
                continue
            with self.subTest(field=field.name):
                self.assertIn(field.name, expected)
                self.assertAlmostEqual(
                    float(field.default), float(expected[field.name]["mean"]), places=4
                )

    def test_the_frame_has_the_columns_the_saved_pipeline_expects(self):
        fields = helpers.build_prediction_fields(self.state)
        frame = helpers.build_prediction_frame(
            fields, {field.name: field.default for field in fields}
        )
        self.assertEqual(list(frame.columns), self.state.feature_columns())
        self.assertEqual(len(frame), 1)

    def test_the_frame_keeps_a_missing_value_missing(self):
        """An empty input box must become ``NaN``, not the string ``""``.

        The saved preprocessor imputes a real missing value; it cannot impute a
        blank string, so this is where a naive form would break.
        """
        fields = helpers.build_prediction_fields(self.state)
        values = self._defaults()
        numeric = next(field for field in fields if field.kind == "number")
        values[numeric.name] = None
        frame = helpers.build_prediction_frame(fields, values)
        self.assertTrue(pd.isna(frame[numeric.name].iloc[0]))

    def test_a_prediction_returns_a_class_from_the_saved_labels(self):
        fields = helpers.build_prediction_fields(self.state)
        outcome = helpers.run_prediction(self.state, fields, self._defaults())
        self.assertTrue(outcome.ok, outcome.message)
        self.assertIn(str(outcome.predicted_class), self.state.class_labels)

    def test_a_prediction_goes_through_the_saved_pipeline(self):
        """The frame the form builds must predict exactly as the pipeline does.

        If the view re-implemented imputation, scaling or encoding, its answer
        would differ from the saved object's for at least some inputs.
        """
        fields = helpers.build_prediction_fields(self.state)
        values = self._defaults()
        frame = helpers.build_prediction_frame(fields, values)
        direct = self.state.model.pipeline.predict(frame)[0]
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertEqual(str(outcome.predicted_class), str(direct))

    def test_probabilities_are_labelled_by_class(self):
        """A bare number is ambiguous in a three-class task."""
        fields = helpers.build_prediction_fields(self.state)
        outcome = helpers.run_prediction(self.state, fields, self._defaults())
        self.assertIsNotNone(outcome.probabilities)
        self.assertEqual(
            [row["class"] for row in outcome.probabilities], self.state.class_labels
        )

    def test_probabilities_are_a_probability_distribution(self):
        fields = helpers.build_prediction_fields(self.state)
        outcome = helpers.run_prediction(self.state, fields, self._defaults())
        total = sum(row["probability"] for row in outcome.probabilities)
        self.assertAlmostEqual(total, 1.0, places=6)
        for row in outcome.probabilities:
            with self.subTest(probability=row):
                self.assertGreaterEqual(row["probability"], 0.0)
                self.assertLessEqual(row["probability"], 1.0)

    def test_the_confidence_is_the_highest_probability(self):
        fields = helpers.build_prediction_fields(self.state)
        outcome = helpers.run_prediction(self.state, fields, self._defaults())
        best = max(row["probability"] for row in outcome.probabilities)
        self.assertAlmostEqual(outcome.confidence, best)

    def test_the_section_reports_whether_probabilities_exist(self):
        """The form must not promise a chart the model cannot produce."""
        section = helpers.build_prediction_section(self.state)
        self.assertTrue(section["available"])
        self.assertEqual(
            section["supports_probability"], self.state.model.supports_probability
        )
        self.assertEqual(section["class_labels"], self.state.class_labels)
        self.assertEqual(section["target_column"], "Outcome")

    def test_a_missing_input_value_is_imputed_by_the_saved_pipeline(self):
        fields = helpers.build_prediction_fields(self.state)
        values = self._defaults()
        numeric = next(field for field in fields if field.kind == "number")
        values[numeric.name] = None
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertTrue(outcome.ok, outcome.message)

    def test_a_prediction_never_calls_a_training_function(self):
        from unittest import mock

        fields = helpers.build_prediction_fields(self.state)
        values = self._defaults()
        with mock.patch(
            "tools.persistence_tools.build_final_pipeline",
            side_effect=AssertionError("prediction must not refit"),
        ):
            outcome = helpers.run_prediction(self.state, fields, values)
        self.assertTrue(outcome.ok, outcome.message)

    def test_an_unknown_categorical_value_is_refused_with_a_reason(self):
        """Silently mapping it would invent a category the model never saw."""
        fields = helpers.build_prediction_fields(self.state)
        values = self._defaults()
        text_field = next(field for field in fields if field.kind == "text")
        values[text_field.name] = "not-a-real-category"
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.message)

    def test_an_empty_form_is_refused_with_a_reason_not_a_traceback(self):
        outcome = helpers.run_prediction(self.state, [], {})
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.message)


class TestPredictionWithoutProbabilitySupport(unittest.TestCase):
    """Check 14b: an estimator with no ``predict_proba`` still works."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = optimized_artifact_dir()
        cls.state = helpers.load_dashboard_state(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def setUp(self) -> None:
        if self.state.model.supports_probability:
            self.skipTest("the selected estimator does expose predict_proba")

    def test_the_form_is_built_normally(self):
        fields = helpers.build_prediction_fields(self.state)
        self.assertEqual([f.name for f in fields], self.state.feature_columns())

    def test_the_prediction_still_succeeds(self):
        fields = helpers.build_prediction_fields(self.state)
        values = {f.name: f.default for f in fields}
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertTrue(outcome.ok, outcome.message)
        self.assertIn(str(outcome.predicted_class), self.state.class_labels)

    def test_the_absent_probabilities_are_explained(self):
        """A blank area would look like a bug rather than a capability."""
        fields = helpers.build_prediction_fields(self.state)
        values = {f.name: f.default for f in fields}
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertIsNone(outcome.probabilities)
        self.assertTrue(outcome.message or outcome.detail)

    def test_the_section_says_so_before_the_user_predicts(self):
        section = helpers.build_prediction_section(self.state)
        self.assertFalse(section["supports_probability"])


# --------------------------------------------------------------------------- #
# 15. Multiclass, and the formatting helpers
# --------------------------------------------------------------------------- #
class TestMulticlassSections(unittest.TestCase):
    """Check 15: three classes through the same code path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = trained_state(MULTICLASS_DATASET)
        cls.state = helpers.load_dashboard_state(cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_three_classes_are_offered_on_the_form(self):
        self.assertEqual(len(self.state.class_labels), 3)
        section = helpers.build_prediction_section(self.state)
        self.assertEqual(len(section["class_labels"]), 3)

    def test_the_class_distribution_has_a_row_per_class(self):
        section = helpers.build_dataset_section(self.state)
        self.assertEqual(len(section["class_distribution"]), 3)

    def test_the_confusion_matrix_is_three_by_three_when_available(self):
        section = helpers.build_confusion_section(self.state)
        if not section["available"]:
            self.skipTest("no stored matrix for this run")
        self.assertEqual(len(section["labels"]), 3)
        self.assertFalse(section["is_binary"])
        self.assertEqual(len(section["frame"]), 9)
        for row in section["matrix"]:
            with self.subTest(row=row):
                self.assertEqual(len(row), 3)

    def test_a_prediction_returns_one_of_the_three_classes(self):
        fields = helpers.build_prediction_fields(self.state)
        values = {field.name: field.default for field in fields}
        outcome = helpers.run_prediction(self.state, fields, values)
        self.assertTrue(outcome.ok, outcome.message)
        self.assertIn(str(outcome.predicted_class), self.state.class_labels)

    def test_probabilities_cover_all_three_classes(self):
        fields = helpers.build_prediction_fields(self.state)
        values = {field.name: field.default for field in fields}
        outcome = helpers.run_prediction(self.state, fields, values)
        if outcome.probabilities is None:
            self.skipTest("the selected estimator exposes no predict_proba")
        self.assertEqual(len(outcome.probabilities), 3)


class TestFormattingHelpers(unittest.TestCase):
    """Check 15b: the small formatters, including the "not recorded" case."""

    def test_a_missing_metric_reads_as_a_dash_not_a_zero(self):
        """A zero would be read as a measured result."""
        self.assertEqual(helpers.format_metric(None), helpers.NOT_RECORDED)
        self.assertNotEqual(helpers.format_metric(None), "0.0000")

    def test_a_measured_metric_is_rounded_to_four_places(self):
        self.assertEqual(helpers.format_metric(0.8761234), "0.8761")
        self.assertEqual(helpers.format_metric(0.5, digits=2), "0.50")

    def test_a_non_numeric_parameter_is_shown_as_text(self):
        self.assertEqual(helpers.format_parameter("linear"), "linear")

    def test_no_parameters_reads_as_a_dash(self):
        self.assertEqual(helpers.format_parameters({}), helpers.NOT_RECORDED)
        self.assertEqual(helpers.format_parameters(None), helpers.NOT_RECORDED)

    def test_parameters_are_shown_in_a_stable_order(self):
        """Two runs must render the same text for the same parameters."""
        first = helpers.format_parameters({"b": 2, "a": 1})
        second = helpers.format_parameters({"a": 1, "b": 2})
        self.assertEqual(first, second)
        self.assertIn("a", first)
        self.assertIn("b", first)


# --------------------------------------------------------------------------- #
# 16. Artifact safety, asserted against the source
# --------------------------------------------------------------------------- #
class TestArtifactSafety(unittest.TestCase):
    """Check 16: the dashboard reads artifacts and writes nothing."""

    def test_the_helper_module_opens_no_file_for_writing(self):
        """Only the persistence layer may create a file."""
        tree = ast.parse(Path(helpers.__file__).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {
                "open",
                "write_text",
                "write_bytes",
            }:
                mode = node.args[0] if node.args else None
                self.fail(
                    f"{Path(helpers.__file__).name} opens a file at line "
                    f"{node.lineno}: the dashboard must not write"
                )

    def test_loading_the_dashboard_writes_nothing_to_the_artifact_directory(self):
        out = trained_state(BINARY_DATASET)
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        before = {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(out.iterdir())
        }
        helpers.load_dashboard_state(out)
        after = {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in sorted(out.iterdir())
        }
        self.assertEqual(before, after)

    def test_the_committed_datasets_are_unchanged(self):
        self.assertEqual(file_hash(BINARY_DATASET), EXPECTED_BINARY_HASH)
        self.assertEqual(file_hash(MULTICLASS_DATASET), EXPECTED_MULTICLASS_HASH)


# --------------------------------------------------------------------------- #
# 17. Rendering the real page
# --------------------------------------------------------------------------- #
class TestRenderedPage(unittest.TestCase):
    """Check 17: ``dashboard.py`` executes without raising.

    Streamlit runs a script top to bottom on every interaction, so a mistake in
    the view only shows up when the page is actually rendered. Streamlit's
    ``AppTest`` executes the script headlessly and reports the exception, which
    is the difference between catching this in the suite and leaving a user to
    discover it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Build a real optimized artifact for the page to render.

        A rendered test is only worth anything against a genuine artifact, and
        it must not depend on someone having run the workflow first - otherwise
        a clean checkout would report a red test that has nothing to do with the
        code. The tuned binary run is used because it exercises the most view
        branches: an optimized selection, two comparison rows for one model, an
        estimator with no ``predict_proba`` and no stored confusion matrix.
        """
        cls.out = optimized_artifact_dir()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def _render_from(self, directory: Path) -> Any:
        """Render the page as if the artifacts lived in ``directory``.

        Only the resolver is redirected. Everything else - the loader, every
        section builder, the whole of ``dashboard.py`` - runs for real, so a bug
        in the view still fails here.

        Args:
            directory: The directory the page should treat as the artifact
                directory.

        Returns:
            The executed application.
        """
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"Streamlit AppTest unavailable: {error}") from error
        import streamlit as st

        # The cache is emptied because ``dashboard.py`` wraps its loader in
        # ``st.cache_data`` with no arguments, which Streamlit keys on the
        # function's own source: without clearing it, whichever test rendered
        # first would decide the state every later test saw.
        st.cache_data.clear()
        real = helpers.artifact_paths
        with mock.patch.object(
            helpers, "artifact_paths", lambda output_dir=None: real(directory)
        ):
            return AppTest.from_file(str(DASHBOARD_MODULE), default_timeout=60).run()

    def _render(self) -> Any:
        """Render the page against this class's artifact.

        Returns:
            The executed application.
        """
        return self._render_from(self.out)

    def test_the_page_renders_with_a_persisted_artifact(self):
        """The full page, against an artifact this class built itself."""
        app = self._render_from(self.out)
        self.assertEqual(list(app.exception), [])

    def test_the_page_renders_with_no_artifact_at_all(self):
        """The empty-directory case must render too, not just not crash.

        The path resolver is redirected rather than the view, so
        ``dashboard.py`` genuinely runs against an unavailable state. Which
        directory is chosen is not what this test is about -
        ``TestLoadingWithoutAnArtifact`` covers that.
        """
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        app = self._render_from(empty)
        self.assertEqual(list(app.exception), [])
        body = " ".join(
            item.value
            for item in list(app.warning) + list(app.info) + list(app.markdown)
        )
        self.assertIn(config.DASHBOARD_NO_ARTIFACT, body)

    def test_the_page_shows_the_title_and_all_twelve_sections(self):
        app = self._render()
        self.assertEqual(app.title[0].value, config.DASHBOARD_TITLE)
        subtitles = [sub.value for sub in app.subheader]
        self.assertEqual(
            subtitles,
            [
                "2. System status",
                "3. Dataset overview",
                "4. Model comparison",
                "5. Evaluation",
                "6. Confusion matrix",
                "7. Optimization",
                "8. LLM interpretation",
                "9. Prediction",
                "10. Model information",
                "11. Experiment tracking",
                "12. Conventional vs Autonomous",
            ],
        )

    def test_the_header_names_the_seven_pipeline_stages(self):
        """The subtitle is the contract that the run had all seven stages."""
        app = self._render()
        captions = " ".join(item.value for item in app.caption)
        self.assertIn(config.DASHBOARD_SUBTITLE, captions)
        for stage in (
            "Data Analysis",
            "Preprocessing",
            "Model Selection",
            "Training",
            "Evaluation",
            "Optimization",
            "Model Persistence",
        ):
            with self.subTest(stage=stage):
                self.assertIn(stage, config.DASHBOARD_SUBTITLE)

    def test_the_page_offers_no_file_uploader(self):
        """Asserted on the rendered app, not only on the source text."""
        app = self._render()
        self.assertEqual(list(app.file_uploader), [])

    def test_the_prediction_form_is_rendered_from_the_saved_metadata(self):
        app = self._render()
        labels = [str(item.label) for item in app.number_input]
        labels += [str(item.label) for item in app.selectbox]
        model = load_persisted_model(self.out)
        self.assertIsNotNone(model, "no artifact to render a form from")
        for feature in model.feature_columns:
            with self.subTest(feature=feature):
                self.assertTrue(
                    any(feature in label for label in labels),
                    f"no input rendered for {feature}",
                )

    def test_a_prediction_runs_from_the_rendered_form(self):
        """The button click path, end to end, with no training involved."""
        app = self._render()
        self.assertEqual(len(app.button), 1)
        with mock.patch.object(
            Pipeline,
            "fit",
            side_effect=AssertionError("the dashboard fitted a pipeline"),
        ) as fit:
            app.button[0].click().run()
        fit.assert_not_called()
        self.assertEqual(list(app.exception), [])

    def test_the_page_offers_no_facilitating_link_for_uploaded_models(self):
        """The security note must be visible, not just implemented."""
        app = self._render()
        body = " ".join(
            [item.value for item in app.info]
            + [item.value for item in app.warning]
            + [item.value for item in app.caption]
            + [item.value for item in app.markdown]
        ).lower()
        self.assertIn("upload", body)


if __name__ == "__main__":
    unittest.main()