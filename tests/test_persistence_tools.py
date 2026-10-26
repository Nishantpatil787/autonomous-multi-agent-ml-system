"""Tests for model persistence (Phase 9, checks 1-10).

Covers the artifact paths, the deterministic selection policy, the final fit, the
metadata document, JSON serialisability, the atomic writes, the trusted reload,
the persistence agent, the security rule that ``joblib.load`` is called from
exactly one place, and the fact that neither dataset is ever written to.

The committed datasets are hashed before and after so a run that wrote to a
fixture cannot pass silently.
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

import numpy as np
import pandas as pd

from agents.data_analysis_agent import DataAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.optimization_agent import OptimizationAgent
from agents.persistence_agent import PersistenceAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.training_agent import TrainingAgent
from tests.helpers import (
    BINARY_DATASET,
    MULTICLASS_DATASET,
    ProjectArtifactIsolationGuard,
    file_hash,
)
from tools.persistence_tools import (
    ArtifactReadError,
    NoModelToPersistError,
    PersistenceError,
    artifact_paths,
    artifacts_exist,
    build_final_pipeline,
    build_metadata,
    build_run_results,
    json_safe,
    library_versions,
    load_persisted_model,
    load_run_results,
    read_json,
    resolve_output_dir,
    select_final_model,
    validate_schema,
    write_artifacts,
    write_json,
)
from utils import config
from utils.state import AgentState

#: Hashes of the committed fixtures. Persistence must never write to a dataset.
EXPECTED_BINARY_HASH = "49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c"
EXPECTED_MULTICLASS_HASH = (
    "0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8"
)

#: Root of the project, used by the source-inspection tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Real state, built once and reused
# --------------------------------------------------------------------------- #
def trained_state(dataset: Path, *, optimize: bool = False) -> AgentState:
    """Run the real pipeline up to (and including) evaluation.

    Optimization is skipped by default so the tests that only need a baseline
    selection stay fast; the tests about the optimized branch opt in.

    Args:
        dataset: The CSV to run on.
        optimize: Whether to also run the hyperparameter search.

    Returns:
        A finished :class:`~utils.state.AgentState`.
    """
    state = DataAnalysisAgent().run(AgentState(dataset_path=str(dataset)))
    state = PreprocessingAgent().run(state)
    state = ModelSelectionAgent().run(state)
    state = TrainingAgent().run(state)
    state = EvaluationAgent().run(state)
    if optimize:
        state = OptimizationAgent().run(state)
    return state


class PersistenceTestCase(unittest.TestCase):
    """Base class: a real trained state, and a private output directory."""

    dataset = BINARY_DATASET
    optimize = False

    @classmethod
    def setUpClass(cls) -> None:
        """Build the state once; fitting five models is the expensive part."""
        cls.state = trained_state(cls.dataset, optimize=cls.optimize)

    def setUp(self) -> None:
        """Give each test its own output directory and remove it afterwards."""
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def persist(self, state: AgentState = None) -> Dict[str, Any]:
        """Select, fit and write, then return the metadata document.

        Args:
            state: The state to persist, or None for the class-level one.

        Returns:
            The metadata document as it was written to disk.
        """
        state = self.state if state is None else state
        selection = select_final_model(state)
        final = build_final_pipeline(state, selection)
        paths = write_artifacts(state, selection, final, output_dir=self.out)
        return read_json(paths.metadata)


def expected_train_rows(state: AgentState) -> int:
    """Return the training-row count the earlier phases must have used.

    The split is rebuilt from the same configuration Phases 4-6 used, and the
    Evaluation Agent stored the hold-out size. Reading that back - rather than
    recomputing ``rows * 0.8`` - is what makes the assertion check the real
    split instead of an arithmetic guess. The duplicates Phase 2 reported are
    subtracted because the split happens after they are removed, so
    ``rows * 0.8`` would be wrong here and the test would silently prove
    nothing.
    """
    return int(
        state.dataset_shape.rows
        - state.evaluation_report.test_samples
        - int(state.duplicate_count or 0)
    )


# --------------------------------------------------------------------------- #
# 1. Artifact paths
# --------------------------------------------------------------------------- #
class TestArtifactPaths(PersistenceTestCase):
    """Check 1: where the three files go."""

    def test_default_directory_is_the_configured_models_directory(self):
        self.assertEqual(resolve_output_dir(), config.MODELS_DIR)

    def test_explicit_directory_wins(self):
        self.assertEqual(resolve_output_dir(self.out), self.out.resolve())

    def test_the_three_names_are_the_documented_ones(self):
        paths = artifact_paths(self.out)
        self.assertEqual(paths.model.name, "best_model.joblib")
        self.assertEqual(paths.metadata.name, "best_model_metadata.json")
        self.assertEqual(paths.results.name, "run_results.json")

    def test_all_three_live_in_the_same_directory(self):
        paths = artifact_paths(self.out)
        for path in (paths.model, paths.metadata, paths.results):
            with self.subTest(path=path.name):
                self.assertEqual(path.parent, paths.directory)

    def test_relative_names_omit_the_directory(self):
        """A report written on one machine must not carry a path from another."""
        self.assertEqual(
            artifact_paths(self.out).relative_names(),
            {
                "model": "best_model.joblib",
                "metadata": "best_model_metadata.json",
                "results": "run_results.json",
            },
        )

    def test_nothing_is_written_before_the_agent_runs(self):
        self.assertEqual(sorted(self.out.iterdir()), [])
        self.assertFalse(artifacts_exist(self.out))

    def test_artifacts_exist_needs_the_model_and_the_metadata(self):
        """The pair is the unit; the report bundle is optional."""
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        paths = write_artifacts(self.state, selection, final, output_dir=self.out)
        self.assertTrue(artifacts_exist(self.out))

        paths.metadata.unlink()
        self.assertFalse(artifacts_exist(self.out))
        self.assertIsNone(load_persisted_model(self.out))

    def test_a_model_without_the_results_bundle_is_still_usable(self):
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        paths = write_artifacts(self.state, selection, final, output_dir=self.out)
        paths.results.unlink()
        self.assertTrue(artifacts_exist(self.out))
        self.assertEqual(load_run_results(self.out), {})
        self.assertIsNotNone(load_persisted_model(self.out))


# --------------------------------------------------------------------------- #
# 2. Deterministic final-model selection
# --------------------------------------------------------------------------- #
class TestSelectionPolicy(PersistenceTestCase):
    """Check 2: the ranking rule is Phase 5/6's own, not a new one."""

    def test_baseline_run_selects_the_phase_five_winner(self):
        selection = select_final_model(self.state)
        best = self.state.evaluation_report.best_model
        self.assertEqual(selection.model_name, best.best_baseline_model)
        self.assertEqual(selection.selection_source, "baseline")
        self.assertEqual(selection.model_type, best.model_type)

    def test_the_rule_is_the_projects_existing_one(self):
        """No new metric and no new threshold is introduced here."""
        rule = select_final_model(self.state).selection_rule
        self.assertIn("f1_score", rule)
        self.assertIn("descending", rule)
        self.assertIn("model name", rule)
        self.assertIn("No new metric", rule)

    def test_selection_is_deterministic_across_calls(self):
        self.assertEqual(
            select_final_model(self.state), select_final_model(self.state)
        )

    def test_the_selection_records_both_candidates(self):
        selection = select_final_model(self.state)
        best = self.state.evaluation_report.best_model
        self.assertEqual(selection.baseline_model_name, best.best_baseline_model)
        self.assertAlmostEqual(selection.baseline_f1_score, best.f1_score)
        # With no search, the optimized candidate is empty rather than invented.
        self.assertEqual(selection.optimized_model_name, "")
        self.assertFalse(selection.optimization_used)

    def test_a_selected_baseline_reports_phase_five_metrics(self):
        """Baseline provenance is Phase 4's hold-out evaluation."""
        metadata = self.persist()
        self.assertEqual(metadata["metrics_source"], "evaluation_holdout")
        best = self.state.evaluation_report.best_model
        self.assertAlmostEqual(metadata["metrics"]["f1_score"], best.f1_score)
        self.assertAlmostEqual(metadata["metrics"]["accuracy"], best.accuracy)
        for key in ("precision", "recall", "f1_score", "accuracy"):
            with self.subTest(key=key):
                self.assertIsNotNone(metadata["metrics"][key])

    def test_a_selected_baseline_stores_its_confusion_matrix(self):
        metadata = self.persist()
        stored = metadata["confusion_matrix"]
        self.assertIsNotNone(stored)
        self.assertEqual(stored["labels"], ["0", "1"])
        for row in stored["matrix"]:
            with self.subTest(row=row):
                self.assertEqual(len(row), 2)

    def test_it_reuses_the_existing_metric_key_names(self):
        """``f1_score`` rather than ``f1``: earlier phases named it that way."""
        metadata = self.persist()
        self.assertIn("f1_score", metadata["metrics"])
        self.assertNotIn("f1", metadata["metrics"])

    def test_no_usable_model_raises_rather_than_writing_something(self):
        with self.assertRaises(NoModelToPersistError):
            select_final_model(AgentState(dataset_path=str(self.dataset)))

    def test_the_error_is_a_persistence_error(self):
        """The graph retry policy keys on the base class, not on each subclass."""
        self.assertTrue(issubclass(NoModelToPersistError, PersistenceError))


class TestOptimizedSelection(unittest.TestCase):
    """Check 2b: the branch where the hyperparameter search wins."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET, optimize=True)
        cls.selection = select_final_model(cls.state)
        cls.out = Path(tempfile.mkdtemp())
        final = build_final_pipeline(cls.state, cls.selection)
        cls.paths = write_artifacts(
            cls.state, cls.selection, final, output_dir=cls.out
        )
        cls.metadata = read_json(cls.paths.metadata)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def setUp(self) -> None:
        if self.selection.selection_source != "optimized":
            self.skipTest(
                "the search did not beat the baseline on this dataset, so the "
                "optimized branch cannot be exercised here"
            )

    def test_it_selects_the_phase_six_winner(self):
        """The recommendation names the winner, so the two must agree."""
        best = self.state.optimization_report.best_optimized_model
        self.assertEqual(self.selection.model_name, best.best_optimized_model)
        self.assertEqual(self.selection.selection_source, "optimized")
        self.assertTrue(self.selection.optimization_used)
        self.assertEqual(
            self.selection.optimized_model_name, best.best_optimized_model
        )

    def test_the_holdout_metrics_come_from_phase_six(self):
        """Provenance must never mix: an optimized score is a Phase 6 score."""
        result = self._winning_result()
        self.assertEqual(self.metadata["metrics_source"], "optimization_holdout")
        self.assertAlmostEqual(
            self.metadata["metrics"]["f1_score"], result.holdout_f1_score
        )
        self.assertAlmostEqual(
            self.metadata["metrics"]["accuracy"], result.holdout_accuracy
        )
        self.assertTrue(result.holdout_evaluated)

    def test_unrecorded_metrics_stay_none_rather_than_being_filled_in(self):
        """Phase 6 records no precision or recall; inventing them would be a
        second evaluation of a model the project only scored once."""
        self.assertIsNone(self.metadata["metrics"]["precision"])
        self.assertIsNone(self.metadata["metrics"]["recall"])

    def test_no_confusion_matrix_is_stored_for_a_tuned_model(self):
        """The absence is explained, not left as a bare ``null``."""
        self.assertIsNone(self.metadata["confusion_matrix"])
        note = self.metadata["confusion_matrix_note"]
        self.assertTrue(note)
        lowered = note.lower()
        self.assertIn("phase 6", lowered)
        self.assertIn("never re-scores", lowered)

    def test_the_search_parameters_travel_with_the_model(self):
        best = self.state.optimization_report.best_optimized_model
        self.assertEqual(self.metadata["best_params"], json_safe(best.best_params))
        self.assertTrue(self.metadata["best_params"])

    def test_the_baseline_candidate_is_still_recorded_for_comparison(self):
        """Both candidates travel in the selection, so the choice is auditable."""
        self.assertEqual(
            self.selection.baseline_model_name,
            self.state.evaluation_report.best_model.best_baseline_model,
        )
        self.assertIsNotNone(self.selection.baseline_f1_score)
        self.assertIsNotNone(self.selection.optimized_f1_score)
        self.assertGreaterEqual(
            self.selection.optimized_f1_score, self.selection.baseline_f1_score
        )

    def _winning_result(self) -> Any:
        """Return the ``OptimizationResult`` for the model that was selected.

        The hold-out numbers live on the result, not on the recommendation, which
        is why the selection has to join the two.

        Returns:
            The matching result.
        """
        for result in self.state.optimization_results:
            if result.model_name == self.selection.model_name:
                return result
        raise AssertionError(
            f"no optimization result for {self.selection.model_name!r}"
        )

    def test_the_reloaded_pipeline_is_the_tuned_one(self):
        loaded = load_persisted_model(self.out)
        estimator = loaded.pipeline.named_steps["model"]
        for name, value in self.metadata["best_params"].items():
            with self.subTest(parameter=name):
                actual = getattr(estimator, name)
                if isinstance(value, (int, float)):
                    self.assertAlmostEqual(float(actual), float(value))
                else:
                    # ``kernel="linear"`` and friends are categorical parameters.
                    self.assertEqual(str(actual), str(value))


class TestOptimizationFallbacks(unittest.TestCase):
    """Check 2c: every way the search can fail to produce a usable winner.

    The state is shared between these tests because building it means running a
    hyperparameter search. Each test therefore saves the attributes it changes
    and :meth:`setUp` puts them back, so order cannot matter.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(BINARY_DATASET, optimize=True)
        cls._saved: Dict[str, Any] = {}

    def setUp(self) -> None:
        """Restore anything a previous test changed."""
        for name, value in self._saved.items():
            setattr(self.state, name, value)
        self._saved.clear()

    def tearDown(self) -> None:
        """Restore immediately after the test, not only before the next one."""
        self._restore_results()
        for name, value in self._saved.items():
            setattr(self.state, name, value)
        self._saved.clear()

    def _replace(self, name: str, value: Any) -> None:
        """Set an attribute on the shared state and remember the old value.

        Args:
            name: The attribute on the shared state.
            value: The value to put there for this test.
        """
        if name not in self._saved:
            self._saved[name] = getattr(self.state, name)
        setattr(self.state, name, value)

    def _remember(self, key: str, value: Any) -> None:
        """Remember a value under a key for :meth:`_restore_results`.

        Used for values that live on a nested report rather than on the state
        itself, where :meth:`_replace` cannot be used.

        Args:
            key: A unique key for this test's saved value.
            value: The value to remember.
        """
        self._saved[key] = value

    def _clear_tuned_winner(self) -> None:
        """Remove the tuned winner from the Phase 6 report.

        The selection policy does not read ``optimization_status`` to decide
        whether a tuned candidate exists - it looks for a usable one. That is
        deliberate: a search that finished and produced nothing usable, and a
        search that never started, both arrive at the same place. So the honest
        way to simulate each failure is to remove the candidate the way the
        agent would have.
        """
        report = self.state.optimization_report
        self._remember("_best_tuned", report.best_optimized_model)
        report.best_optimized_model = None

    def _clear_tuned_holdout_score(self) -> None:
        """Strip the hold-out F1 from the tuned winner, leaving it in place."""
        report = self.state.optimization_report
        winner = report.best_optimized_model
        self._remember("_winner_f1", winner.holdout_f1_score)
        winner.holdout_f1_score = None

    def _restore_results(self) -> None:
        """Put back anything the test replaced on the shared report."""
        saved = self._saved
        if "_best_tuned" in saved:
            self.state.optimization_report.best_optimized_model = saved["_best_tuned"]
        if "_winner_f1" in saved:
            self.state.optimization_report.best_optimized_model.holdout_f1_score = (
                saved["_winner_f1"]
            )
        self._saved.clear()

    def test_a_failed_search_falls_back_to_the_baseline(self):
        """A failed search records no tuned winner, so the baseline is used."""
        self._replace("optimization_status", "failed")
        self._clear_tuned_winner()
        selection = select_final_model(self.state)
        self.assertEqual(selection.selection_source, "baseline")
        self.assertFalse(selection.optimization_used)
        self.assertEqual(selection.optimization_status, "failed")

    def test_a_search_that_never_started_falls_back_to_the_baseline(self):
        self._replace("optimization_status", "not_started")
        self._clear_tuned_winner()
        selection = select_final_model(self.state)
        self.assertEqual(selection.selection_source, "baseline")

    def test_a_tuned_model_without_a_holdout_f1_is_not_selectable(self):
        """A tuned model the search never scored on the hold-out set.

        The recommendation keeps its name and its parameters but carries no
        hold-out F1, so the documented primary metric is missing and the candidate
        is unusable. The baseline is persisted instead rather than a model whose
        score nobody measured.
        """
        self._clear_tuned_holdout_score()
        selection = select_final_model(self.state)
        self.assertEqual(selection.selection_source, "baseline")
        self.assertFalse(selection.optimization_used)

    def test_the_fallback_still_carries_the_phase_five_metrics(self):
        self._replace("optimization_status", "failed")
        self._clear_tuned_winner()
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        paths = write_artifacts(self.state, selection, final, output_dir=out)
        metadata = read_json(paths.metadata)
        self.assertEqual(metadata["selection_source"], "baseline")
        self.assertEqual(metadata["metrics_source"], "evaluation_holdout")
        self.assertIsNotNone(metadata["confusion_matrix"])

    def test_a_failed_search_writes_the_baseline_pipeline_unchanged(self):
        self._replace("optimization_status", "failed")
        self._clear_tuned_winner()
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        self.assertEqual(final.strategy, "reused_fitted_pipeline")
        self.assertIs(final.pipeline, self.state.best_pipeline)


# --------------------------------------------------------------------------- #
# 3. The final fit
# --------------------------------------------------------------------------- #
class TestFinalFit(PersistenceTestCase):
    """Check 3: fitting the deployable artifact, after selection."""

    def test_a_baseline_selection_reuses_the_already_fitted_pipeline(self):
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        self.assertEqual(final.strategy, "reused_fitted_pipeline")
        self.assertIs(final.pipeline, self.state.best_pipeline)

    def test_the_pipeline_carries_preprocessing_and_estimator(self):
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        self.assertEqual(
            list(final.pipeline.named_steps), ["preprocessor", "model"]
        )

    def test_the_reused_pipeline_records_no_training_sample_count(self):
        """It was fitted earlier, by the Training Agent; Phase 9 did not refit.

        The count is deliberately zero rather than a guess: reporting a number
        here would imply Phase 9 performed the fit.
        """
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        self.assertEqual(final.strategy, "reused_fitted_pipeline")
        self.assertEqual(final.train_samples, 0)

    def test_a_rebuilt_pipeline_records_the_training_split_size(self):
        """The rebuilt path refits, so it can say how many rows it used."""
        state = trained_state(self.dataset, optimize=True)
        selection = select_final_model(state)
        if selection.selection_source != "optimized":
            self.skipTest("the baseline won, so no pipeline was rebuilt")
        final = build_final_pipeline(state, selection)
        rows = state.dataset_shape.rows
        self.assertGreater(final.train_samples, 0)
        self.assertLess(final.train_samples, rows)
        self.assertEqual(final.train_samples, expected_train_rows(state))
        # Never train + test together: the hold-out rows stay unseen.
        self.assertLess(
            final.train_samples + int(state.evaluation_report.test_samples), rows
        )

    def test_the_rebuilt_note_states_that_fitting_follows_selection(self):
        state = trained_state(self.dataset, optimize=True)
        selection = select_final_model(state)
        if selection.selection_source != "optimized":
            self.skipTest("the baseline won, so no pipeline was rebuilt")
        final = build_final_pipeline(state, selection)
        note = final.note.lower()
        self.assertIn("after", note)
        self.assertIn("hold-out", note)

    def test_fitting_does_not_change_any_recorded_metric(self):
        """Persistence must not move a number Phases 4-6 already produced."""
        before_experiments = [r.f1_score for r in self.state.experiment_results]
        before_best = self.state.evaluation_report.best_model.f1_score
        before_statuses = (
            self.state.evaluation_status,
            self.state.optimization_status,
            self.state.training_status,
        )

        selection = select_final_model(self.state)
        build_final_pipeline(self.state, selection)

        self.assertEqual([r.f1_score for r in self.state.experiment_results],
                         before_experiments)
        self.assertEqual(self.state.evaluation_report.best_model.f1_score, before_best)
        self.assertEqual(
            (self.state.evaluation_status, self.state.optimization_status,
             self.state.training_status),
            before_statuses,
        )

    def test_the_metadata_reports_which_strategy_was_used(self):
        metadata = self.persist()
        self.assertEqual(metadata["final_fit"]["strategy"], "reused_fitted_pipeline")
        self.assertTrue(metadata["final_fit"]["note"])


# --------------------------------------------------------------------------- #
# 4. Metadata and JSON serialisability
# --------------------------------------------------------------------------- #
class TestMetadata(PersistenceTestCase):
    """Check 4: the metadata document."""

    def setUp(self) -> None:
        super().setUp()
        self.metadata = self.persist()

    def test_it_survives_a_json_round_trip(self):
        self.assertEqual(json.loads(json.dumps(self.metadata)), self.metadata)

    def test_every_leaf_is_a_json_primitive(self):
        """No Python objects, and no tuples, which JSON would silently turn into
        lists and change a document's meaning."""

        def walk(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    with self.subTest(path=f"{path}.{key}"):
                        self.assertIsInstance(key, str)
                    walk(item, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple)):
                with self.subTest(path=path):
                    self.assertNotIsInstance(value, tuple)
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")
                return
            with self.subTest(path=path):
                self.assertIsInstance(value, (str, int, float, bool, type(None)))

        walk(self.metadata)

    def test_numpy_scalars_become_plain_numbers(self):
        """``json_safe`` handles numpy before falling back to a string, because a
        stringified ``numpy.int64(7)`` would read as a measurement of "7"."""
        self.assertIsInstance(json_safe(np.int64(7)), int)
        self.assertIsInstance(json_safe(np.float64(0.5)), float)
        self.assertIsInstance(json_safe(np.bool_(True)), bool)
        self.assertEqual(json_safe(np.array([1, 2])), [1, 2])
        self.assertEqual(json_safe(np.float32(1.5)), 1.5)

    def test_json_safe_does_not_lose_information(self):
        self.assertEqual(json_safe({"a": np.int64(3), "b": [np.float64(0.5)]}),
                         {"a": 3, "b": [0.5]})

    def test_the_required_fields_are_present(self):
        for key in (
            "model_name",
            "model_type",
            "selection_source",
            "target_column",
            "feature_columns",
            "task_type",
            "metrics",
            "cv_metrics",
            "best_params",
            "optimization_used",
            "schema_version",
        ):
            with self.subTest(key=key):
                self.assertIn(key, self.metadata)

    def test_it_only_includes_fields_that_exist_in_the_state(self):
        """No fabricated values for a run that recorded nothing.

        A baseline selection has no tuned parameters, so it must not present the
        untuned estimator's defaults as if a search had chosen them. What it does
        carry is the seed, which is a real recorded value.
        """
        text = json.dumps(self.metadata)
        self.assertEqual(self.metadata["optimized_model_name"], "")
        self.assertEqual(self.metadata["optimization_used"], False)
        # Absent, not zero: a zero here would read as a model that scored nothing.
        self.assertIsNone(self.metadata["optimized_f1_score"])
        self.assertEqual(
            self.metadata["best_params"].get("random_state"), config.RANDOM_STATE
        )
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_feature_columns_match_the_pipeline_inputs(self):
        self.assertEqual(
            self.metadata["feature_columns"],
            list(self.state.preprocessing_numerical_features or [])
            + list(self.state.preprocessing_categorical_features or []),
        )
        self.assertEqual(
            self.metadata["feature_count"], len(self.metadata["feature_columns"])
        )

    def test_the_target_column_is_not_a_feature(self):
        self.assertNotIn(
            self.metadata["target_column"], self.metadata["feature_columns"]
        )

    def test_it_stores_no_absolute_path(self):
        text = json.dumps(self.metadata)
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn(self.out.as_posix(), text)
        self.assertEqual(self.metadata["artifact"], "best_model.joblib")

    def test_it_stores_no_secret_or_environment_value(self):
        text = json.dumps(self.metadata).lower()
        for word in ("api_key", "apikey", "password", "secret", "token",
                     "os.environ", "authorization"):
            with self.subTest(word=word):
                self.assertNotIn(word, text)

    def test_it_stores_no_dataset_row(self):
        """A single row can identify a person; the file holds aggregates only."""
        frame = pd.read_csv(self.dataset)
        value = str(frame[self.state.target_column].iloc[0])
        with self.subTest(value=value):
            self.assertNotIn(f'"{value}": ', json.dumps(self.metadata))

    def test_class_labels_come_from_the_fitted_estimator(self):
        self.assertEqual(self.metadata["class_labels"], ["0", "1"])

    def test_library_versions_are_recorded_for_reproducibility(self):
        """A pickle only reloads in an environment that matches it."""
        versions = self.metadata["library_versions"]
        self.assertIn("scikit-learn", versions)
        self.assertIn("joblib", versions)
        self.assertEqual(versions, library_versions())

    def test_the_size_recorded_matches_the_file(self):
        self.assertGreater(self.metadata["artifact_size_bytes"], 0)
        self.assertEqual(
            self.metadata["artifact_size_bytes"],
            artifact_paths(self.out).model.stat().st_size,
        )

    def test_it_records_its_limitations(self):
        self.assertTrue(self.metadata["limitations"])
        joined = " ".join(self.metadata["limitations"]).lower()
        self.assertIn("split", joined)
        self.assertIn("upload", joined)


# --------------------------------------------------------------------------- #
# 5. Writing, atomicity and the trusted reload
# --------------------------------------------------------------------------- #
class TestWriteAndReload(PersistenceTestCase):
    """Check 5: save, then predict from the saved artifact."""

    def setUp(self) -> None:
        super().setUp()
        self.metadata = self.persist()
        self.loaded = load_persisted_model(self.out)

    def test_all_three_files_exist_and_are_non_empty(self):
        paths = artifact_paths(self.out)
        for path in (paths.model, paths.metadata, paths.results):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_the_reloaded_pipeline_predicts_the_same_class(self):
        row = self._a_row()
        in_memory = self.state.best_pipeline.predict(row)
        self.assertEqual(
            str(in_memory[0]), str(self.loaded.pipeline.predict(row)[0])
        )

    def test_the_reloaded_pipeline_agrees_on_probabilities(self):
        if not self.loaded.supports_probability:
            self.skipTest("the selected estimator exposes no predict_proba")
        row = self._a_row()
        np.testing.assert_allclose(
            self.state.best_pipeline.predict_proba(row),
            self.loaded.pipeline.predict_proba(row),
        )

    def test_the_reloaded_pipeline_needs_no_preprocessing_from_the_caller(self):
        """The saved object does the imputing, encoding and scaling itself.

        The row goes in with a missing numerical value and no pre-transformed
        columns. If the caller had to impute or scale first, this would raise.
        """
        row = self._a_row()
        numeric = self.state.preprocessing_numerical_features
        self.assertGreater(int(row[numeric[0]].isna().sum()), 0)
        self.assertEqual(len(self.loaded.pipeline.predict(row)), 1)

    def test_the_reloaded_metadata_is_attached(self):
        self.assertEqual(self.loaded.model_name, self.metadata["model_name"])
        self.assertEqual(self.loaded.target_column, self.state.target_column)
        self.assertEqual(
            self.loaded.feature_columns, self.metadata["feature_columns"]
        )
        self.assertEqual(self.loaded.class_labels, ["0", "1"])

    def test_probability_support_is_reported_not_assumed(self):
        self.assertIsInstance(self.loaded.supports_probability, bool)

    def test_a_single_file_write_is_atomic(self):
        """A write replaces the target and leaves no temporary file behind."""
        target = self.out / "atomic.json"
        write_json(target, {"first": True})
        write_json(target, {"second": True})
        self.assertEqual(read_json(target), {"second": True})
        expected = {
            "atomic.json",
            "best_model.joblib",
            "best_model_metadata.json",
            "run_results.json",
        }
        self.assertEqual({p.name for p in self.out.iterdir()}, expected)

    def test_write_json_creates_the_directory_it_was_asked_for(self):
        target = self.out / "deep" / "deeper" / "value.json"
        write_json(target, {"a": 1})
        self.assertEqual(read_json(target), {"a": 1})

    def test_a_failure_on_the_bundle_still_leaves_a_usable_model(self):
        """The model and its metadata are written before the report bundle.

        Ordering that way means a failure on the third write cannot leave metadata
        describing a model that is not there. What it does leave is a model the
        dashboard can load, with the report sections missing - which the dashboard
        reports as its own situation rather than as a broken artifact.
        """
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        with mock.patch(
            "tools.persistence_tools.build_run_results",
            side_effect=RuntimeError("bundle failed"),
        ):
            with self.assertRaises(RuntimeError):
                selection = select_final_model(self.state)
                write_artifacts(
                    self.state,
                    selection,
                    build_final_pipeline(self.state, selection),
                    output_dir=out,
                )
        self.assertTrue(artifacts_exist(out))
        self.assertIsNotNone(load_persisted_model(out))
        # The report bundle was never written, so there is nothing to read.
        self.assertEqual(load_run_results(out), {})
        self.assertFalse(artifact_paths(out).results.exists())

    def test_a_reload_after_a_second_run_reflects_the_second_run(self):
        """An explicit new run overwrites; there is no implicit merge."""
        second = trained_state(MULTICLASS_DATASET)
        write_artifacts(
            second,
            select_final_model(second),
            build_final_pipeline(second, select_final_model(second)),
            output_dir=self.out,
        )
        reloaded = load_persisted_model(self.out)
        self.assertEqual(reloaded.target_column, "class_label")
        self.assertEqual(len(reloaded.class_labels), 3)

    def _a_row(self) -> pd.DataFrame:
        """Return one input row, target removed and a value made missing."""
        frame = pd.read_csv(self.dataset)
        row = frame.drop(columns=[self.state.target_column]).head(1).copy()
        numeric = self.state.preprocessing_numerical_features
        if numeric:
            row[numeric[0]] = None
        return row


# --------------------------------------------------------------------------- #
# 6. Errors: missing, malformed, unwritable
# --------------------------------------------------------------------------- #
class TestReadErrors(PersistenceTestCase):
    """Check 6: what happens when an artifact is absent or damaged."""

    def test_a_missing_model_is_reported_as_none_not_raised(self):
        """The dashboard must start before anything has been persisted."""
        self.assertIsNone(load_persisted_model(self.out))

    def test_a_corrupt_model_is_reported_as_a_read_error(self):
        paths = artifact_paths(self.out)
        paths.model.write_text("this is not a pickle", encoding="utf-8")
        paths.metadata.write_text(
            json.dumps({"schema_version": config.PERSISTENCE_SCHEMA_VERSION}),
            encoding="utf-8",
        )
        with self.assertRaises(ArtifactReadError):
            load_persisted_model(self.out)

    def test_metadata_without_a_schema_version_is_refused(self):
        """Guessing at an unknown layout would give a wrong reading rather than
        an honest "not available"."""
        with self.assertRaises(ArtifactReadError):
            validate_schema({"model_name": "x"}, name="metadata")

    def test_metadata_from_a_newer_project_is_refused(self):
        with self.assertRaises(ArtifactReadError):
            validate_schema(
                {"schema_version": config.PERSISTENCE_SCHEMA_VERSION + 1},
                name="metadata",
            )

    def test_a_non_numeric_schema_version_is_refused(self):
        with self.assertRaises(ArtifactReadError):
            validate_schema({"schema_version": "one"}, name="metadata")

    def test_a_mismatched_artifact_pair_is_refused(self):
        """Part of the set was replaced, so the pair cannot be trusted together."""
        paths = artifact_paths(self.out)
        paths.model.write_bytes(b"")
        paths.metadata.write_text(
            json.dumps(
                {
                    "schema_version": config.PERSISTENCE_SCHEMA_VERSION,
                    "artifact": "some_other_model.joblib",
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ArtifactReadError):
            load_persisted_model(self.out)

    def test_invalid_json_is_reported_as_a_read_error(self):
        paths = artifact_paths(self.out)
        paths.results.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(ArtifactReadError):
            load_run_results(self.out)

    def test_an_unwritable_directory_is_reported_not_silently_skipped(self):
        blocked = self.out / "models"
        blocked.write_text("a file, not a directory", encoding="utf-8")
        selection = select_final_model(self.state)
        final = build_final_pipeline(self.state, selection)
        with self.assertRaises(PersistenceError):
            write_artifacts(self.state, selection, final, output_dir=blocked)

    def test_every_read_error_shares_one_base_class(self):
        self.assertTrue(issubclass(ArtifactReadError, PersistenceError))


# --------------------------------------------------------------------------- #
# 7. The run-results bundle
# --------------------------------------------------------------------------- #
class TestRunResults(PersistenceTestCase):
    """Check 7: the document the dashboard reads its sections from."""

    def setUp(self) -> None:
        super().setUp()
        self.selection = select_final_model(self.state)
        self.results = build_run_results(self.state, llm_results=[],
                                         selection=self.selection)

    def test_it_has_the_sections_the_dashboard_asks_for(self):
        for key in ("dataset", "evaluation", "optimization", "baseline_models",
                    "optimized_models", "final_model", "llm", "notes",
                    "produced_by", "schema_version"):
            with self.subTest(key=key):
                self.assertIn(key, self.results)

    def test_it_is_json_serialisable(self):
        self.assertEqual(json.loads(json.dumps(self.results)), self.results)

    def test_it_copies_rather_than_recomputes(self):
        """The dataset section is the Phase 2 record, unchanged."""
        self.assertEqual(
            self.results["dataset"]["rows"], self.state.dataset_shape.rows
        )
        self.assertEqual(
            self.results["dataset"]["target_column"], self.state.target_column
        )

    def test_the_dataset_section_holds_aggregates_only(self):
        """It must not carry a per-row list that could identify a person."""
        expected = {
            "name", "rows", "columns", "numerical_feature_count",
            "categorical_feature_count", "numerical_features",
            "categorical_features", "excluded_features", "total_missing_values",
            "affected_missing_columns", "duplicate_count", "class_distribution",
            "problem_type", "problem_type_reason", "target_column",
            "target_detection_method", "target_detection_confidence",
            "numerical_statistics", "categorical_statistics", "issue_count",
            "issues_by_severity", "summary_text",
        }
        self.assertEqual(set(self.results["dataset"]), expected)
        # A raw frame would have one entry per row; a distribution has one per
        # class. Checking the length makes that difference visible.
        self.assertLessEqual(
            len(self.results["dataset"]["class_distribution"]),
            self.results["dataset"]["rows"],
        )

    def test_the_llm_section_works_with_nothing_generated(self):
        self.assertEqual(self.results["llm"], {"enabled": False, "results": []})

    def test_the_llm_section_copies_passed_records_verbatim(self):
        class _Result:
            def to_dict(self) -> Dict[str, Any]:
                return {"role": "evaluation", "status": "completed", "model": "x"}

        with_llm = build_run_results(self.state, llm_results=[_Result()])
        self.assertTrue(with_llm["llm"]["enabled"])
        self.assertEqual(len(with_llm["llm"]["results"]), 1)

    def test_baseline_rows_keep_their_recorded_rank(self):
        ranks = [row["rank"] for row in self.results["baseline_models"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_baseline_rows_are_labelled_baseline(self):
        for row in self.results["baseline_models"]:
            with self.subTest(model=row["model_name"]):
                self.assertEqual(row["status"], "baseline")

    def test_the_final_model_section_names_the_selection(self):
        self.assertEqual(
            self.results["final_model"]["model_name"], self.selection.model_name
        )
        self.assertEqual(
            self.results["final_model"]["selection_source"],
            self.selection.selection_source,
        )


# --------------------------------------------------------------------------- #
# 8. Security and artifact safety
# --------------------------------------------------------------------------- #
class TestSecurityAndSafety(unittest.TestCase):
    """Check 8: the loading rule, and that nothing else is touched."""

    def _project_sources(self) -> List[Path]:
        """Return every first-party project ``.py`` file, excluding caches.

        A virtual environment is not project source. ``rglob`` would otherwise
        descend into ``.venv`` and find a third-party test fixture that is
        deliberately not UTF-8, making this security check fail for a reason
        unrelated to the project. The same exclusion keeps ``.git`` and tool
        caches out of the scan.
        """
        excluded_parts = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "env",
            "node_modules",
            "site-packages",
            "venv",
        }
        return [
            path
            for path in sorted(PROJECT_ROOT.rglob("*.py"))
            if not excluded_parts.intersection(path.parts)
        ]

    def test_only_the_persistence_module_calls_joblib_load(self):
        """Unpickling executes code, so there must be exactly one door.

        Checked by parsing every file rather than by importing, because an
        import-only check would miss a call inside a function that never runs.
        """
        offenders: List[str] = []
        for path in self._project_sources():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                dump = ast.dump(node.func)
                if "attr: 'load'" not in dump and "id: 'load'" not in dump:
                    continue
                if "joblib" in dump:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            f"joblib.load must live in tools/persistence_tools.py only, found {offenders}",
        )

    def test_the_loading_function_takes_a_directory_not_a_model_file(self):
        """There is no argument through which a caller could name a file."""
        import inspect

        parameters = inspect.signature(load_persisted_model).parameters
        self.assertEqual(list(parameters), ["output_dir"])

    @staticmethod
    def _is_csv_only_dataset_uploader(node: ast.Call) -> bool:
        """Return True only for an uploader that declares CSV as its sole type."""
        for keyword in node.keywords:
            if keyword.arg != "type":
                continue
            value = keyword.value
            if isinstance(value, ast.List):
                declared = {
                    str(element.value).lower().lstrip(".")
                    for element in value.elts
                    if isinstance(element, ast.Constant)
                }
                return len(value.elts) == 1 and declared == {"csv"}
            if isinstance(value, ast.Constant):
                return str(value.value).lower().lstrip(".") == "csv"
        return False

    def test_no_module_offers_to_accept_an_uploaded_artifact(self):
        """Checked against the parsed call graph, not the text.

        The dashboard's docstring explains *why* there is no model uploader, so a
        text search would flag its own explanation. A CSV dataset uploader is
        allowed because a CSV is data; a model file is a pickle and stays banned.
        """
        offenders: List[str] = []
        banned_attributes = {"file_uploader", "upload_file", "unpickle"}
        banned_names = {"unpickle", "loads"}
        for path in self._project_sources():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in banned_attributes:
                    if func.attr == "file_uploader" and self._is_csv_only_dataset_uploader(
                        node
                    ):
                        continue
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
                if (
                    isinstance(func, ast.Name)
                    and func.id in banned_names
                    and "pickle" in ast.dump(node)
                ):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_the_dashboard_module_does_not_import_agents(self):
        """It reads artifacts; it must not be able to start a workflow."""
        tree = ast.parse((PROJECT_ROOT / "dashboard.py").read_text(encoding="utf-8"))
        imported: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            with self.subTest(module=name):
                self.assertFalse(name.startswith("agents"))
                self.assertFalse(name.startswith("graph"))
                self.assertFalse(name.startswith("optuna"))

    def test_persistence_never_writes_to_the_datasets_directory(self):
        state = trained_state(BINARY_DATASET)
        before = {
            path.name: file_hash(path)
            for path in sorted(config.DATASETS_DIR.glob("*.csv"))
        }
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        selection = select_final_model(state)
        write_artifacts(
            state, selection, build_final_pipeline(state, selection),
            output_dir=out,
        )
        after = {
            path.name: file_hash(path)
            for path in sorted(config.DATASETS_DIR.glob("*.csv"))
        }
        self.assertEqual(before, after)

    def test_the_committed_datasets_are_unchanged(self):
        """The Phase 9 acceptance check, asserted rather than assumed."""
        self.assertEqual(file_hash(BINARY_DATASET), EXPECTED_BINARY_HASH)
        self.assertEqual(file_hash(MULTICLASS_DATASET), EXPECTED_MULTICLASS_HASH)

    def test_persistence_never_writes_to_the_preprocessing_configuration(self):
        state = trained_state(BINARY_DATASET)
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        selection = select_final_model(state)
        write_artifacts(
            state, selection, build_final_pipeline(state, selection),
            output_dir=out,
        )
        # The only files the run created are the three artifacts.
        self.assertEqual(
            sorted(path.name for path in out.iterdir()),
            ["best_model.joblib", "best_model_metadata.json", "run_results.json"],
        )


# --------------------------------------------------------------------------- #
# 9. The agent
# --------------------------------------------------------------------------- #
class TestPersistenceAgent(PersistenceTestCase):
    """Check 9: the agent that drives the tools."""

    def test_a_successful_run_records_completed_and_writes(self):
        state = PersistenceAgent().run(self.state, output_dir=self.out)
        self.assertEqual(state.persistence_status, "completed")
        self.assertTrue(state.has_persistence_report)
        self.assertTrue(artifacts_exist(self.out))
        self.assertIsNotNone(state.final_model_selection)
        self.assertEqual(state.model_artifact_name, "best_model.joblib")

    def test_it_adds_no_new_status_value(self):
        """A written artifact is still a completed run; ``status`` is unchanged."""
        before = self.state.status
        state = PersistenceAgent().run(self.state, output_dir=self.out)
        self.assertEqual(state.status, before)
        self.assertEqual(state.persistence_status, "completed")

    def test_it_sets_only_its_own_two_fields(self):
        """It must not touch an earlier phase's recorded outcome."""
        before = (
            self.state.evaluation_status,
            self.state.optimization_status,
            self.state.training_status,
            self.state.preprocessing_status,
        )
        PersistenceAgent().run(self.state, output_dir=self.out)
        after = (
            self.state.evaluation_status,
            self.state.optimization_status,
            self.state.training_status,
            self.state.preprocessing_status,
        )
        self.assertEqual(before, after)

    def test_a_disabled_run_records_skipped_and_writes_nothing(self):
        state = PersistenceAgent().run(self.state, output_dir=self.out, enabled=False)
        self.assertEqual(state.persistence_status, "skipped")
        self.assertFalse(artifacts_exist(self.out))

    def test_a_run_with_no_model_records_failed_and_raises(self):
        """The agent reports into the state *and* raises.

        Both halves matter. The typed error is what the graph node turns into a
        ``PARTIAL`` workflow rather than a crash, and the recorded status is what
        the report and the dashboard read afterwards.
        """
        empty = AgentState(dataset_path=str(self.dataset))
        with self.assertRaises(NoModelToPersistError):
            PersistenceAgent().run(empty, output_dir=self.out)
        self.assertEqual(empty.persistence_status, "failed")
        self.assertFalse(artifacts_exist(self.out))
        # The status and the message are the record; the typed exception is the
        # signal. Nothing invents a report for a run that produced no artifact.
        self.assertEqual(len(empty.errors), 1)
        self.assertIn("Training Agent", empty.errors[0])

    def test_a_failure_after_a_good_run_leaves_the_old_artifact_loadable(self):
        """Overwrite happens only as part of a successful write."""
        good = PersistenceAgent().run(self.state, output_dir=self.out)
        self.assertEqual(good.persistence_status, "completed")

        with self.assertRaises(NoModelToPersistError):
            PersistenceAgent().run(
                AgentState(dataset_path=str(self.dataset)), output_dir=self.out
            )
        self.assertIsNotNone(load_persisted_model(self.out))

    def test_a_refusal_to_overwrite_leaves_the_old_artifact_intact(self):
        PersistenceAgent().run(self.state, output_dir=self.out)
        before = artifact_paths(self.out).model.read_bytes()
        with self.assertRaises(PersistenceError):
            PersistenceAgent().run(self.state, output_dir=self.out, overwrite=False)
        self.assertEqual(artifact_paths(self.out).model.read_bytes(), before)
        self.assertIsNotNone(load_persisted_model(self.out))

    def test_a_second_run_overwrites_and_stays_loadable(self):
        PersistenceAgent().run(self.state, output_dir=self.out)
        second = PersistenceAgent().run(self.state, output_dir=self.out)
        self.assertEqual(second.persistence_status, "completed")
        self.assertIsNotNone(load_persisted_model(self.out))

    def test_the_report_names_the_three_artifacts_relative_to_the_directory(self):
        state = PersistenceAgent().run(self.state, output_dir=self.out)
        report = state.persistence_report
        self.assertEqual(report.model_artifact, "best_model.joblib")
        self.assertEqual(report.metadata_artifact, "best_model_metadata.json")
        self.assertEqual(report.results_artifact, "run_results.json")
        self.assertNotIn(self.out.as_posix(), report.model_artifact)

    def test_the_report_states_the_features_and_classes(self):
        state = PersistenceAgent().run(self.state, output_dir=self.out)
        report = state.persistence_report
        self.assertEqual(report.class_count, 2)
        self.assertEqual(
            report.feature_count,
            len(self.state.preprocessing_numerical_features or [])
            + len(self.state.preprocessing_categorical_features or []),
        )


# --------------------------------------------------------------------------- #
# 10. Multiclass
# --------------------------------------------------------------------------- #
class TestMulticlassArtifacts(unittest.TestCase):
    """Check 10: the same code path with three classes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = trained_state(MULTICLASS_DATASET)
        cls.out = Path(tempfile.mkdtemp())
        cls.result = PersistenceAgent().run(cls.state, output_dir=cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_run_completes(self):
        self.assertEqual(self.result.persistence_status, "completed")

    def test_three_class_labels_are_persisted(self):
        loaded = load_persisted_model(self.out)
        self.assertEqual(len(loaded.class_labels), 3)
        self.assertEqual(self.result.persistence_report.class_count, 3)

    def test_a_three_class_matrix_is_stored_when_the_baseline_wins(self):
        stored = read_json(artifact_paths(self.out).metadata)["confusion_matrix"]
        if stored is None:
            self.skipTest("an optimized model was selected, so no matrix exists")
        self.assertEqual(len(stored["labels"]), 3)
        for row in stored["matrix"]:
            with self.subTest(row=row):
                self.assertEqual(len(row), 3)

    def test_the_saved_pipeline_predicts_three_classes(self):
        loaded = load_persisted_model(self.out)
        frame = pd.read_csv(MULTICLASS_DATASET)
        row = frame.drop(columns=["class_label"]).head(5)
        predictions = loaded.pipeline.predict(row)
        self.assertEqual(len(predictions), 5)
        for value in predictions:
            with self.subTest(prediction=value):
                self.assertIn(str(value), loaded.class_labels)


class TestProjectArtifactIsolationGuard(unittest.TestCase):
    """Check 12: the guard that catches a test writing into ``models/``.

    Phase 9 turned persistence on by default, and several Phase 8 integration
    tests drive the real agents on the real datasets. Those tests therefore
    replaced the committed artifact with a different one while the suite still
    reported success.

    A guard cannot fix that by itself, so these tests only establish that the
    guard is capable of reporting the three ways it could be violated. If the
    guard silently passed, it would be worse than having none: it would look
    like the problem was handled.
    """

    def setUp(self) -> None:
        """Create a throwaway directory with one file per failure mode."""
        super().setUp()
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        (self.out / "kept.txt").write_text("original", encoding="utf-8")
        (self.out / "doomed.txt").write_text("temporary", encoding="utf-8")
        self.guard = ProjectArtifactIsolationGuard(self.out)
        self.guard.start()

    def test_an_untouched_directory_reports_nothing(self):
        self.assertEqual(self.guard.verify(), [])

    def test_a_created_file_is_reported(self):
        (self.out / "extra.txt").write_text("new", encoding="utf-8")
        self.assertEqual(self.guard.verify(), ["extra.txt: created by a test"])

    def test_a_deleted_file_is_reported(self):
        (self.out / "doomed.txt").unlink()
        self.assertEqual(self.guard.verify(), ["doomed.txt: deleted by a test"])

    def test_an_overwritten_file_is_reported(self):
        """The failure this guard actually exists for.

        Rewriting a model with a different model leaves the same file name and a
        plausible size, so nothing else in the suite would notice.
        """
        (self.out / "kept.txt").write_text("replaced", encoding="utf-8")
        self.assertEqual(self.guard.verify(), ["kept.txt: overwritten by a test"])

    def test_a_missing_directory_is_not_a_violation(self):
        """A fresh checkout has no ``models/`` yet; that is a valid state."""
        absent = Path(tempfile.mkdtemp()) / "not_created"
        self.addCleanup(shutil.rmtree, absent.parent, ignore_errors=True)
        guard = ProjectArtifactIsolationGuard(absent)
        guard.start()
        self.assertEqual(guard.verify(), [])

    def test_the_guard_is_installed_for_the_session(self):
        """``conftest.py`` must hold one, or nothing enforces any of the above."""
        conftest = Path(__file__).parent / "conftest.py"
        self.assertTrue(conftest.exists(), "tests/conftest.py is missing")
        source = conftest.read_text(encoding="utf-8")
        self.assertIn("ProjectArtifactIsolationGuard", source)
        self.assertIn("pytest_sessionstart", source)
        self.assertIn("pytest_sessionfinish", source)

    def test_no_test_writes_into_the_configured_models_directory(self):
        """Every real-agent persistence call must name an output directory.

        This is the check the guard cannot make on its own. A guard reports *that*
        a file changed; only a source-level check can say *which call* was
        responsible while the developer is still looking at the code.

        The projects in this suite are the workflows that the CLI runs. A test
        that exercises one of them without saying where the artifact should go
        writes to the project's own ``models/``, so each such call is required
        to either name a directory or turn persistence off. The same rule covers
        calling ``PersistenceAgent().run`` directly: unlike ``run_workflow`` it
        has no workflow plumbing, so ``output_dir`` (or ``enabled=False``) is
        the only way to keep the write out of ``models/``.
        """
        offenders: List[str] = []
        for path in sorted(Path(__file__).parent.glob("test_*.py")):
            # ``utf-8-sig`` because some of these modules carry a byte order
            # mark, which is legal for the interpreter and not for ``ast``.
            text = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == "run_workflow":
                    keywords = {keyword.arg for keyword in node.keywords}
                    if keywords & {"enable_persistence", "persistence_output_dir"}:
                        continue
                    label = "run_workflow"
                elif self._is_direct_persistence_run(node):
                    keywords = {keyword.arg for keyword in node.keywords}
                    if keywords & {"output_dir", "enabled"}:
                        continue
                    label = "PersistenceAgent().run"
                else:
                    continue
                # A call that only exercises the wiring, with the agents patched,
                # never reaches the persistence stage. The patch may live in the
                # test itself or in the class's ``setUp``, so both are read.
                if self._uses_stub_agents(text, tree, node):
                    continue
                offenders.append(f"{path.name}:{node.lineno} ({label})")
        self.assertEqual(
            offenders,
            [],
            "these calls would write into models/: " + ", ".join(offenders),
        )

    def test_the_mlflow_state_helper_never_touches_the_configured_models(self):
        """Regression: MLflow tests stay inside their private artifact directory.

        ``tests/test_mlflow_tools.py::persisted_state`` used to set a dynamic
        ``persistence_output_dir`` attribute on the state and then call
        ``PersistenceAgent().run(state)`` without a directory. ``AgentState``
        has no such field, so the attribute was invisible and every write
        resolved to ``config.MODELS_DIR``: each Phase 10/11 suite run silently
        replaced the canonical deployment artifact and only a non-deterministic
        ``run_results.json`` betrayed it. The helper now passes the directory
        exactly like the graph node. Run the real helper the way its test
        classes do and prove ``models/`` stays byte-for-byte identical.
        """
        from tests.test_mlflow_tools import persisted_state

        before = {
            path.name: file_hash(path) for path in sorted(config.MODELS_DIR.iterdir())
        }
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        persisted_state(BINARY_DATASET, out)
        after = {
            path.name: file_hash(path) for path in sorted(config.MODELS_DIR.iterdir())
        }
        self.assertEqual(before, after, "the canonical models/ directory changed")
        self.assertTrue(artifacts_exist(out))

    @staticmethod
    def _is_direct_persistence_run(node: ast.Call) -> bool:
        """Whether the call is ``PersistenceAgent().run(...)``.

        The workflow adds no plumbing here: the agent resolves its own output
        directory, so a bare call falls back to the configured ``models/``
        directory.

        Args:
            node: The call to classify.

        Returns:
            ``True`` for ``PersistenceAgent().run(...)``.
        """
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr != "run":
            return False
        target = node.func.value
        return (
            isinstance(target, ast.Call)
            and isinstance(target.func, ast.Name)
            and target.func.id == "PersistenceAgent"
        )

    @staticmethod
    def _uses_stub_agents(text: str, tree: ast.AST, node: ast.AST) -> bool:
        """Whether the call is surrounded by stubbed agents.

        The stub patch is not always next to the call: several test classes
        install it once in ``setUp`` and then make every call in the class. Both
        the enclosing function and the enclosing class are therefore searched.

        Args:
            text: The module source.
            tree: The parsed module.
            node: The call to locate.

        Returns:
            ``True`` when the agents are patched, so no real agent runs and no
            real artifact can be written.
        """
        for scope in TestProjectArtifactIsolationGuard._enclosing_scopes(tree, node):
            segment = ast.get_source_segment(text, scope) or ""
            if "patch_agents" in segment or "use_stubs" in segment:
                return True
        return False

    @staticmethod
    def _enclosing_scopes(tree: ast.AST, node: ast.AST) -> List[Any]:
        """Return the function and the class that contain ``node``.

        Args:
            tree: The parsed module.
            node: The call to locate.

        Returns:
            The innermost enclosing function and class, in that order. A module
            level call yields only the module, which is the module's own source.
        """
        line = getattr(node, "lineno", 0)
        function: Any = None
        klass: Any = None
        for candidate in ast.walk(tree):
            if getattr(candidate, "lineno", 0) > line:
                continue
            if isinstance(candidate, ast.FunctionDef) and (
                function is None or candidate.lineno > function.lineno
            ):
                function = candidate
            if isinstance(candidate, ast.ClassDef) and (
                klass is None or candidate.lineno > klass.lineno
            ):
                klass = candidate
        scopes = [scope for scope in (function, klass) if scope is not None]
        return scopes or [tree]


if __name__ == "__main__":
    unittest.main()