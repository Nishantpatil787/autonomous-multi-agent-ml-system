"""Tests for ``tools.dashboard_training_tools``.

This module is the only part of the dashboard that writes files and starts the
existing workflow. The tests therefore concentrate on two promises:

* an upload is checked before anything is written, and a saved upload never
  overwrites another file;
* a run is published only after both the workflow and persistence complete, and
  a failed run leaves the previous artifact set untouched.

The workflow itself is not re-tested here. One end-to-end run proves the wiring;
the failure and publishing cases use a fake ``run_workflow`` so they are fast and
deterministic.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest import mock

import pandas as pd

from tools import dashboard_training_tools as training
from tools.dashboard_tools import load_dashboard_state
from tools.persistence_tools import artifact_paths
from utils import config

ARTIFACT_NAMES = (
    config.PERSISTENCE_MODEL_FILENAME,
    config.PERSISTENCE_METADATA_FILENAME,
    config.PERSISTENCE_RESULTS_FILENAME,
)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return a frame as UTF-8 CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


def training_frame() -> pd.DataFrame:
    """Return a small frame that clears every pre-run validation."""
    return pd.DataFrame(
        {
            "hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "group": ["a", "b"] * 5,
            "passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def write_artifacts(directory: Path, payload: bytes = b"NEW") -> None:
    """Write the three artifact files into ``directory``."""
    paths = artifact_paths(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path in (paths.model, paths.metadata, paths.results):
        path.write_bytes(payload + b":" + path.name.encode("utf-8"))


def read_artifacts(directory: Path) -> Dict[str, bytes]:
    """Return the three artifact bytes keyed by bare filename."""
    paths = artifact_paths(directory)
    return {
        path.name: path.read_bytes()
        for path in (paths.model, paths.metadata, paths.results)
    }


class UploadInspectionTests(unittest.TestCase):
    """Check 1: an upload is validated before it can be saved or trained on."""

    def test_a_valid_csv_is_summarised(self):
        frame = pd.DataFrame(
            {
                "hours": [1, 2, 3, 4, 5, 6, 7, 8],
                "group": ["a", "b"] * 4,
                "passed": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        dataset = training.inspect_upload("data.csv", csv_bytes(frame))
        self.assertEqual(dataset.filename, "data.csv")
        self.assertEqual(dataset.rows, 8)
        self.assertEqual(dataset.column_count, 3)
        self.assertEqual(dataset.columns, ["hours", "group", "passed"])
        self.assertEqual(dataset.missing_values, 0)
        self.assertEqual(dataset.duplicate_rows, 0)
        self.assertEqual(len(dataset.preview), 5)
        self.assertEqual(set(dataset.column_dtypes), {"hours", "group", "passed"})
        self.assertEqual(dataset.warnings, [])

    def test_the_extension_is_case_insensitive_and_the_directory_is_stripped(self):
        frame = training_frame()
        dataset = training.inspect_upload("../../UPPER.CSV", csv_bytes(frame))
        self.assertEqual(dataset.filename, "UPPER.CSV")

    def test_an_empty_file_is_rejected(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", b"")
        self.assertIn("empty", str(caught.exception).lower())

    def test_a_model_file_is_rejected(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("model.pkl", b"not a pickle")
        self.assertIn("not accepted", str(caught.exception))

    def test_an_oversized_file_is_rejected_before_parsing(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", csv_bytes(training_frame()), max_bytes=8)
        self.assertIn("above", str(caught.exception))
        self.assertIn("limit", str(caught.exception))

    def test_a_duplicate_header_is_rejected(self):
        payload = b"a,a\n1,2\n3,4\n5,6\n7,8\n9,10\n"
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", payload)
        self.assertIn("duplicate", str(caught.exception).lower())

    def test_a_non_utf8_header_is_rejected(self):
        payload = b"\xff,b\n1,2\n3,4\n5,6\n7,8\n9,10\n"
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", payload)
        self.assertIn("UTF-8", str(caught.exception))

    def test_an_empty_header_is_rejected(self):
        payload = b"\n1,2\n3,4\n5,6\n7,8\n9,10\n"
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", payload)
        self.assertIn("header", str(caught.exception).lower())

    def test_a_header_only_file_is_rejected(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", b"a,b\n")
        self.assertIn("no usable rows", str(caught.exception))

    def test_a_single_column_is_rejected(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", b"a\n1\n2\n3\n4\n5\n")
        self.assertIn("one feature", str(caught.exception))

    def test_too_few_rows_are_rejected(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload("data.csv", b"a,b\n1,2\n3,4\n")
        self.assertIn("at least", str(caught.exception))

    def test_a_parser_error_is_reported_as_a_validation_error(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload(
                "data.csv", b"a,b\n1,2\n3,4\n5,6\n7,8\n9,10,11\n"
            )
        self.assertIn("could not be parsed", str(caught.exception))

    def test_missing_values_and_duplicate_rows_are_reported_not_silently_fixed(self):
        frame = pd.DataFrame(
            {
                "hours": [1, 2, 3, 4, 5, 6, 7, 7],
                "group": ["a", "b", "a", "b", "a", "b", "a", "a"],
                "passed": [0, 1, 0, 1, 0, 1, 1, 1],
            }
        )
        frame.loc[0, "passed"] = None
        dataset = training.inspect_upload("data.csv", csv_bytes(frame))
        self.assertEqual(dataset.missing_values, 1)
        self.assertEqual(dataset.duplicate_rows, 1)
        body = " ".join(dataset.warnings)
        self.assertIn("missing", body.lower())
        self.assertIn("duplicate", body.lower())
        # The frame itself is untouched.
        self.assertTrue(dataset.frame["passed"].isna().any())

    def test_a_bad_body_encoding_is_reported(self):
        with self.assertRaises(training.UploadValidationError) as caught:
            training.inspect_upload(
                "data.csv", b"a,b\n1,\xff\n3,4\n5,6\n7,8\n9,10\n"
            )
        self.assertIn("UTF-8", str(caught.exception))


class FeatureSummaryTests(unittest.TestCase):
    """Check 2: the dashboard groups columns exactly as the pipeline does."""

    def test_the_target_is_excluded_from_the_feature_count(self):
        frame = training_frame()
        summary = training.summarize_features(frame, "passed")
        self.assertEqual(summary.numerical, ["hours"])
        self.assertEqual(summary.categorical, ["group"])
        self.assertEqual(summary.feature_count, 2)
        self.assertEqual(summary.unsupported, [])

    def test_without_a_target_every_column_is_counted(self):
        summary = training.summarize_features(training_frame())
        self.assertEqual(summary.feature_count, 3)


class TargetValidationTests(unittest.TestCase):
    """Check 3: an unusable target is reported before a long run starts."""

    def test_an_empty_choice_is_rejected(self):
        result = training.validate_target(training_frame(), "")
        self.assertFalse(result.ok)
        self.assertIn("Select", result.message)

    def test_an_unknown_column_is_rejected(self):
        result = training.validate_target(training_frame(), "nope")
        self.assertFalse(result.ok)
        self.assertIn("not present", result.message)

    def test_missing_labels_are_rejected(self):
        frame = training_frame()
        frame.loc[0, "passed"] = None
        result = training.validate_target(frame, "passed")
        self.assertFalse(result.ok)
        self.assertIn("missing value", result.message)

    def test_a_single_class_is_rejected(self):
        frame = pd.DataFrame({"x": range(10), "y": [1] * 10})
        result = training.validate_target(frame, "y")
        self.assertFalse(result.ok)
        self.assertIn("single distinct", result.message)

    def test_too_many_classes_are_rejected(self):
        frame = pd.DataFrame(
            {
                "x": range(105),
                "y": [index // 5 for index in range(105)],
            }
        )
        result = training.validate_target(frame, "y")
        self.assertFalse(result.ok)
        self.assertIn(str(config.MAX_CLASSES_FOR_CLASSIFICATION), result.message)

    def test_a_class_with_one_sample_is_rejected(self):
        frame = pd.DataFrame(
            {
                "x": range(11),
                "y": [0] * 5 + [1] * 5 + [2],
            }
        )
        result = training.validate_target(frame, "y")
        self.assertFalse(result.ok)
        self.assertIn(str(config.CV_MIN_FOLDS), result.message)

    def test_a_target_with_no_remaining_feature_is_rejected(self):
        frame = pd.DataFrame({"y": [0, 0, 1, 1, 1, 1]})
        result = training.validate_target(frame, "y")
        self.assertFalse(result.ok)
        self.assertIn("No usable feature", result.message)

    def test_a_balanced_binary_target_is_accepted_with_cross_validation(self):
        result = training.validate_target(training_frame(), "passed")
        self.assertTrue(result.ok)
        self.assertEqual(result.class_count, 2)
        self.assertEqual(result.min_class_count, 5)
        self.assertTrue(result.cross_validation)

    def test_a_small_class_accepts_the_run_but_skips_cross_validation(self):
        frame = pd.DataFrame(
            {
                "x": range(10),
                "y": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            }
        )
        result = training.validate_target(frame, "y")
        self.assertTrue(result.ok)
        self.assertFalse(result.cross_validation)

    def test_a_multiclass_target_is_accepted(self):
        frame = pd.DataFrame(
            {
                "x": range(15),
                "y": [0] * 5 + [1] * 5 + [2] * 5,
            }
        )
        result = training.validate_target(frame, "y")
        self.assertTrue(result.ok)
        self.assertEqual(result.class_count, 3)
        self.assertTrue(result.cross_validation)

    def test_the_target_name_is_stripped(self):
        result = training.validate_target(training_frame(), " passed ")
        self.assertTrue(result.ok)
        self.assertEqual(result.target, "passed")


class UploadSavingTests(unittest.TestCase):
    """Check 4: saving is safe, uniquely named and never overwrites."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_a_valid_upload_is_saved_byte_for_byte(self):
        payload = csv_bytes(training_frame())
        path = training.save_upload("data.csv", payload, uploads_dir=self.directory)
        self.assertEqual(path.parent, self.directory)
        self.assertTrue(path.name.startswith("data__"))
        self.assertTrue(path.name.endswith(".csv"))
        self.assertEqual(path.read_bytes(), payload)

    def test_two_uploads_with_the_same_name_get_different_files(self):
        payload = csv_bytes(training_frame())
        first = training.save_upload("data.csv", payload, uploads_dir=self.directory)
        second = training.save_upload("data.csv", payload, uploads_dir=self.directory)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), payload)
        self.assertEqual(second.read_bytes(), payload)

    def test_a_traversal_filename_cannot_escape_the_upload_directory(self):
        payload = csv_bytes(training_frame())
        path = training.save_upload(
            "../../evil.csv", payload, uploads_dir=self.directory
        )
        self.assertEqual(path.parent, self.directory)
        self.assertTrue(path.name.startswith("evil__"))
        self.assertNotIn("..", path.name)

    def test_invalid_content_is_never_written(self):
        with self.assertRaises(training.UploadValidationError):
            training.save_upload("data.csv", b"", uploads_dir=self.directory)
        self.assertEqual(list(self.directory.glob("*.csv")), [])

    def test_a_model_file_is_never_saved(self):
        with self.assertRaises(training.UploadValidationError):
            training.save_upload(
                "model.joblib", csv_bytes(training_frame()), uploads_dir=self.directory
            )
        self.assertEqual(list(self.directory.glob("*")), [])

    def test_duplicate_columns_are_rejected_before_writing(self):
        payload = b"a,a\n1,2\n3,4\n5,6\n7,8\n9,10\n"
        with self.assertRaises(training.UploadValidationError):
            training.save_upload("data.csv", payload, uploads_dir=self.directory)
        self.assertEqual(list(self.directory.glob("*.csv")), [])


class PublishTests(unittest.TestCase):
    """Check 5: the published set is complete or the previous set survives."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.source = self.root / "staging"
        self.destination = self.root / "models"
        self.source.mkdir()
        self.destination.mkdir()
        write_artifacts(self.source, b"NEW")
        self.old = {
            name: b"OLD:" + name.encode("utf-8") for name in ARTIFACT_NAMES
        }
        for name, data in self.old.items():
            (self.destination / name).write_bytes(data)

    def test_a_complete_set_is_published(self):
        training.publish_artifacts(self.source, self.destination)
        published = read_artifacts(self.destination)
        for name in ARTIFACT_NAMES:
            self.assertEqual(published[name], b"NEW:" + name.encode("utf-8"))
        self.assertEqual(
            [path.name for path in self.destination.iterdir() if path.name.startswith(".")],
            [],
        )

    def test_a_missing_source_file_keeps_the_previous_set(self):
        (self.source / config.PERSISTENCE_RESULTS_FILENAME).unlink()
        with self.assertRaises(training.WorkflowLaunchError):
            training.publish_artifacts(self.source, self.destination)
        for name, data in self.old.items():
            self.assertEqual((self.destination / name).read_bytes(), data)
        self.assertEqual(
            [path.name for path in self.destination.iterdir() if path.name.startswith(".")],
            [],
        )

    def test_a_copy_failure_keeps_the_previous_set(self):
        real_copy = shutil.copy2
        calls = {"count": 0}

        def flaky_copy(source: Any, target: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated copy failure")
            return real_copy(source, target)

        with mock.patch.object(training.shutil, "copy2", side_effect=flaky_copy):
            with self.assertRaises(training.WorkflowLaunchError):
                training.publish_artifacts(self.source, self.destination)
        for name, data in self.old.items():
            self.assertEqual((self.destination / name).read_bytes(), data)
        self.assertEqual(
            [path.name for path in self.destination.iterdir() if path.name.startswith(".")],
            [],
        )

    def test_a_replace_failure_restores_the_previous_set(self):
        real_replace = os.replace
        calls = {"count": 0}

        def flaky_replace(source: Any, target: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated replace failure")
            return real_replace(source, target)

        with mock.patch.object(training.os, "replace", side_effect=flaky_replace):
            with self.assertRaises(training.WorkflowLaunchError):
                training.publish_artifacts(self.source, self.destination)

        for name, data in self.old.items():
            self.assertEqual((self.destination / name).read_bytes(), data)
        self.assertEqual(
            [path.name for path in self.destination.iterdir() if path.name.startswith(".")],
            [],
        )

    def test_a_replace_failure_does_not_leave_new_files_when_there_was_no_old_set(self):
        empty = self.root / "empty"
        empty.mkdir()
        real_replace = os.replace
        calls = {"count": 0}

        def flaky_replace(source: Any, target: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated replace failure")
            return real_replace(source, target)

        with mock.patch.object(training.os, "replace", side_effect=flaky_replace):
            with self.assertRaises(training.WorkflowLaunchError):
                training.publish_artifacts(self.source, empty)

        self.assertEqual(
            [path.name for path in empty.iterdir() if not path.name.startswith(".")],
            [],
        )


class RunPipelineTests(unittest.TestCase):
    """Check 6: the launch path validates, reports and publishes correctly."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.csv = self.root / "data.csv"
        training_frame().to_csv(self.csv, index=False)
        self.published = self.root / "models"
        self.published.mkdir()
        self.old = {
            name: b"OLD:" + name.encode("utf-8") for name in ARTIFACT_NAMES
        }
        for name, data in self.old.items():
            (self.published / name).write_bytes(data)

    def _run(self, **overrides: Any) -> training.TrainingOutcome:
        kwargs: Dict[str, Any] = {
            "dataset_path": str(self.csv),
            "target_column": "passed",
            "enable_optimization": False,
            "n_trials": 1,
            "enable_tracking": False,
            "enable_llm": False,
            "staging_root": self.root / "staging",
        }
        kwargs.update(overrides)
        with mock.patch.object(config, "MODELS_DIR", self.published):
            return training.run_training_pipeline(**kwargs)

    def _assert_old_artifact(self) -> None:
        for name, data in self.old.items():
            self.assertEqual((self.published / name).read_bytes(), data)

    def test_a_missing_dataset_is_reported(self):
        outcome = self._run(dataset_path=str(self.root / "nope.csv"))
        self.assertFalse(outcome.ok)
        self.assertIn("no longer available", outcome.message)

    def test_an_unreadable_dataset_is_reported(self):
        broken = self.root / "broken.csv"
        broken.write_bytes(b"\xff\xfe\x00")
        outcome = self._run(dataset_path=str(broken))
        self.assertFalse(outcome.ok)
        self.assertIn("could not be read", outcome.message)

    def test_an_invalid_target_is_reported_before_the_workflow(self):
        outcome = self._run(target_column="nope")
        self.assertFalse(outcome.ok)
        self.assertIn("not present", outcome.message)
        self.assertFalse(outcome.published)

    def test_invalid_trial_count_is_reported(self):
        outcome = self._run(n_trials=0)
        self.assertFalse(outcome.ok)
        self.assertIn("between 1 and 200", outcome.message)

    def test_a_failed_workflow_does_not_publish(self):
        result = SimpleNamespace(
            workflow_status="failed",
            final_agent_state=SimpleNamespace(persistence_status="failed"),
            node_records=[],
            failed_node="",
            last_error=SimpleNamespace(
                error_type="ValueError", message="simulated failure", node="Training"
            ),
        )
        with mock.patch(
            "graph.workflow.run_workflow", return_value=result
        ):
            outcome = self._run()
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.published)
        self.assertEqual(outcome.workflow_status, "failed")
        self.assertIn("did not complete", outcome.message)
        self.assertEqual(outcome.failed_node, "Training")
        self.assertIn("simulated failure", outcome.detail)
        self._assert_old_artifact()

    def test_an_incomplete_persistence_does_not_publish(self):
        result = SimpleNamespace(
            workflow_status="completed",
            final_agent_state=SimpleNamespace(persistence_status="failed"),
            node_records=[],
            failed_node="",
            last_error=None,
        )
        with mock.patch(
            "graph.workflow.run_workflow", return_value=result
        ):
            outcome = self._run()
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.published)
        self.assertIn("was not saved", outcome.message)
        self._assert_old_artifact()

    def test_a_completed_workflow_without_artifacts_does_not_publish(self):
        result = SimpleNamespace(
            workflow_status="completed",
            final_agent_state=SimpleNamespace(persistence_status="completed"),
            node_records=[],
            failed_node="",
            last_error=None,
        )
        with mock.patch(
            "graph.workflow.run_workflow", return_value=result
        ):
            outcome = self._run()
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.published)
        self.assertIn("was not written", outcome.message)
        self._assert_old_artifact()

    def test_an_exception_from_the_workflow_is_reported(self):
        with mock.patch(
            "graph.workflow.run_workflow", side_effect=RuntimeError("boom")
        ):
            outcome = self._run()
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.published)
        self.assertIn("unexpected error", outcome.message)
        self.assertIn("RuntimeError", outcome.detail)
        self._assert_old_artifact()

    def test_a_completed_run_is_staged_and_published(self):
        def fake_workflow(**kwargs: Any) -> Any:
            staging = Path(kwargs["persistence_output_dir"])
            write_artifacts(staging, b"NEW")
            return SimpleNamespace(
                workflow_status="completed",
                final_agent_state=SimpleNamespace(persistence_status="completed"),
                node_records=[
                    SimpleNamespace(display_name="Data Analysis", status="completed")
                ],
                failed_node="",
                last_error=None,
            )

        with mock.patch("graph.workflow.run_workflow", side_effect=fake_workflow):
            outcome = self._run()

        self.assertTrue(outcome.ok)
        self.assertTrue(outcome.published)
        self.assertEqual(outcome.workflow_status, "completed")
        self.assertEqual(len(outcome.stages), 1)
        self.assertEqual(outcome.output_dir, self.published.resolve())
        published = read_artifacts(self.published)
        for name in ARTIFACT_NAMES:
            self.assertEqual(published[name], b"NEW:" + name.encode("utf-8"))
        staging = self.root / "staging"
        self.assertEqual(len(list(staging.iterdir())), 1)
        self.assertTrue(next(staging.iterdir()).is_dir())

    def test_the_real_workflow_writes_a_loadable_artifact(self):
        """One real run proves the wrapper calls the shared pipeline correctly."""
        output = self.root / "real"
        outcome = self._run(
            output_dir=output,
            enable_optimization=False,
            enable_tracking=False,
        )
        self.assertTrue(outcome.ok, outcome.message)
        self.assertTrue(outcome.published)
        self.assertEqual(outcome.workflow_status, "completed")
        for name in ARTIFACT_NAMES:
            self.assertTrue((output / name).is_file())
        state = load_dashboard_state(output)
        self.assertTrue(state.available)
        self.assertTrue(state.results_available)
        self.assertEqual(state.target_column, "passed")
        self.assertIsNotNone(state.model)
        self.assertTrue(state.model_name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
