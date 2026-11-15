"""Session-wide safety checks for the test suite.

Two guards live here, and both exist because of real defects rather than
hypothetical ones.

Phase 9 turned model persistence on by default. Several Phase 8 integration
tests call ``run_workflow`` with the real agents on the real datasets, so every
one of them wrote into the project's own ``models/`` directory. The suite
passed, and the committed artifact was silently replaced by a worse one - a
baseline model in place of the optimized model a real run produces.

Phase 10 turned experiment tracking on in the default workflow. The same
integration tests would otherwise record a run into the user's real
``experiments/tracking/mlflow.db`` on every call - including runs over the
non-existent ``stub.csv`` fixtures that only the tests use. That fills the
dashboard's "recent runs" table with test data that is not an experiment.

A test that writes into the project's artifacts is wrong even when it passes, and
the mistake is invisible in the output. So each protected directory is hashed
before the first test and again after the last one, and any difference is
reported as a failure of the session rather than of some unrelated individual
test.

The fix belongs in the tests themselves - pass an explicit artifact directory,
turn persistence off, or pass ``enable_tracking=False`` when tracking is not what
is being tested. These guards only make sure a future omission is loud.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from tests.helpers import ProjectArtifactIsolationGuard
from utils import config

GUARDS = (
    ProjectArtifactIsolationGuard(),  # models/
    ProjectArtifactIsolationGuard(config.MLFLOW_TRACKING_DIR),  # experiments/tracking/
)


def pytest_report_header(config: "pytest.Config") -> str:
    """Tell the reader these checks are active, before any test runs.

    Args:
        config: The pytest configuration object.

    Returns:
        A one-line header naming every existing protected directory, or an
        empty string when there is nothing to protect.
    """
    del config
    protected = [guard.directory.name for guard in GUARDS if guard.directory.is_dir()]
    if not protected:
        return ""
    names = ", ".join(f"{name}/" for name in protected)
    return (
        f"project artifact isolation: {names} is protected and must be "
        "byte-identical when the session ends"
    )


def _snapshot_all() -> None:
    """Take the baseline snapshot for every protected directory."""
    for guard in GUARDS:
        guard.start()


def pytest_sessionstart(session: "pytest.Session") -> None:
    """Hash the protected directories before the first test.

    Args:
        session: The pytest session.
    """
    del session
    _snapshot_all()


def pytest_sessionfinish(
    session: "pytest.Session", exitstatus: "pytest.ExitCode"
) -> None:
    """Report any change to the protected directories.

    A modified artifact directory is escalated to a non-zero exit status. That
    is deliberate: the suite's own tests may all have passed, but the session
    still did something it must not do.

    Args:
        session: The pytest session.
        exitstatus: The status the session would otherwise report.
    """
    problems: list[str] = []
    for guard in GUARDS:
        problems.extend(guard.verify())
    if not problems:
        return
    detail = "\n".join(f"    - {problem}" for problem in problems)
    reporter = getattr(session, "config", None)
    message = (
        "a protected project directory was modified while the tests ran:\n"
        f"{detail}\n"
        "A test ran a real workflow stage without isolating its artifacts. Pass "
        "output_dir= or enable_persistence=False to run_workflow / "
        "PersistenceAgent.run, or pass enable_tracking=False when tracking is "
        "not what the test is about."
    )
    if reporter is not None:
        reporter.pluginmanager.getplugin("terminalreporter").write(
            f"\nERROR: {message}\n", red=True
        )
    else:  # pragma: no cover - only when pytest internals change
        print(f"\nERROR: {message}")
    session.exitstatus = 1


@pytest.fixture(scope="session", autouse=True)
def _project_artifacts_untouched() -> Iterator[None]:
    """Guarantee the guards' baseline exists even under ``unittest`` runners.

    The project's normal command is ``pytest -q``, where
    :func:`pytest_sessionstart` handles this. ``python -m unittest discover``
    ignores ``conftest.py`` entirely, and wiring the same check into both
    runners would leave the guard quietly absent from half of them.

    Args:
        None: The fixture takes no argument.

    Yields:
        ``None`` once the session is under way.
    """
    for guard in GUARDS:
        if not guard.before:
            guard.start()
    yield