"""Regression tests for the environment-backed ``OLLAMA_BASE_URL`` knob.

``utils/config.py`` reads almost every tuning knob from an environment variable
with a documented default, and ``.env.example`` advertises ``OLLAMA_BASE_URL``
as one of them. This module pins that contract: the value comes from the
environment when set, and falls back to the loopback default otherwise. It is
also the security-relevant part of the Phase 8 boundary - the loopback check
lives in ``llm/client.py`` (covered by ``test_llm_client.py``); these tests
only pin that the knob can actually be turned from the outside in the first
place.

Each test runs in a fresh interpreter so that a ``config`` module already
imported into this process cannot mask the environment override.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from tests.helpers import PROJECT_ROOT

OLLAMA_DEFAULT = "http://localhost:11434"

_PROBE = [
    sys.executable,
    "-c",
    "from utils import config; print(config.OLLAMA_BASE_URL)",
]


def _capture(env_delta: dict) -> str:
    """Run the probe in a fresh interpreter and return ``config.OLLAMA_BASE_URL``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env.pop("OLLAMA_BASE_URL", None)
    env.update(env_delta)
    proc = subprocess.run(
        _PROBE,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


class OLLamaBaseURLEnvTests(unittest.TestCase):
    """The documented environment knob for the Ollama base URL is honored."""

    def test_default_loopback_address_when_unset(self) -> None:
        """No environment variable -> the loopback default."""
        self.assertEqual(_capture({}), OLLAMA_DEFAULT)

    def test_environment_override_is_honored(self) -> None:
        """The variable advertised by .env.example actually takes effect."""
        self.assertEqual(
            _capture({"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}),
            "http://127.0.0.1:11434",
        )

    def test_a_blank_override_is_passed_through_not_silently_rewritten(self) -> None:
        """Config forwards what the environment says; cannot be reached from here.

        A blank value is handed to the client, whose loopback validation
        (``test_llm_client.py``) refuses it instead of silently connecting to the
        default host.
        """
        self.assertEqual(_capture({"OLLAMA_BASE_URL": ""}), "")


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()