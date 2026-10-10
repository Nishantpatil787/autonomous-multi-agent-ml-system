"""Test package for the autonomous multi-agent ML system.

Run the whole suite from the project root:

    python -m unittest discover -s tests -t .

Logging is reduced to WARNING level here so the test output stays readable.
Errors and warnings are still printed. Set LOG_LEVEL=INFO before running the
tests when the full agent log is needed.
"""

import os

os.environ.setdefault("LOG_LEVEL", "WARNING")