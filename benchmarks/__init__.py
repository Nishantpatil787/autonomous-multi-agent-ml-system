"""Phase 11: a controlled benchmark of the conventional vs autonomous workflows.

The benchmark answers one question with numbers instead of opinion: given the
*same* dataset, the *same* train/test split, the *same* preprocessing
principles and the *same* candidate model families, what do the two workflows
produce and what do they cost?

- :mod:`benchmarks.benchmark_config` holds the manual grid of configurations
  the conventional workflow explores (derived from the Phase 6 search spaces,
  not copied from them) and the shared benchmark constants.
- :mod:`benchmarks.conventional_workflow` is the "traditional manual-style ML
  process" side. It is a plain script built on the project's reusable
  sklearn tools; it never instantiates an agent, never touches LangGraph, an
  LLM or Optuna.
- :mod:`benchmarks.benchmark_runner` runs both workflows on one or both
  committed datasets, compares the measured outcomes, records the result in a
  dedicated (non-canonical) MLflow store on a best-effort basis and writes the
  comparison JSON, CSV and charts under ``reports/``.

Two properties make the comparison scientific rather than anecdotal:

1. Everything that can vary is held fixed: same dataset bytes (hash checked),
   same split function with the same seed, the same preprocessing
   configuration builder, the same metric definitions and the same candidate
   families.
2. Everything that is being measured is defined: one experiment is one unique
   model configuration trained and evaluated once, CV folds are not counted as
   experiments, and both sides are timed with ``time.perf_counter`` around the
   machine-learning work only (no report writing, no dashboard work).
"""

__version__ = "11.0.0"

__all__ = [
    "benchmark_config",
    "conventional_workflow",
    "benchmark_runner",
]