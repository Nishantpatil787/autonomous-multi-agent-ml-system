# Final Results

Measured results from Phase 11's controlled Conventional-vs-Autonomous benchmark
(`python app.py --benchmark`), re-stated here as the project's final results.
Source of truth: `reports/phase11_comparison.json` (generated 2026-10-07), CSV
`reports/phase11_comparison.csv`, and the per-dataset charts.

Fairness contract behind both tables: the same committed datasets, the same
hold-out split (`test_size=0.2`, `random_state=42`, stratified), the same five
model families, the same preprocessing defaults and the same weighted metrics.

- **Conventional**: a tools-only manual 21-config grid (no agents, no LangGraph,
  no LLM, no Optuna).
- **Autonomous**: the existing `run_workflow` graph, LLM off, Optuna on
  (20 trials per tuned model), persistence to a scratch directory.
- One experiment = one unique model configuration scored on the hold-out split.

---

## Binary Classification

Dataset: `datasets/sample_classification.csv` (123 rows, 9 columns, 7 features
used; binary target `Outcome`; SHA-256
`49ea0e2964cfce753578f23753352b453167e3f48b902a53562e1cd98f8fd23c`).

| Metric | Conventional | Autonomous | Difference |
|---|---:|---:|---:|
| Best model | Support Vector Machine | Support Vector Machine | same family |
| Best configuration | `C=10.0, kernel=linear` | `C=14.5660968…, kernel=linear` (Optuna) | — |
| Weighted F1 | 0.957780 | 0.957780 | +0.000000 |
| Accuracy | 0.958333 | 0.958333 | +0.000000 |
| Precision | 0.960938 | not recorded (Phase 6 limitation) | — |
| Recall | 0.958333 | not recorded (Phase 6 limitation) | — |
| CV F1 (mean) | 0.893303 | 0.904579 | +0.011276 |
| Runtime (s) | 13.135 | 14.307 | +1.172 |
| Experiments | 21 | 45 | +24 |

Verdict as recorded: **autonomous matched or exceeded conventional F1
(delta_f1 = +0.0000)**.

---

## Multiclass Classification

Dataset: `datasets/sample_multiclass.csv` (92 rows, 7 columns, 5 features used;
3-class target `class_label`; SHA-256
`0b70768d76af4aa27384a2c7eeb57587e85a6e775b1142c4cb81526e376539a8`).

| Metric | Conventional | Autonomous | Difference |
|---|---:|---:|---:|
| Best model | Decision Tree | Decision Tree | same family |
| Best configuration | `max_depth=2` | `max_depth=8, min_samples_split=20, min_samples_leaf=8` (Optuna) | — |
| Weighted F1 | 1.0000 | 1.0000 | +0.000000 |
| Accuracy | 1.0000 | 1.0000 | +0.000000 |
| Precision | 1.0000 | not recorded (Phase 6 limitation) | — |
| Recall | 1.0000 | not recorded (Phase 6 limitation) | — |
| CV F1 (mean) | 1.0000 | 1.0000 | +0.000000 |
| Runtime (s) | 13.397 | 41.682 | +28.285 |
| Experiments | 21 | 45 | +24 |

Verdict as recorded: **autonomous matched or exceeded conventional F1
(delta_f1 = +0.0000)**.

---

## Overall Observations

Evidence-based, from the measured benchmark:

1. **Hold-out performance matched exactly on both datasets.** The same model
   family won on each side (SVM on binary, Decision Tree on multiclass) and the
   hold-out F1 deltas are measured zeros — Optuna's `C≈14.57` lies on the same
   flat plateau as the grid's `C=10`, and the multiclass fixture is a small
   synthetic sample both sides solved perfectly.
2. **The conventional side needed fewer experiments.** 21 grid configurations
   versus 45 autonomous work items (5 baseline fits + 40 Optuna trials). The
   autonomous side explores a broader search space by design.
3. **Autonomous runtime was longer** (binary +1.172 s, multiclass +28.285 s).
   Runtime differences are workload cost, not quality; timings are single-run
   `time.perf_counter()` measurements around the ML work.
4. **Automation trade-off.** The autonomous workflow replaced the manual 21-point
   grid search with the existing multi-agent graph and an Optuna search, matched
   the conventional hold-out scores, and automated the experimentation it
   performed. It does **not** claim universal superiority — on these two fixtures
   the careful manual grid was cheaper and scored identically.
5. **The supporting cross-validated F1 differed slightly on binary**
   (0.904579 autonomous vs 0.893303 conventional, +0.011276); on multiclass both
   sides reached 1.0000. These are single-split single-seed measurements on small
   synthetic datasets; nothing here estimates real-world generalization.