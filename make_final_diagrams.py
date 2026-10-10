"""Generate the three Phase 12 final diagrams.

Run from the project root:

    python make_final_diagrams.py

Writes (overwrites):

    reports/final_architecture.png  - agent pipeline + optional LLM layer
    reports/final_workflow.png      - compiled graph routing (Should Optimize?)
    reports/final_comparison.png    - benchmark summary from phase11_comparison.json

Uses only matplotlib (already a project dependency: requirements.txt). It reads
the real benchmark JSON; it never re-runs the benchmark and never trains.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
COMPARISON_JSON = REPORTS / "phase11_comparison.json"

# Consistent colours (matplotlib default palette).
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GREY = "#666666"
LIGHT = "#f4f4f4"


def _box(ax, x, y, w, h, title, subtitle="", fc=BLUE):
    """Draw a labelled rounded box centred at (x, y)."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor=fc,
        mutation_scale=1,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + (0.0 if not subtitle else 0.012),
        title,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white" if fc in (BLUE, GREEN) else "black",
        zorder=3,
    )
    if subtitle:
        ax.text(
            x,
            y - 0.045,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if fc in (BLUE, GREEN) else "black",
            zorder=3,
        )
    return box


def _arrow(ax, x1, y1, x2, y2, color="black", style="-|>", ls="-", lw=1.4, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def _arch_page() -> None:
    """reports/final_architecture.png"""
    fig, ax = plt.subplots(figsize=(7.6, 10.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.4)
    ax.axis("off")

    x = 5.0  # main column centre
    y = 14.0
    dy = 1.16
    w, h = 5.4, 0.72

    nodes = [
        ("Dataset", "CSV input", GREY),
        ("Data Analysis Agent", "types, quality, target", BLUE),
        ("Preprocessing Agent", "impute, encode, scale", BLUE),
        ("Model Selection Agent", "5-family registry", BLUE),
        ("Training Agent", "split + CV + ranking", BLUE),
        ("Evaluation Agent", "metrics, gap, matrix", BLUE),
        ("Optuna Optimization", "conditional, TPE sampler", ORANGE),
        ("Final Model Selection", "existing ranking rule", BLUE),
        ("Model Persistence", "joblib pipeline + metadata", BLUE),
        ("MLflow", "local experimental tracking", GREEN),
        ("Streamlit Dashboard", "read-only, sections 1-12", GREEN),
    ]
    for title, sub, color in nodes:
        _box(ax, x, y, w, h, title, sub, color)
        if y < 14.0:
            _arrow(ax, x, y + h / 2, x, y + h / 2 + dy - h)
        y -= dy

    # Optional LLM layer (interpretation only) on the right.
    lx = 8.55
    ly = 13.9
    _box(ax, lx, ly, 2.5, 0.8, "Ollama / Local LLM", "optional layer", ORANGE)
    for ty in (13.9, 8.5):
        _arrow(ax, lx - 1.25, ly, x + w / 2 + 0.05, ty, color=ORANGE, ls="--", rad=-0.25)
    ax.text(
        lx,
        ly - 0.75,
        "interpretation only -\nnever routes, never trains",
        ha="center",
        va="top",
        fontsize=6.4,
        color=GREY,
    )

    ax.text(
        5,
        0.35,
        "Deterministic ML pipeline and LangGraph routing remain authoritative;\nthe optional LLM layer only explains completed stages.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=GREY,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "final_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/final_architecture.png")


def _workflow_page() -> None:
    """reports/final_workflow.png"""
    fig, ax = plt.subplots(figsize=(7.2, 9.4), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.2)
    ax.axis("off")

    x = 5.0
    y = 12.6
    dy = 1.3
    w, h = 4.6, 0.7
    prev_bottom = y - 0.31  # bottom edge of the START box

    _box(ax, x, y, 3.0, 0.62, "START", fc=GREY)
    y -= dy
    for title, sub in [
        ("Data Analysis", ""),
        ("Preprocessing", ""),
        ("Model Selection", ""),
        ("Training", ""),
        ("Evaluation", ""),
    ]:
        _box(ax, x, y, w, h, title, sub)
        _arrow(ax, x, prev_bottom, x, y + h / 2)
        prev_bottom = y - h / 2
        y -= dy

    # Decision box after Evaluation (the "Should Optimize?" gate).
    decision_y = y + 0.22
    _arrow(ax, x, prev_bottom, x, decision_y + 0.5)
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.9, decision_y - 0.5),
            3.8,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="black",
            facecolor=ORANGE,
            zorder=2,
        )
    )
    ax.text(x, decision_y, "Optimize?", ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

    # Yes path -> Optuna -> Persistence.
    optuna_y = decision_y - 1.7
    _box(ax, 7.2, optuna_y, 4.2, 0.7, "Optuna Optimization", "20 trials / model, TPE", ORANGE)
    _arrow(ax, x + 1.9, decision_y, 5.1, optuna_y)

    persist_y = optuna_y - 1.35
    _box(ax, 5.0, persist_y, 4.6, 0.7, "Model Persistence", "joblib + metadata", BLUE)
    _arrow(ax, 7.2, optuna_y - 0.35, 5.0, persist_y + 0.35)
    ax.text(6.55, (optuna_y + persist_y) / 2, "Yes", fontsize=7.5, color=GREY)

    # No path -> Persistence (elbow around the left).
    _arrow(ax, x - 1.9, decision_y, x - 3.05, decision_y)
    _arrow(ax, x - 3.05, decision_y, x - 3.05, persist_y)
    _arrow(ax, x - 3.05, persist_y, x - 2.3, persist_y)
    ax.text(x - 3.1, decision_y + 0.18, "No", fontsize=7.5, color=GREY, ha="center")

    # END
    end_y = persist_y - 1.0
    _box(ax, x, end_y, 3.0, 0.62, "END", fc=GREY)
    _arrow(ax, x, persist_y - 0.35, x, end_y + 0.31)

    ax.text(
        x,
        0.35,
        "Failure policy (per node): one bounded retry, then error routing that records the failure\nwithout masking it. Optimize? is decided deterministically; a low gain skips the search (recorded as 'small_gain').",
        ha="center",
        va="center",
        fontsize=6.6,
        color=GREY,
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "final_workflow.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/final_workflow.png")


def _comparison_page() -> None:
    """reports/final_comparison.png from reports/phase11_comparison.json"""
    if not COMPARISON_JSON.exists():
        sys.exit(f"missing benchmark data: {COMPARISON_JSON}")
    data = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))

    ds_order = ["classification", "multiclass"]
    labels = {"classification": "Binary\n(classification)", "multiclass": "Multiclass"}

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), dpi=200)

    metric_configs = [
        ("f1_score", "Hold-out weighted F1", 0.0),
        ("runtime_seconds", "Runtime (seconds)", None),
        ("experiments", "Experiments", None),
    ]

    for ax, (key, title, _floor) in zip(axes, metric_configs):
        conv, auto = [], []
        for ds in ds_order:
            rec = data["datasets"][ds]
            conv.append(rec["conventional"][key])
            auto.append(rec["autonomous"][key])
        xx = [0, 1]
        width = 0.34
        bars_c = ax.bar([v - width / 2 for v in xx], conv, width, label="Conventional", color=BLUE)
        bars_a = ax.bar([v + width / 2 for v in xx], auto, width, label="Autonomous", color=ORANGE)
        for bars in (bars_c, bars_a):
            for b in bars:
                val = b.get_height()
                txt = f"{val:.4f}" if key == "f1_score" else f"{val:,.1f}" if key == "runtime_seconds" else f"{int(val)}"
                ax.annotate(
                    txt,
                    (b.get_x() + b.get_width() / 2, val),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.4,
                )
        ax.set_xticks(xx)
        ax.set_xticklabels([labels[d] for d in ds_order], fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if key == "f1_score":
            ax.set_ylim(0.8, 1.05)
        if key == "runtime_seconds":
            ax.legend(fontsize=7)
        if key == "experiments":
            ax.set_ylim(0, 60)

    axes[2].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Phase 11 benchmark: Conventional vs Autonomous (dataset hash-pinned)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "final_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote reports/final_comparison.png")


def main() -> int:
    _arch_page()
    _workflow_page()
    _comparison_page()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())