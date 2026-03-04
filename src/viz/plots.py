"""Visualization module for the quantization comparative study.

Generates four families of plots:

1. **Scatter: accuracy vs bandwidth** (bytes_per_param_actual vs score)
2. **Scatter: accuracy vs deployment metrics** (tok/s, ms/tok, peak VRAM)
3. **Bar: MLP vs attention sensitivity** from ablation results
4. **Heatmap: per-layer MSE** by module family
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# 1.  Accuracy vs Bandwidth
# ---------------------------------------------------------------------------

def plot_accuracy_vs_bandwidth(
    eval_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Scatter: composite score vs actual bytes per parameter."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in eval_df["method"].unique():
        sub = eval_df[eval_df["method"] == method]
        ax.scatter(
            sub["bytes_per_param_actual"],
            sub["score"],
            label=method.upper(),
            s=80,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["variant"],
                (row["bytes_per_param_actual"], row["score"]),
                fontsize=7,
                ha="left",
                va="bottom",
            )

    ax.set_xlabel("Effective bytes / parameter")
    ax.set_ylabel("Composite accuracy score")
    ax.set_title("Accuracy vs. Bandwidth (Sweet-Spot Plot)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_bandwidth.png", dpi=200)
    plt.close(fig)
    logger.info("Saved accuracy_vs_bandwidth.png")


# ---------------------------------------------------------------------------
# 2.  Accuracy vs Deployment Metrics
# ---------------------------------------------------------------------------

def plot_accuracy_vs_deployment(
    eval_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Scatter panels: score vs tok/s, ms/tok, peak VRAM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("tok_s", "Tokens / sec"),
        ("ms_per_token", "ms / token"),
        ("peak_vram_gb", "Peak VRAM (GB)"),
    ]
    available = [m for m in metrics if m[0] in eval_df.columns]
    if not available:
        logger.warning("No deployment metrics found in eval data; skipping.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        for method in eval_df["method"].unique():
            sub = eval_df[eval_df["method"] == method]
            ax.scatter(
                sub[col], sub["score"],
                label=method.upper(), s=80, edgecolors="k", linewidths=0.5,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Composite score")
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy vs. Deployment Metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_deployment.png", dpi=200)
    plt.close(fig)
    logger.info("Saved accuracy_vs_deployment.png")


# ---------------------------------------------------------------------------
# 3.  MLP vs Attention Sensitivity (bar chart)
# ---------------------------------------------------------------------------

def plot_ablation_sensitivity(
    eval_df: pd.DataFrame,
    output_dir: str | Path,
    task_col: str = "score",
) -> None:
    """Grouped bar chart showing score drop when quantizing only MLP or attn.

    For each method: ``delta_attn = score(full) - score(attn_only)``
    and ``delta_mlp = score(full) - score(mlp_only)``.
    A *larger* delta means that family is *more sensitive* to quantization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for method in eval_df["method"].unique():
        sub = eval_df[eval_df["method"] == method]
        full = sub.loc[sub["variant"] == "full_quant", task_col]
        attn = sub.loc[sub["variant"] == "attn_only_quant", task_col]
        mlp = sub.loc[sub["variant"] == "mlp_only_quant", task_col]
        if full.empty:
            continue
        full_val = full.values[0]
        if not attn.empty:
            rows.append({
                "method": method.upper(),
                "family": "Attn projections",
                "delta": full_val - attn.values[0],
            })
        if not mlp.empty:
            rows.append({
                "method": method.upper(),
                "family": "MLP projections",
                "delta": full_val - mlp.values[0],
            })

    if not rows:
        logger.warning("Not enough ablation data for sensitivity plot.")
        return

    bar_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=bar_df, x="method", y="delta", hue="family", ax=ax)
    ax.set_ylabel(f"Score drop (full_quant − <family>_only)")
    ax.set_xlabel("Quantization method")
    ax.set_title("Module-Family Sensitivity: MLP vs Attention")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_sensitivity.png", dpi=200)
    plt.close(fig)
    logger.info("Saved ablation_sensitivity.png")


# ---------------------------------------------------------------------------
# 4.  Per-Layer MSE Heatmap
# ---------------------------------------------------------------------------

def plot_layer_mse_heatmap(
    layer_stats_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Heatmap of per-layer MSE, faceted by method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(str(layer_stats_path))

    methods = df["method"].unique()
    for method in methods:
        sub = df[df["method"] == method]
        pivot = sub.pivot_table(
            index="layer_idx",
            columns="module_family",
            values="mse",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, max(8, len(pivot) * 0.3)))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.3,
            cbar_kws={"label": "MSE"},
        )
        ax.set_title(f"Per-Layer MSE — {method.upper()}")
        ax.set_ylabel("Transformer layer index")
        ax.set_xlabel("Module family")
        fig.tight_layout()
        fig.savefig(output_dir / f"layer_mse_{method}.png", dpi=200)
        plt.close(fig)
        logger.info("Saved layer_mse_%s.png", method)


# ---------------------------------------------------------------------------
# Convenience: generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    eval_jsonl: str | Path,
    layer_stats_parquet: str | Path,
    output_dir: str | Path,
) -> None:
    """Read results files and produce every plot."""
    eval_df = pd.read_json(str(eval_jsonl), lines=True)

    plot_accuracy_vs_bandwidth(eval_df, output_dir)
    plot_accuracy_vs_deployment(eval_df, output_dir)
    plot_ablation_sensitivity(eval_df, output_dir)
    plot_layer_mse_heatmap(layer_stats_parquet, output_dir)

    logger.info("All plots saved to %s", output_dir)
