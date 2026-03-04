#!/usr/bin/env python3
"""Step 4: Generate all visualizations.

Reads ``results/eval.jsonl`` and ``results/layer_stats.parquet``, then
produces scatter, bar, and heatmap plots to ``results/plots/``.

Usage::

    python scripts/04_generate_plots.py [--config config/default.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.viz.plots import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])

    eval_jsonl = results_dir / "eval.jsonl"
    layer_stats = results_dir / "layer_stats.parquet"

    if not eval_jsonl.exists():
        logger.error("%s not found. Run 02_run_eval.py first.", eval_jsonl)
        return

    generate_all_plots(
        eval_jsonl=eval_jsonl,
        layer_stats_parquet=layer_stats,
        output_dir=plots_dir,
    )

    logger.info("All plots written to %s", plots_dir)


if __name__ == "__main__":
    main()
