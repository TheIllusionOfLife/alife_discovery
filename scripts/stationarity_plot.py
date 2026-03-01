#!/usr/bin/env python3
"""Stationarity Plot: mean entity size and mean AI vs step with ±1σ shading.

Reads step_timeseries.parquet, aggregates across runs by step.

Usage:
    uv run python scripts/stationarity_plot.py \
        --in-file data/experiment/logs/step_timeseries.parquet \
        --out-dir data/experiment/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stationarity Time-Series Plot")
    p.add_argument("--in-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.in_file)
    steps = table.column("step").to_numpy(zero_copy_only=False)
    mean_size = table.column("mean_entity_size").to_numpy(zero_copy_only=False)
    mean_ai = table.column("mean_assembly_index").to_numpy(zero_copy_only=False)

    unique_steps = np.sort(np.unique(steps))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel, title in [
        (axes[0], mean_size, "Mean Entity Size", "Entity Size Over Time"),
        (axes[1], mean_ai, "Mean Assembly Index", "Assembly Index Over Time"),
    ]:
        means = []
        stds = []
        for s in unique_steps:
            mask = steps == s
            vals = metric[mask]
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(unique_steps, means_arr, color="#4878d0", linewidth=1.5)
        ax.fill_between(
            unique_steps,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.2,
            color="#4878d0",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    out_path = args.out_dir / "stationarity.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
