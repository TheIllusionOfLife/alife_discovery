#!/usr/bin/env python3
"""Plot catalytic K positive control comparison.

Produces catalytic_comparison.pdf with side-by-side baseline vs catalytic
assembly distributions and excess-vs-size scatter.

Usage:
    uv run python scripts/plot_catalytic_control.py \
        --baseline-file data/catalytic_control/baseline/entity_log_combined.parquet \
        --catalytic-file data/catalytic_control/catalytic/entity_log_combined.parquet \
        --out-dir data/catalytic_control/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Catalytic K Control Comparison")
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--catalytic-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/catalytic_control/figures"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pq.read_table(args.baseline_file)
    catalytic = pq.read_table(args.catalytic_file)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Assembly index distribution comparison
    ax = axes[0]
    for label, tbl, color in [
        ("Baseline", baseline, "#4878d0"),
        ("Catalytic", catalytic, "#ee854a"),
    ]:
        ai = tbl.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
        bins = np.arange(0, int(ai.max()) + 2) - 0.5
        ax.hist(ai, bins=bins, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel("Assembly Index")
    ax.set_ylabel("Density")
    ax.set_title("Assembly Index Distribution")
    ax.legend()

    # Panel 2: Excess vs entity size (scatter)
    ax = axes[1]
    for label, tbl, color, marker in [
        ("Baseline", baseline, "#4878d0", "o"),
        ("Catalytic", catalytic, "#ee854a", "s"),
    ]:
        if "assembly_index_null_mean" not in tbl.column_names:
            continue
        ai = tbl.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
        nm = tbl.column("assembly_index_null_mean").to_numpy(zero_copy_only=False)
        sz = tbl.column("entity_size").to_numpy(zero_copy_only=False).astype(float)
        excess = ai - nm
        # Aggregate by size
        unique_sizes = np.unique(sz)
        mean_excess = [float(excess[sz == s].mean()) for s in unique_sizes]
        ax.scatter(unique_sizes, mean_excess, label=label, color=color, marker=marker, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Entity Size")
    ax.set_ylabel("Mean Excess (a_i − null mean)")
    ax.set_title("Excess vs Entity Size")
    ax.legend()

    plt.tight_layout()
    out_path = args.out_dir / "catalytic_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
