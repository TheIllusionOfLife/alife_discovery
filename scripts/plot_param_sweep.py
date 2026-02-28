#!/usr/bin/env python3
"""Plot parameter sweep results as multi-panel heatmaps.

Left: density × grid (drift=1.0) showing pct_excess and max_entity_size.
Right: density × drift (20×20 grid) showing same metrics.

Usage:
    uv run python scripts/plot_param_sweep.py \
        --in-file data/param_sweep/param_sweep_summary.parquet \
        --out-dir data/param_sweep/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Parameter Sweep Results")
    p.add_argument("--in-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/param_sweep/figures"))
    return p.parse_args()


def _heatmap(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    fmt: str = ".1f",
) -> None:
    """Draw an annotated heatmap on the given axes."""
    x_unique = np.sort(np.unique(x_vals))
    y_unique = np.sort(np.unique(y_vals))
    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    for xi, yi, zi in zip(x_vals, y_vals, z_vals, strict=True):
        row = np.searchsorted(y_unique, yi)
        col = np.searchsorted(x_unique, xi)
        grid[row, col] = zi
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(x_unique)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v, float) else str(v) for v in x_unique])
    ax.set_yticks(range(len(y_unique)))
    ax.set_yticklabels([f"{v:.3f}" for v in y_unique])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate cells
    for r in range(len(y_unique)):
        for c in range(len(x_unique)):
            val = grid[r, c]
            if not np.isnan(val):
                lum = 0.299 * im.cmap(im.norm(val))[0] + 0.587 * im.cmap(im.norm(val))[1]
                color = "white" if lum < 0.5 else "black"
                ax.text(c, r, f"{val:{fmt}}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.in_file)
    df = {col: table.column(col).to_numpy(zero_copy_only=False) for col in table.column_names}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: density × grid (drift=1.0)
    mask_drift1 = df["drift_probability"] == 1.0
    grid_labels = df["grid_width"][mask_drift1].astype(float)
    density = df["density_ratio"][mask_drift1]
    pct_excess = df["pct_excess_p05"][mask_drift1]
    _heatmap(
        axes[0],
        grid_labels,
        density,
        pct_excess,
        xlabel="Grid Width",
        ylabel="Density Ratio",
        title="% Excess (p<0.05) by Grid × Density",
    )

    # Right: density × drift (20×20 grid)
    mask_20 = df["grid_width"] == 20
    drift = df["drift_probability"][mask_20]
    density_20 = df["density_ratio"][mask_20]
    pct_excess_20 = df["pct_excess_p05"][mask_20]
    _heatmap(
        axes[1],
        drift,
        density_20,
        pct_excess_20,
        xlabel="Drift Probability",
        ylabel="Density Ratio",
        title="% Excess (p<0.05) by Drift × Density (20×20)",
    )

    plt.tight_layout()
    out_path = args.out_dir / "param_sweep.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
