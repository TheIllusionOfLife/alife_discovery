#!/usr/bin/env python3
"""Generate consolidated sensitivity analysis figure (Fig 6).

Combines parameter sweep heatmaps (left) with catalytic comparison (right).

Usage:
    uv run python scripts/plot_sensitivity.py \
        --sweep-file data/param_sweep_v2/param_sweep_summary.parquet \
        --baseline-file data/catalytic_v2/baseline/entity_log_combined.parquet \
        --catalytic-file data/catalytic_v2/catalytic/entity_log_combined.parquet \
        --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidated Sensitivity Figure")
    p.add_argument("--sweep-file", type=Path, required=True)
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--catalytic-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
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
    x_unique = np.sort(np.unique(x_vals))
    y_unique = np.sort(np.unique(y_vals))
    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    for xi, yi, zi in zip(x_vals, y_vals, z_vals, strict=True):
        row = np.searchsorted(y_unique, yi)
        col = np.searchsorted(x_unique, xi)
        grid[row, col] = zi
    if grid.size == 0:
        ax.set_title(title, fontsize=9)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(x_unique)))
    ax.set_xticklabels(
        [f"{v:.2f}" if isinstance(v, float) else str(v) for v in x_unique], fontsize=7
    )
    ax.set_yticks(range(len(y_unique)))
    ax.set_yticklabels([f"{v:.3f}" for v in y_unique], fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    for r in range(len(y_unique)):
        for c in range(len(x_unique)):
            val = grid[r, c]
            if not np.isnan(val):
                rgba = im.cmap(im.norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.5 else "black"
                ax.text(c, r, f"{val:{fmt}}", ha="center", va="center", fontsize=7, color=color)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for f in (args.sweep_file, args.baseline_file, args.catalytic_file):
        if not f.exists():
            sys.exit(f"Error: file not found: {f}")

    # Load sweep data
    sweep = pq.read_table(args.sweep_file)
    df = {col: sweep.column(col).to_numpy(zero_copy_only=False) for col in sweep.column_names}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Density × Grid (drift=1.0)
    mask_drift1 = np.isclose(df["drift_probability"], 1.0)
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
        title="% Excess (p<0.05)\nGrid × Density (drift=1.0)",
    )

    # Panel 2: Density × Drift (20×20)
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
        title="% Excess (p<0.05)\nDrift × Density (20×20)",
    )

    # Panel 3: Catalytic comparison
    ax = axes[2]
    baseline = pq.read_table(args.baseline_file)
    catalytic = pq.read_table(args.catalytic_file)

    for label, tbl, color in [
        ("Baseline", baseline, "#4878d0"),
        (r"Catalytic ($\kappa$=3)", catalytic, "#ee854a"),
    ]:
        ai = tbl.column("assembly_index").to_numpy(zero_copy_only=False).astype(float)
        if len(ai) == 0:
            continue
        bins = np.arange(0, int(ai.max()) + 2) - 0.5
        ax.hist(ai, bins=bins, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel("Assembly Index", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("Catalytic K Control", fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = args.out_dir / "fig6_sensitivity.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
