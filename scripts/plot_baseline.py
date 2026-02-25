#!/usr/bin/env python3
"""Standalone figure generator for Experiment 1 (Discovery Baseline).

Reads entity_log_combined.parquet and produces:
  - experiment1_ai_cn_scatter.pdf  — scatter (a_i vs n_i, coloured by entity size)
  - experiment1_ai_cn_heatmap.pdf  — 2-D histogram with log-colorscale + marginals
  - experiment1_size_dist.pdf      — entity size bar chart with MAX_ENTITY_SIZE reference

Usage:
    uv run python scripts/plot_baseline.py \\
        --in-file data/experiment1/entity_log_combined.parquet \\
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from alife_discovery.config.constants import MAX_ENTITY_SIZE

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(in_file: Path) -> dict[str, np.ndarray]:
    """Load entity log parquet; return numpy arrays for the three key columns."""
    table = pq.read_table(in_file, columns=["assembly_index", "copy_number_at_step", "entity_size"])
    return {
        "assembly_index": table.column("assembly_index")
        .to_numpy(zero_copy_only=False)
        .astype(np.int64),
        "copy_number_at_step": table.column("copy_number_at_step")
        .to_numpy(zero_copy_only=False)
        .astype(np.int64),
        "entity_size": table.column("entity_size").to_numpy(zero_copy_only=False).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Figure 1: Scatter — (a_i, n_i) coloured by entity size
# ---------------------------------------------------------------------------


def plot_scatter(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Scatter: x = assembly_index, y = copy_number_at_step (log scale), colour = entity_size."""
    ai = data["assembly_index"]
    cn = data["copy_number_at_step"]
    sz = data["entity_size"]

    # Clip to >= 1 so log y-axis never receives non-positive values
    cn_plot = np.maximum(cn, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        ai,
        cn_plot,
        c=sz,
        cmap="viridis",
        alpha=0.4,
        s=8,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Entity size (nodes)")

    ax.set_xlabel("Assembly Index ($a_i$)")
    ax.set_ylabel("Copy Number ($n_i$)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)
    ax.set_xlim(left=0)
    ax.set_title(title or f"Entity discovery landscape ($N={len(ai):,}$ observations)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter:          {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap — 2-D histogram with log-colorscale + marginal histograms
# ---------------------------------------------------------------------------


def plot_heatmap(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> None:
    """2-D histogram of (a_i, log10(n_i)) with log-colorscale and marginal counts."""
    ai = data["assembly_index"]
    cn = data["copy_number_at_step"]

    # Work in log10 space for copy number to avoid extreme axis range
    log_cn = np.log10(np.maximum(cn, 1).astype(float))

    ai_max = int(ai.max()) if len(ai) > 0 else 10
    log_cn_max = float(log_cn.max()) if len(log_cn) > 0 else 2.0

    # At least 2 bins per axis; align AI bins to integers
    ai_bins = np.arange(0, ai_max + 2) - 0.5
    n_cn_bins = max(int(log_cn_max * 10) + 2, 10)
    cn_bins = np.linspace(0.0, max(log_cn_max + 0.1, 0.5), n_cn_bins)

    counts, xedges, yedges = np.histogram2d(ai, log_cn, bins=[ai_bins, cn_bins])

    # Build figure with marginals using GridSpec
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.05,
        wspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main heatmap — log colour scale (vmin=1 avoids log(0))
    im = ax_main.pcolormesh(
        xedges,
        yedges,
        counts.T,
        norm=mcolors.LogNorm(vmin=1, vmax=max(float(counts.max()), 1.0)),
        cmap="plasma",
    )
    ax_main.set_xlabel("Assembly Index ($a_i$)")
    ax_main.set_ylabel(r"$\log_{10}$(Copy Number $n_i$)")

    # Top marginal — assembly index distribution
    ai_marginal, _ = np.histogram(ai, bins=xedges)
    ax_top.bar(
        xedges[:-1] + 0.5,
        ai_marginal,
        width=np.diff(xedges),
        color="steelblue",
        alpha=0.8,
    )
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title(title or f"Assembly-copy-number joint distribution ($N={len(ai):,}$)")

    # Right marginal — log copy-number distribution
    cn_marginal, _ = np.histogram(log_cn, bins=yedges)
    ax_right.barh(
        yedges[:-1] + np.diff(yedges) / 2,
        cn_marginal,
        height=np.diff(yedges),
        color="steelblue",
        alpha=0.8,
    )
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Colorbar next to right marginal
    cbar = fig.colorbar(im, ax=[ax_main, ax_right], location="right", fraction=0.04, pad=0.02)
    cbar.set_label("Count (log scale)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap:          {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Entity size distribution bar chart
# ---------------------------------------------------------------------------


def plot_size_dist(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Bar chart of entity_size observation counts with MAX_ENTITY_SIZE reference line."""
    sz = data["entity_size"]
    sizes, counts = np.unique(sz, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sizes, counts, color="teal", alpha=0.85, zorder=2)
    ax.axvline(
        MAX_ENTITY_SIZE,
        color="crimson",
        linestyle="--",
        linewidth=1.4,
        label=f"MAX_ENTITY_SIZE = {MAX_ENTITY_SIZE}",
        zorder=3,
    )
    ax.set_xlabel("Entity Size (nodes)")
    ax.set_ylabel("Observation Count")
    ax.set_xticks(sizes)
    ax.legend()
    ax.set_title(title or "Entity size distribution")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved size distribution: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Experiment 1 figures from entity_log_combined.parquet"
    )
    p.add_argument(
        "--in-file",
        type=Path,
        required=True,
        help="Path to entity_log_combined.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for PDFs (default: figures/)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for all figures",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.in_file.exists():
        sys.exit(f"error: input file not found: {args.in_file}")

    print(f"Loading: {args.in_file}")
    data = load_data(args.in_file)
    n = len(data["assembly_index"])
    print(f"  {n:,} entity observations loaded")

    if n == 0:
        sys.exit("error: no entity observations in input file")

    plot_scatter(data, args.out_dir / "experiment1_ai_cn_scatter.pdf", args.title)
    plot_heatmap(data, args.out_dir / "experiment1_ai_cn_heatmap.pdf", args.title)
    plot_size_dist(data, args.out_dir / "experiment1_size_dist.pdf", args.title)
    print("Done.")


if __name__ == "__main__":
    main()
