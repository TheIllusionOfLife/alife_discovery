#!/usr/bin/env python3
"""Standalone figure generator for Experiment 2 (Phase Diagram).

Reads phase_diagram.parquet and produces:
  - figures/phase_diagram_sequential.pdf   — heatmap for sequential update mode
  - figures/phase_diagram_synchronous.pdf  — heatmap for synchronous update mode
  - figures/phase_diagram_combined.pdf     — side-by-side 1×2 panel

Usage:
    uv run python scripts/plot_phase_diagram.py \\
        --in-file data/phase_diagram/phase_diagram.parquet \\
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Ordered axes values (fixed regardless of data order)
OBSERVATION_RANGES = [1, 2, 3]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(in_file: Path) -> list[dict]:
    """Load phase_diagram.parquet and return as a list of row dicts."""
    table = pq.read_table(in_file)
    return table.to_pylist()


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------


def _build_matrix(rows: list[dict], mode: str) -> np.ndarray:
    """Return a 2-D array shaped (len(NOISE_LEVELS), len(OBSERVATION_RANGES)).

    Rows  → noise_level (index 0 = lowest, last = highest)
    Cols  → observation_range (index 0 = 1, last = 3)
    Missing combos filled with NaN; duplicate rows aggregated via mean.
    """
    accum: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["update_mode"] != mode:
            continue
        obs_range = int(row["observation_range"])
        noise_val = float(row["noise_level"])
        if obs_range not in OBSERVATION_RANGES:
            logger.warning("Skipping row: unexpected observation_range=%s", obs_range)
            continue
        # Use tolerance matching to avoid float equality issues with stored values
        noise_matches = [i for i, nl in enumerate(NOISE_LEVELS) if abs(noise_val - nl) < 1e-9]
        if not noise_matches:
            logger.warning("Skipping row: unexpected noise_level=%s", noise_val)
            continue
        r_idx = OBSERVATION_RANGES.index(obs_range)
        n_idx = noise_matches[0]
        accum[(n_idx, r_idx)].append(float(row["p_discovery"]))

    matrix = np.full((len(NOISE_LEVELS), len(OBSERVATION_RANGES)), np.nan)
    for (n_idx, r_idx), vals in accum.items():
        matrix[n_idx, r_idx] = float(np.mean(vals))
    return matrix


def _annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    vmax: float,
) -> None:
    """Draw numeric annotations in each cell; use luminance-aware text colour."""
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if np.isnan(val):
                text = "N/A"
                color = "0.5"
            else:
                text = f"{val:.3f}"
                rgba = cmap(norm(val))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.5 else "black"
            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )


def plot_heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vmax: float | None = None,
) -> plt.cm.ScalarMappable:
    """Draw a single heatmap panel on *ax*; return the ScalarMappable for colorbar."""
    if vmax is None:
        vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    vmax = max(vmax, 1e-6)  # avoid zero-range colormap

    # imshow: rows=NOISE_LEVELS (index 0 at top by default), flip with origin='lower'
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    _annotate_cells(ax, matrix, vmax)

    ax.set_xticks(range(len(OBSERVATION_RANGES)))
    ax.set_xticklabels([str(v) for v in OBSERVATION_RANGES])
    ax.set_xlabel("Observation Range")

    ax.set_yticks(range(len(NOISE_LEVELS)))
    ax.set_yticklabels([str(v) for v in NOISE_LEVELS])
    ax.set_ylabel("Noise Level")

    ax.set_title(title)
    return im


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def plot_single(
    rows: list[dict],
    mode: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Generate a standalone PDF for one update_mode."""
    matrix = _build_matrix(rows, mode)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = plot_heatmap_panel(ax, matrix, title or f"P(discovery) — {mode}")
    fig.colorbar(im, ax=ax, label="P(discovery)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {mode} heatmap:   {out_path}")


def plot_combined(
    rows: list[dict],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Generate a 1×2 combined PDF with one panel per update_mode."""
    modes = ["sequential", "synchronous"]
    matrices = [_build_matrix(rows, m) for m in modes]

    # Shared colour scale across both panels for direct comparability
    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices])
    vmax = float(all_vals.max()) if len(all_vals) > 0 else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    im = None
    for ax, mode, matrix in zip(axes, modes, matrices, strict=True):
        im = plot_heatmap_panel(ax, matrix, mode.capitalize(), vmax=vmax)

    axes[0].set_ylabel("Noise Level")
    axes[1].set_ylabel("")

    if im is None:
        raise RuntimeError("plot_combined: no panel was rendered (modes list may be empty)")
    fig.colorbar(im, ax=axes.tolist(), label="P(discovery)")
    if title:
        fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined heatmap: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Experiment 2 phase diagram figures from phase_diagram.parquet"
    )
    p.add_argument(
        "--in-file",
        type=Path,
        required=True,
        help="Path to phase_diagram.parquet",
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
        help="Optional suptitle for the combined figure",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.in_file.exists():
        sys.exit(f"error: input file not found: {args.in_file}")

    print(f"Loading: {args.in_file}")
    rows = load_data(args.in_file)
    required_cols = {"update_mode", "observation_range", "noise_level", "p_discovery"}
    if rows and not required_cols.issubset(rows[0].keys()):
        missing = required_cols - rows[0].keys()
        sys.exit(f"error: input file missing required columns: {missing}")
    modes = sorted({r["update_mode"] for r in rows})
    print(f"  {len(rows)} rows loaded  (modes: {modes})")

    plot_single(rows, "sequential", args.out_dir / "phase_diagram_sequential.pdf")
    plot_single(rows, "synchronous", args.out_dir / "phase_diagram_synchronous.pdf")
    plot_combined(rows, args.out_dir / "phase_diagram_combined.pdf", args.title)
    print("Done.")


if __name__ == "__main__":
    main()
