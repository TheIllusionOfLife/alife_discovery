#!/usr/bin/env python3
"""Standalone figure generator for Experiment 3 (Shuffle-Bond Null Model Audit).

Reads entity_log_combined.parquet (must include null columns) and produces:
  - assembly_audit_dist.pdf   — overlaid histograms (observed vs null a_i)
  - assembly_audit_excess.pdf — scatter (entity_size vs a_i - null_mean)

Usage:
    uv run python scripts/plot_assembly_audit.py \\
        --in-file data/assembly_audit/entity_log_combined.parquet \\
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = [
    "assembly_index",
    "assembly_index_null_mean",
    "assembly_index_null_std",
    "entity_size",
]


def load_data(in_file: Path) -> dict[str, np.ndarray]:
    """Load entity log parquet; validate null columns are present."""
    table = pq.read_table(in_file, columns=_REQUIRED_COLUMNS)
    return {
        "assembly_index": table.column("assembly_index")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        "null_mean": table.column("assembly_index_null_mean")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        "null_std": table.column("assembly_index_null_std")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64),
        "entity_size": table.column("entity_size").to_numpy(zero_copy_only=False).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Figure 1: Overlaid histograms — observed vs null a_i
# ---------------------------------------------------------------------------


def plot_dist(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Overlaid histogram of observed a_i (blue) vs null_mean (orange) with mean lines."""
    ai = data["assembly_index"]
    null_mean = data["null_mean"]

    combined_max = max(float(ai.max()), float(null_mean.max())) if len(ai) > 0 else 1.0
    bins = np.arange(-0.5, combined_max + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        ai,
        bins=bins,
        alpha=0.6,
        color="steelblue",
        label=f"Observed $a_i$ (mean={ai.mean():.2f})",
        density=True,
    )
    ax.hist(
        null_mean,
        bins=bins,
        alpha=0.6,
        color="darkorange",
        label=f"Null mean $a_i$ (mean={null_mean.mean():.2f})",
        density=True,
    )

    ax.axvline(ai.mean(), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(null_mean.mean(), color="darkorange", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Assembly Index ($a_i$)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Observed vs null $a_i$ distribution ($N={len(ai):,}$)")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dist figure:   {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Scatter — entity_size vs (a_i - null_mean)
# ---------------------------------------------------------------------------


def plot_excess(
    data: dict[str, np.ndarray],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Scatter of (entity_size, a_i - null_mean); colour = entity_size; y=0 reference."""
    ai = data["assembly_index"]
    null_mean = data["null_mean"]
    sizes = data["entity_size"]

    excess = ai - null_mean

    fig, ax = plt.subplots(figsize=(7, 5))

    # Jitter x slightly for readability
    rng = np.random.default_rng(0)
    x_jitter = sizes + rng.uniform(-0.25, 0.25, size=len(sizes))

    sc = ax.scatter(
        x_jitter,
        excess,
        c=sizes,
        cmap="viridis",
        alpha=0.3,
        s=6,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Entity size (nodes)")

    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.2, label="$y = 0$")

    ax.set_xlabel("Entity Size (nodes)")
    ax.set_ylabel("$a_i - $ null mean")
    ax.set_title(title or "Assembly excess over null model by entity size")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved excess figure: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Experiment 3 figures from entity_log_combined.parquet"
    )
    p.add_argument(
        "--in-file",
        type=Path,
        required=True,
        help="Path to entity_log_combined.parquet (must include null columns)",
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
        help="Optional title override for all figures",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.in_file.exists():
        sys.exit(f"error: input file not found: {args.in_file}")

    print(f"Loading: {args.in_file}")
    try:
        data = load_data(args.in_file)
    except Exception as exc:
        sys.exit(
            f"error: could not load null columns from {args.in_file}.\n"
            f"Run assembly_audit.py with --n-null > 0 to generate them.\n"
            f"Details: {exc}"
        )

    n = len(data["assembly_index"])
    print(f"  {n:,} entity observations loaded")

    if n == 0:
        sys.exit("error: no entity observations in input file")

    plot_dist(data, args.out_dir / "assembly_audit_dist.pdf", args.title)
    plot_excess(data, args.out_dir / "assembly_audit_excess.pdf", args.title)
    print("Done.")


if __name__ == "__main__":
    main()
