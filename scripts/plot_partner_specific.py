#!/usr/bin/env python3
"""Figure: Partner-Specific Rule Expressiveness Comparison.

Compares standard rules (60-entry dominant-type) vs partner-specific rules
(45-entry per-partner) on key metrics: assembly index distribution, entity
size distribution, and excess rate by size class.

Usage:
    uv run python scripts/plot_partner_specific.py \\
        --std-file data/partner_specific/standard/entity_log_combined.parquet \\
        --ps-file data/partner_specific/partner_specific/entity_log_combined.parquet \\
        --out-dir paper/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pyarrow.parquet as pq

_REQUIRED_COLUMNS = [
    "assembly_index",
    "assembly_index_null_mean",
    "assembly_index_null_std",
    "assembly_index_null_pvalue",
    "entity_size",
]

_COLORS = {"Standard": "#2166ac", "Partner-Specific": "#d6604d"}
_LABELS = {"Standard": "Standard (60-entry)", "Partner-Specific": "Partner-specific (45-entry)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partner-Specific Expressiveness Figure")
    p.add_argument(
        "--std-file",
        type=Path,
        default=Path("data/partner_specific/standard/entity_log_combined.parquet"),
    )
    p.add_argument(
        "--ps-file",
        type=Path,
        default=Path("data/partner_specific/partner_specific/entity_log_combined.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    return p.parse_args()


def load_data(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=_REQUIRED_COLUMNS)
    ai = table.column("assembly_index").to_numpy(zero_copy_only=False).astype(np.float64)
    null_mean = (
        table.column("assembly_index_null_mean").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    null_std = (
        table.column("assembly_index_null_std").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    pvalue = (
        table.column("assembly_index_null_pvalue").to_numpy(zero_copy_only=False).astype(np.float64)
    )
    size = table.column("entity_size").to_numpy(zero_copy_only=False).astype(np.int64)
    return {
        "ai": ai,
        "null_mean": null_mean,
        "null_std": null_std,
        "pvalue": pvalue,
        "size": size,
        "n": len(ai),
    }


def compute_excess_by_size(data: dict[str, np.ndarray]) -> dict[int, float]:
    """Fraction of entities with empirical p < 0.05, per size class."""
    result = {}
    for s in range(1, 8):
        mask = data["size"] == s
        if mask.sum() == 0:
            continue
        excess = (data["pvalue"][mask] < 0.05).mean() * 100
        result[s] = excess
    return result


def plot_comparison(
    datasets: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> None:
    """Two-panel comparison: a_i distribution + excess by size class."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # --- Panel A: assembly index distribution (non-trivial entities) ---
    ax = axes[0]
    max_ai = int(max(d["ai"].max() for d in datasets.values()))
    bins = np.arange(-0.5, max_ai + 1.5, 1.0)
    for key, data in datasets.items():
        nontrivial = data["ai"][data["ai"] > 0]
        ax.hist(
            nontrivial,
            bins=bins,
            density=True,
            alpha=0.6,
            color=_COLORS[key],
            label=_LABELS[key],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("Assembly index $a_i$ (non-trivial entities)")
    ax.set_ylabel("Density")
    ax.set_title("A. Assembly index distribution")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # --- Panel B: excess rate by entity size ---
    ax = axes[1]
    sizes = sorted(set(s for d in datasets.values() for s in compute_excess_by_size(d).keys()))
    x = np.arange(len(sizes))
    width = 0.35
    for i, (key, data) in enumerate(datasets.items()):
        excess = compute_excess_by_size(data)
        vals = [excess.get(s, 0.0) for s in sizes]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=_LABELS[key],
            color=_COLORS[key],
            alpha=0.8,
        )
    ax.set_xlabel("Entity size (blocks)")
    ax.set_ylabel("Excess rate (%)")
    ax.set_title("B. Excess assembly by size class")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = out_dir / "fig_partner_specific.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, np.ndarray]] = {}
    for key, path in [("Standard", args.std_file), ("Partner-Specific", args.ps_file)]:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {key}")
            continue
        datasets[key] = load_data(path)
        d = datasets[key]
        total_excess = (d["pvalue"] < 0.05).sum() / d["n"] * 100
        print(
            f"{key}: n={d['n']:,}, mean_ai={d['ai'].mean():.4f}, "
            f"max_ai={d['ai'].max():.0f}, overall_excess={total_excess:.1f}%"
        )

    if len(datasets) < 2:
        print("Need both standard and partner-specific data to plot comparison.")
        return

    plot_comparison(datasets, args.out_dir)


if __name__ == "__main__":
    main()
