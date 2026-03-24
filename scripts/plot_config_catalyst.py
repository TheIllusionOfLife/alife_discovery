#!/usr/bin/env python3
"""Plot config-dependent catalyst comparison (3-panel figure).

Usage:
    uv run python scripts/plot_config_catalyst.py \
        --baseline-file data/config_catalyst/baseline/entity_log_combined.parquet \
        --uniform-catalyst-file data/config_catalyst/uniform_catalyst/entity_log_combined.parquet \
        --config-catalyst-file data/config_catalyst/config_catalyst/entity_log_combined.parquet \
        --out-dir data/config_catalyst/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot config catalyst comparison")
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--uniform-catalyst-file", type=Path, required=True)
    p.add_argument("--config-catalyst-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/config_catalyst/figures"))
    return p.parse_args()


def _load_data(path: Path) -> dict[str, np.ndarray]:
    """Load entity log and extract key arrays."""
    tbl = pq.read_table(path)
    data: dict[str, np.ndarray] = {
        "assembly_index": np.array(tbl.column("assembly_index").to_pylist(), dtype=float),
        "entity_size": np.array(tbl.column("entity_size").to_pylist(), dtype=float),
    }
    if "assembly_index_null_pvalue" in tbl.column_names:
        data["pvalue"] = np.array(tbl.column("assembly_index_null_pvalue").to_pylist(), dtype=float)
    return data


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "Baseline": _load_data(args.baseline_file),
        "Uniform κ": _load_data(args.uniform_catalyst_file),
        "Config-specific κ": _load_data(args.config_catalyst_file),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Assembly index distribution
    ax = axes[0]
    for label, data in conditions.items():
        ai = data["assembly_index"]
        if len(ai) == 0:
            continue
        bins = np.arange(0, int(ai.max()) + 2) - 0.5
        ax.hist(ai, bins=bins, alpha=0.5, label=label, density=True)
    ax.set_xlabel("Assembly index")
    ax.set_ylabel("Density")
    ax.set_title("Assembly index distribution")
    ax.legend(fontsize=11)

    # Panel 2: Entity size distribution
    ax = axes[1]
    for label, data in conditions.items():
        sz = data["entity_size"]
        if len(sz) == 0:
            continue
        bins = np.arange(0, int(sz.max()) + 2) - 0.5
        ax.hist(sz, bins=bins, alpha=0.5, label=label, density=True)
    ax.set_xlabel("Entity size")
    ax.set_ylabel("Density")
    ax.set_title("Entity size distribution")
    ax.legend(fontsize=11)

    # Panel 3: Excess rate (% with p < 0.05)
    ax = axes[2]
    labels_list = []
    excess_rates = []
    for label, data in conditions.items():
        if "pvalue" in data:
            pv = data["pvalue"]
            excess = float((pv < 0.05).mean()) * 100
        else:
            excess = 0.0
        labels_list.append(label)
        excess_rates.append(excess)
    bars = ax.bar(labels_list, excess_rates, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Excess rate (%)")
    ax.set_title("% entities with p < 0.05")
    ax.set_ylim(0, max(excess_rates) * 1.3 + 1)
    for bar, rate in zip(bars, excess_rates, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()
    out_path = args.out_dir / "config_catalyst_comparison.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
