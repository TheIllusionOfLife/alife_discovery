#!/usr/bin/env python3
"""Render conceptual positioning figure.

X-axis: Rule bias (random → structured)
Y-axis: Assembly complexity (size-driven → structurally non-trivial)

Plots regions for: this paper's baseline, parameter sweep, catalytic control,
config-specific catalyst, partner-specific rules, and future work.

Usage:
    uv run python scripts/render_concept_figure.py --out-dir paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render conceptual positioning figure")
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    return p.parse_args()


def render_concept_figure(out_dir: Path, *, return_fig: bool = False) -> plt.Figure | None:
    """Create the conceptual positioning figure."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Rule bias (random → structured)", fontsize=11)
    ax.set_ylabel("Assembly complexity", fontsize=11)

    # Regions
    regions = [
        {
            "label": "This paper\n(baseline)",
            "xy": (0.05, 0.05),
            "width": 0.18,
            "height": 0.15,
            "color": "#4C72B0",
            "alpha": 0.7,
        },
        {
            "label": "Parameter\nsweep",
            "xy": (0.05, 0.22),
            "width": 0.18,
            "height": 0.12,
            "color": "#4C72B0",
            "alpha": 0.5,
        },
        {
            "label": "Uniform κ",
            "xy": (0.15, 0.05),
            "width": 0.15,
            "height": 0.15,
            "color": "#DD8452",
            "alpha": 0.6,
        },
        {
            "label": "Config-\nspecific κ",
            "xy": (0.25, 0.08),
            "width": 0.15,
            "height": 0.18,
            "color": "#55A868",
            "alpha": 0.6,
        },
        {
            "label": "Partner-\nspecific",
            "xy": (0.32, 0.05),
            "width": 0.15,
            "height": 0.20,
            "color": "#C44E52",
            "alpha": 0.6,
        },
    ]

    for r in regions:
        patch = FancyBboxPatch(
            r["xy"],
            r["width"],
            r["height"],
            boxstyle="round,pad=0.02",
            facecolor=r["color"],
            alpha=r["alpha"],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        cx = r["xy"][0] + r["width"] / 2
        cy = r["xy"][1] + r["height"] / 2
        ax.text(
            cx,
            cy,
            r["label"],
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )

    # Future work region (dashed)
    future = FancyBboxPatch(
        (0.50, 0.35),
        0.45,
        0.55,
        boxstyle="round,pad=0.03",
        facecolor="#CCCCCC",
        alpha=0.3,
        edgecolor="#666666",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(future)
    ax.text(
        0.725,
        0.62,
        "Future work\n(motif-selective,\nadaptive rules,\nrule evolution)",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        style="italic",
    )

    # Boundary line
    ax.plot([0.0, 0.48], [0.38, 0.38], "k--", linewidth=1, alpha=0.5)
    ax.text(
        0.24,
        0.40,
        "boundary: size-driven ↔ structurally non-trivial",
        ha="center",
        va="bottom",
        fontsize=7,
        alpha=0.6,
    )

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["Random", "Low bias", "High bias"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["Size-driven", "Moderate", "Non-trivial"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concept_positioning.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if return_fig:
        return fig
    plt.close(fig)
    return None


def main() -> None:
    args = parse_args()
    render_concept_figure(args.out_dir)


if __name__ == "__main__":
    main()
