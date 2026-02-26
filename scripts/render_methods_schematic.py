#!/usr/bin/env python3
"""Render methods schematic: discovery protocol pipeline (Figure 1).

Produces a horizontal pipeline diagram illustrating the five stages of
the discovery protocol, using matplotlib patches and arrows.

Usage:
    uv run python scripts/render_methods_schematic.py --out-dir data/methods_schematic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Stage:
    """A single stage in the pipeline diagram."""

    label: str
    sublabel: str
    color: str


STAGES: list[Stage] = [
    Stage("Rule Sampling", "60-entry\nrule table", "#3b82f6"),
    Stage("Block World Sim", "M/C/K blocks\n+ bond dynamics", "#22c55e"),
    Stage("Entity Detection", "Bond-graph BFS\n+ canonicalize", "#f59e0b"),
    Stage("AT Measurement", r"$a_i$ (DP)" + "\n+ copy number", "#ef4444"),
    Stage("Observatory", r"($a_i$, $n_i$)" + "\nscatter", "#8b5cf6"),
]

# Layout constants
_BOX_WIDTH = 1.6
_BOX_HEIGHT = 0.7
_BOX_GAP = 0.6
_ARROW_GAP = 0.08
_SUBLABEL_OFFSET = -0.65
_FIG_DPI = 150
_BOX_FONTSIZE = 9
_SUB_FONTSIZE = 7
_TITLE_FONTSIZE = 12


def render_schematic(
    *,
    out_dir: Path,
    return_fig: bool = False,
) -> Figure | None:
    """Render the methods schematic and save as PDF.

    Parameters
    ----------
    out_dir:
        Directory to write ``methods_schematic.pdf``.
    return_fig:
        If True, return the figure instead of closing it (for testing).
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(STAGES)
    total_width = n * _BOX_WIDTH + (n - 1) * _BOX_GAP
    fig_width = total_width + 1.0
    fig_height = 2.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect("equal")
    ax.set_axis_off()

    for i, stage in enumerate(STAGES):
        x = i * (_BOX_WIDTH + _BOX_GAP)
        y = 0.0

        # Box
        box = FancyBboxPatch(
            (x, y - _BOX_HEIGHT / 2),
            _BOX_WIDTH,
            _BOX_HEIGHT,
            boxstyle="round,pad=0.08",
            facecolor=stage.color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.add_patch(box)

        # Label (centered in box)
        ax.text(
            x + _BOX_WIDTH / 2,
            y,
            stage.label,
            ha="center",
            va="center",
            fontsize=_BOX_FONTSIZE,
            fontweight="bold",
            color="white",
        )

        # Sublabel (below box)
        ax.text(
            x + _BOX_WIDTH / 2,
            y + _SUBLABEL_OFFSET,
            stage.sublabel,
            ha="center",
            va="top",
            fontsize=_SUB_FONTSIZE,
            color="#374151",
        )

        # Arrow to next box
        if i < n - 1:
            arrow = FancyArrowPatch(
                (x + _BOX_WIDTH + _ARROW_GAP, y),
                (x + _BOX_WIDTH + _BOX_GAP - _ARROW_GAP, y),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#374151",
                linewidth=1.5,
            )
            ax.add_patch(arrow)

    fig.suptitle(
        "Discovery Protocol Pipeline",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
        y=0.95,
    )
    fig.tight_layout()

    pdf_path = out_dir / "methods_schematic.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=_FIG_DPI)
    print(f"Methods schematic: {pdf_path}")

    if return_fig:
        return fig

    plt.close(fig)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render methods schematic (Figure 1)")
    p.add_argument("--out-dir", type=Path, default=Path("data/methods_schematic"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    render_schematic(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
