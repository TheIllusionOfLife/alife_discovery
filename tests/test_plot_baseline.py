"""Tests for plot_baseline.py zoom inset feature."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_baseline import plot_heatmap


class TestZoomInset:
    """Tests for the high-AI zoom inset on the heatmap figure."""

    def test_heatmap_with_high_ai_data_creates_inset(self, tmp_path: Path) -> None:
        """When data has a_i >= 3 with >=5 points, the heatmap should include an inset axes."""
        data = {
            "assembly_index": np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 3]),
            "copy_number_at_step": np.array([10, 20, 30, 5, 8, 3, 2, 1, 1, 1, 2]),
            "entity_size": np.array([1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 4]),
        }
        out_path = tmp_path / "heatmap.pdf"
        fig = plot_heatmap(data, out_path, return_fig=True)
        assert out_path.exists()
        # Inset is created via ax.inset_axes() and appears as a child of main axes
        main_ax = fig.get_axes()[0]
        main_area = main_ax.get_position().width * main_ax.get_position().height
        inset_children = [
            c
            for c in main_ax.child_axes
            if c.get_position().width * c.get_position().height < main_area
        ]
        assert len(inset_children) >= 1, "Expected an inset axes as child of main axes"
        plt.close(fig)

    def test_heatmap_without_high_ai_skips_inset(self, tmp_path: Path) -> None:
        """When all a_i < 3, no inset should be added (no crash)."""
        data = {
            "assembly_index": np.array([0, 0, 1, 1, 2]),
            "copy_number_at_step": np.array([10, 20, 5, 8, 3]),
            "entity_size": np.array([1, 1, 2, 2, 3]),
        }
        out_path = tmp_path / "heatmap_no_inset.pdf"
        fig = plot_heatmap(data, out_path, return_fig=True)
        assert out_path.exists()
        # No inset: main axes should have no child axes
        main_ax = fig.get_axes()[0]
        assert len(main_ax.child_axes) == 0, "Expected no inset axes"
        plt.close(fig)
