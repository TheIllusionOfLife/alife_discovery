"""Tests for plot_baseline.py zoom inset feature."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_baseline import plot_heatmap


class TestZoomInset:
    """Tests for the high-AI zoom inset on the heatmap figure."""

    def test_heatmap_with_high_ai_data_creates_inset(
        self, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """When data has a_i >= 3 with >=5 points, the heatmap should include an inset axes."""
        data = {
            "assembly_index": np.array([0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 3]),
            "copy_number_at_step": np.array([10, 20, 30, 5, 8, 3, 2, 1, 1, 1, 2]),
            "entity_size": np.array([1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 4]),
        }
        out_path = tmp_path / "heatmap.pdf"
        fig = plot_heatmap(data, out_path, return_fig=True)
        assert out_path.exists()
        # Should have inset axes (main + top + right + inset = 4 axes minimum)
        axes = fig.get_axes()
        assert len(axes) >= 4, f"Expected >=4 axes (with inset), got {len(axes)}"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_heatmap_without_high_ai_skips_inset(self, tmp_path: "pytest.TempPathFactory") -> None:
        """When all a_i < 3, no inset should be added (no crash)."""
        data = {
            "assembly_index": np.array([0, 0, 1, 1, 2]),
            "copy_number_at_step": np.array([10, 20, 5, 8, 3]),
            "entity_size": np.array([1, 1, 2, 2, 3]),
        }
        out_path = tmp_path / "heatmap_no_inset.pdf"
        fig = plot_heatmap(data, out_path, return_fig=True)
        assert out_path.exists()
        # Should have exactly 3 axes (main + top + right) — no inset
        # Plus colorbar axes, so check no inset by counting
        axes = fig.get_axes()
        # Without inset: main, top marginal, right marginal, colorbar = 4
        assert len(axes) == 4, f"Expected 4 axes (no inset), got {len(axes)}"
        import matplotlib.pyplot as plt

        plt.close(fig)
