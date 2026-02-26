"""Tests for scripts/render_methods_schematic.py — methods schematic (Figure 1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Integration: schematic PDF creation
# ---------------------------------------------------------------------------


class TestSchematicIntegration:
    """Integration test: methods schematic PDF exists on disk."""

    def test_schematic_pdf_created(self, tmp_path: Path) -> None:
        from scripts.render_methods_schematic import render_schematic

        render_schematic(out_dir=tmp_path)

        pdf_path = tmp_path / "methods_schematic.pdf"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_schematic_stages_present(self, tmp_path: Path) -> None:
        """All pipeline stage labels appear in the rendered figure."""
        from scripts.render_methods_schematic import STAGES, render_schematic

        fig = render_schematic(out_dir=tmp_path, return_fig=True)
        assert fig is not None

        # Collect all text from the figure
        texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        all_text = " ".join(texts)
        plt.close(fig)

        for stage in STAGES:
            assert stage.label in all_text, f"Stage label '{stage.label}' not found in figure"
