"""Tests for scripts/render_entity_gallery.py — entity gallery (Figure 2)."""

from __future__ import annotations

from pathlib import Path
from random import Random
from unittest.mock import patch

import networkx as nx
import pytest

from alife_discovery.config.constants import ENTITY_SNAPSHOT_INTERVAL
from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import BlockWorld, generate_block_rule_table
from alife_discovery.domain.entity import (
    canonicalize_entity,
    detect_entities,
    entity_graph_hash,
)
from alife_discovery.metrics.assembly import assembly_index_exact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_graph(block_types: list[str], edges: list[tuple[int, int]]) -> nx.Graph:
    """Build a small labeled graph for testing."""
    g = nx.Graph()
    for i, bt in enumerate(block_types):
        g.add_node(i, block_type=bt)
    g.add_edges_from(edges)
    return g


# ---------------------------------------------------------------------------
# capture_entities
# ---------------------------------------------------------------------------


class TestCaptureEntities:
    """Verify run-and-capture produces entity dict with graphs."""

    def test_capture_entities_from_simulation(self) -> None:
        """A tiny sim should produce at least one entity type with a graph."""
        # Import the function under test
        from scripts.render_entity_gallery import capture_entities

        result = capture_entities(
            n_rules=2,
            seeds=1,
            steps=30,
            grid_width=10,
            grid_height=10,
            n_blocks=10,
            noise_level=0.01,
        )
        # result: dict[str, EntityRecord]
        # Should have at least 1 entity type (isolated blocks always exist)
        assert len(result) >= 1
        for h, rec in result.items():
            assert isinstance(h, str)
            assert isinstance(rec.graph, nx.Graph)
            assert rec.assembly_index >= 0
            assert rec.total_copy_count >= 1
            assert rec.entity_size >= 1

    def test_copy_counts_accumulate(self) -> None:
        """Same entity type across snapshots should accumulate copy counts."""
        from scripts.render_entity_gallery import capture_entities

        # With very low noise, isolated blocks (a_i=0, size=1) should dominate
        result = capture_entities(
            n_rules=1,
            seeds=1,
            steps=20,
            grid_width=10,
            grid_height=10,
            n_blocks=5,
            noise_level=0.0,  # no bond breaking -> bonds form and persist
        )
        # At least one entity should have copy_count > 1
        # (multiple snapshots of the same isolated blocks)
        max_copy = max(rec.total_copy_count for rec in result.values())
        assert max_copy > 1


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------


class TestSelectTopK:
    """Verify ranking by a_i × copy_count selects correct entities."""

    def test_selection_top_k_basic(self) -> None:
        from scripts.render_entity_gallery import EntityRecord, select_top_k

        records = {
            "aaa": EntityRecord(
                graph=_make_small_graph(["M"], []),
                assembly_index=0,
                total_copy_count=100,
                entity_size=1,
            ),
            "bbb": EntityRecord(
                graph=_make_small_graph(["M", "C"], [(0, 1)]),
                assembly_index=1,
                total_copy_count=50,
                entity_size=2,
            ),
            "ccc": EntityRecord(
                graph=_make_small_graph(["M", "C", "K"], [(0, 1), (1, 2)]),
                assembly_index=2,
                total_copy_count=30,
                entity_size=3,
            ),
        }
        top = select_top_k(records, k=2)
        # ccc: score=60, bbb: score=50, aaa: score=0
        assert len(top) == 2
        assert top[0][0] == "ccc"
        assert top[1][0] == "bbb"

    def test_selection_top_k_larger_than_pool(self) -> None:
        from scripts.render_entity_gallery import EntityRecord, select_top_k

        records = {
            "only": EntityRecord(
                graph=_make_small_graph(["M"], []),
                assembly_index=1,
                total_copy_count=1,
                entity_size=1,
            ),
        }
        top = select_top_k(records, k=5)
        assert len(top) == 1

    def test_selection_excludes_zero_score(self) -> None:
        """Entities with a_i=0 have score 0 and should still appear if k allows."""
        from scripts.render_entity_gallery import EntityRecord, select_top_k

        records = {
            "zero": EntityRecord(
                graph=_make_small_graph(["M"], []),
                assembly_index=0,
                total_copy_count=100,
                entity_size=1,
            ),
        }
        top = select_top_k(records, k=1)
        assert len(top) == 1  # included even with score=0


# ---------------------------------------------------------------------------
# render_entity_subplot
# ---------------------------------------------------------------------------


class TestRenderEntitySubplot:
    """Verify render_entity_subplot runs without error on known graphs."""

    def test_render_single_entity(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from scripts.render_entity_gallery import render_entity_subplot

        g = _make_small_graph(["M", "C", "K"], [(0, 1), (1, 2), (0, 2)])
        fig, ax = plt.subplots()
        render_entity_subplot(ax, g, assembly_index=3, copy_count=10, entity_hash="abc123")
        plt.close(fig)

    def test_render_single_node(self) -> None:
        """Single-node entity should render without error."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from scripts.render_entity_gallery import render_entity_subplot

        g = _make_small_graph(["K"], [])
        fig, ax = plt.subplots()
        render_entity_subplot(ax, g, assembly_index=0, copy_count=5, entity_hash="def456")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Integration: gallery PDF creation
# ---------------------------------------------------------------------------


class TestGalleryIntegration:
    """Integration test: tiny sim → gallery PDF + CSV exist on disk."""

    def test_gallery_pdf_created(self, tmp_path: Path) -> None:
        from scripts.render_entity_gallery import main_gallery

        main_gallery(
            n_rules=3,
            seeds=1,
            steps=30,
            top_k=4,
            grid_width=10,
            grid_height=10,
            n_blocks=10,
            noise_level=0.01,
            out_dir=tmp_path,
        )
        pdf_path = tmp_path / "entity_gallery.pdf"
        csv_path = tmp_path / "entity_gallery_meta.csv"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_gallery_csv_columns(self, tmp_path: Path) -> None:
        import csv

        from scripts.render_entity_gallery import main_gallery

        main_gallery(
            n_rules=2,
            seeds=1,
            steps=20,
            top_k=3,
            grid_width=10,
            grid_height=10,
            n_blocks=8,
            noise_level=0.01,
            out_dir=tmp_path,
        )
        csv_path = tmp_path / "entity_gallery_meta.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        expected_cols = {"rank", "entity_hash", "assembly_index", "copy_count", "entity_size", "score"}
        assert expected_cols.issubset(set(rows[0].keys()))
