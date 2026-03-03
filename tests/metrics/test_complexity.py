"""Tests for graph automorphism and typed motif census metrics."""

from __future__ import annotations

import networkx as nx

from alife_discovery.metrics.complexity import (
    graph_automorphism_count,
    typed_motif_census,
)


def make_graph(nodes_with_types: list[tuple[int, str]], edges: list[tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    for node_id, btype in nodes_with_types:
        g.add_node(node_id, block_type=btype)
    g.add_edges_from(edges)
    return g


class TestGraphAutomorphismCount:
    def test_single_node(self) -> None:
        g = make_graph([(0, "M")], [])
        assert graph_automorphism_count(g) == 1

    def test_edge_same_labels(self) -> None:
        """P_2 with same labels: 2 automorphisms (swap endpoints)."""
        g = make_graph([(0, "M"), (1, "M")], [(0, 1)])
        assert graph_automorphism_count(g) == 2

    def test_edge_different_labels(self) -> None:
        """P_2 with different labels: only identity."""
        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        assert graph_automorphism_count(g) == 1

    def test_triangle_all_same(self) -> None:
        """K_3 all same type: |Aut| = 6 (S_3)."""
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
        assert graph_automorphism_count(g) == 6

    def test_triangle_mixed(self) -> None:
        """K_3 with 2M+1C: |Aut| = 2 (swap the two M nodes)."""
        g = make_graph([(0, "M"), (1, "M"), (2, "C")], [(0, 1), (1, 2), (0, 2)])
        assert graph_automorphism_count(g) == 2

    def test_path_3_all_same(self) -> None:
        """P_3 all same: |Aut| = 2 (flip endpoints)."""
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
        assert graph_automorphism_count(g) == 2

    def test_empty_graph(self) -> None:
        g = nx.Graph()
        assert graph_automorphism_count(g) == 1


class TestTypedMotifCensus:
    def test_triangle(self) -> None:
        """Triangle has one 3-clique motif."""
        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2), (0, 2)])
        census = typed_motif_census(g)
        assert census["triangles"] == 1

    def test_path_3(self) -> None:
        """P_3: one open wedge, zero triangles."""
        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2)])
        census = typed_motif_census(g)
        assert census["triangles"] == 0
        assert census["open_wedges"] == 1

    def test_single_edge(self) -> None:
        """Single edge: no wedges, no triangles."""
        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        census = typed_motif_census(g)
        assert census["triangles"] == 0
        assert census["open_wedges"] == 0

    def test_single_node(self) -> None:
        g = make_graph([(0, "M")], [])
        census = typed_motif_census(g)
        assert census["triangles"] == 0
        assert census["open_wedges"] == 0

    def test_typed_triangle_counts(self) -> None:
        """Typed triangle labels are reported."""
        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2), (0, 2)])
        census = typed_motif_census(g)
        # Should have typed_triangles key
        assert "typed_triangles" in census
        assert len(census["typed_triangles"]) == 1

    def test_4_cycle(self) -> None:
        """C_4: 4 open wedges, 0 triangles."""
        g = make_graph(
            [(0, "M"), (1, "M"), (2, "M"), (3, "M")],
            [(0, 1), (1, 2), (2, 3), (3, 0)],
        )
        census = typed_motif_census(g)
        assert census["triangles"] == 0
        assert census["open_wedges"] == 4
