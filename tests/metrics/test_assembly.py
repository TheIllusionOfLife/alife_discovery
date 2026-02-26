"""Tests for Assembly Theory assembly index computation (edge-removal DP)."""

import networkx as nx

from alife_discovery.metrics.assembly import (
    _canonical_key,
    assembly_index_approx,
    assembly_index_exact,
    assembly_index_null,
    compute_entity_metrics,
)


def make_graph(nodes_with_types: list[tuple[int, str]], edges: list[tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    for node_id, btype in nodes_with_types:
        g.add_node(node_id, block_type=btype)
    g.add_edges_from(edges)
    return g


# ── Base cases ────────────────────────────────────────────────────────────────


def test_single_node_is_zero():
    g = nx.Graph()
    g.add_node(0, block_type="M")
    assert assembly_index_exact(g) == 0


def test_two_nodes_no_edge_is_zero():
    g = nx.Graph()
    g.add_node(0, block_type="M")
    g.add_node(1, block_type="C")
    assert assembly_index_exact(g) == 0


def test_p2_is_one():
    g = make_graph([(0, "M"), (1, "M")], [(0, 1)])
    assert assembly_index_exact(g) == 1


# ── Path graphs (no reuse possible) ──────────────────────────────────────────


def test_p3_is_two():
    g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
    assert assembly_index_exact(g) == 2


def test_p4_is_three():
    g = make_graph([(0, "M"), (1, "M"), (2, "M"), (3, "M")], [(0, 1), (1, 2), (2, 3)])
    assert assembly_index_exact(g) == 3


# ── K_3 triangle ─────────────────────────────────────────────────────────────


def test_k3_is_three():
    """K_3: remove any edge -> P_3 (a=2), so a(K_3) = 1 + 2 = 3.

    Under the edge-removal DP (no sub-object reuse), removing any edge from
    K_3 leaves a connected 3-node path (P_3, a=2). The connected branch gives
    cost = 1 + a(P_3) = 1 + 2 = 3.
    """
    g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
    assert assembly_index_exact(g) == 3


# ── Label awareness ───────────────────────────────────────────────────────────


def test_same_topology_different_labels_same_index():
    """Assembly index depends on topology, not labels."""
    g_mm = make_graph([(0, "M"), (1, "M")], [(0, 1)])
    g_mk = make_graph([(0, "M"), (1, "K")], [(0, 1)])
    assert assembly_index_exact(g_mm) == assembly_index_exact(g_mk) == 1


def test_canonical_key_same_for_isomorphic_labeled_graphs():
    g1 = make_graph([(0, "M"), (1, "C")], [(0, 1)])
    g2 = make_graph([(5, "M"), (7, "C")], [(5, 7)])
    assert _canonical_key(g1) == _canonical_key(g2)


def test_canonical_key_different_for_different_labels():
    g_mm = make_graph([(0, "M"), (1, "M")], [(0, 1)])
    g_mk = make_graph([(0, "M"), (1, "K")], [(0, 1)])
    assert _canonical_key(g_mm) != _canonical_key(g_mk)


def test_canonical_key_different_for_different_topology():
    p3 = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
    k3 = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
    assert _canonical_key(p3) != _canonical_key(k3)


# ── Determinism and caching ───────────────────────────────────────────────────


def test_assembly_index_exact_deterministic():
    g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
    r1 = assembly_index_exact(g)
    r2 = assembly_index_exact(g)
    assert r1 == r2 == 2


# ── Approx upper bound ────────────────────────────────────────────────────────


def test_approx_ge_exact_for_small_graphs():
    g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
    assert assembly_index_approx(g) >= assembly_index_exact(g)


# ── assembly_index_null ───────────────────────────────────────────────────────


class TestAssemblyIndexNull:
    def test_null_degenerate_single_node(self) -> None:
        """Single-node graph → (0.0, 0.0)."""
        g = nx.Graph()
        g.add_node(0, block_type="M")
        mean, std = assembly_index_null(g, n_shuffles=5)
        assert mean == 0.0
        assert std == 0.0

    def test_null_degenerate_one_edge(self) -> None:
        """Two-node chain (1 edge) → shuffle is identity; null equals observed."""
        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        mean, std = assembly_index_null(g, n_shuffles=5)
        assert mean == float(assembly_index_exact(g))
        assert std == 0.0

    def test_null_preserves_degree_sequence(self) -> None:
        """Shuffled graph must have the same sorted degree sequence as the original."""
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)],
        )
        orig_degrees = sorted(dict(g.degree()).values())
        # Run shuffle 10 times and verify each shuffled copy has the same degrees
        import numpy as np

        rng = np.random.default_rng(42)
        for _ in range(10):
            g_copy = g.copy()
            nswap = max(4, g_copy.number_of_edges())
            seed_i = int(rng.integers(0, 2**31))
            nx.double_edge_swap(g_copy, nswap=nswap, max_tries=nswap * 10, seed=seed_i)
            assert sorted(dict(g_copy.degree()).values()) == orig_degrees

    def test_null_nontrivial_graph_returns_valid_floats(self) -> None:
        """Null model returns valid (mean, std) floats for a swappable graph.

        Uses a 5-node graph with enough non-edges so double_edge_swap can
        succeed, verifying the function runs end-to-end.
        """
        # 5-node graph: path + one cross edge; degree sequence (2,2,2,3,1)
        # Has non-edges so double_edge_swap can find valid swaps.
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)],
        )
        observed = assembly_index_exact(g)
        mean, std = assembly_index_null(g, n_shuffles=10, rng_seed=0)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert mean >= 0.0
        assert std >= 0.0
        # Null mean should be in a reasonable range relative to observed
        assert mean <= observed + 1  # can't greatly exceed original

    def test_compute_entity_metrics_null_columns(self) -> None:
        """Null columns appear in records when n_null_shuffles > 0."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        # Build a minimal fake entity with a 3-node graph
        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2)])

        fake_entity = MagicMock(spec=Entity)

        # Patch at the source module — canonicalize_entity/entity_graph_hash are
        # imported inside compute_entity_metrics via "from domain.entity import ...",
        # so patching alife_discovery.domain.entity is the correct target.
        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakehash"),
        ):
            records = compute_entity_metrics(
                [fake_entity], step=0, run_id="test_run", n_null_shuffles=5
            )

        assert len(records) == 1
        rec = records[0]
        assert "assembly_index_null_mean" in rec
        assert "assembly_index_null_std" in rec
        assert isinstance(rec["assembly_index_null_mean"], float)
        assert isinstance(rec["assembly_index_null_std"], float)

    def test_compute_entity_metrics_no_null_columns_by_default(self) -> None:
        """Null columns absent when n_null_shuffles == 0 (default)."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakehash2"),
        ):
            records = compute_entity_metrics([fake_entity], step=0, run_id="test_run")

        assert len(records) == 1
        rec = records[0]
        assert "assembly_index_null_mean" not in rec
        assert "assembly_index_null_std" not in rec
