"""Tests for Assembly Theory assembly index computation (edge-removal DP)."""

import networkx as nx

from alife_discovery.metrics.assembly import (
    _canonical_key,
    assembly_index_approx,
    assembly_index_exact,
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
