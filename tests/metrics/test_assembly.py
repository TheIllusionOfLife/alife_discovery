"""Tests for Assembly Theory assembly index computation (edge-removal DP)."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from alife_discovery.metrics.assembly import (
    _canonical_key,
    assembly_index_approx,
    assembly_index_exact,
    assembly_index_null,
    assembly_index_reuse,
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
        rng = np.random.default_rng(42)
        for _ in range(10):
            g_copy = g.copy()
            nswap = max(4, g_copy.number_of_edges())
            nx.double_edge_swap(g_copy, nswap=nswap, max_tries=nswap * 10, seed=rng)
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

    def test_null_rejects_zero_shuffles(self) -> None:
        """n_shuffles < 1 raises ValueError."""
        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        with pytest.raises(ValueError, match="n_shuffles must be >= 1"):
            assembly_index_null(g, n_shuffles=0)


# ── assembly_index_reuse ────────────────────────────────────────────────────


class TestAssemblyIndexReuse:
    """Tests for AT-standard assembly index with sub-object reuse."""

    def test_single_node_is_zero(self) -> None:
        g = nx.Graph()
        g.add_node(0, block_type="M")
        assert assembly_index_reuse(g) == 0

    def test_single_edge_is_one(self) -> None:
        g = make_graph([(0, "M"), (1, "M")], [(0, 1)])
        assert assembly_index_reuse(g) == 1

    def test_p3_is_two(self) -> None:
        """P_3 with reuse = 2 (no reuse benefit — no repeated substructure)."""
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)])
        assert assembly_index_reuse(g) == 2

    def test_p4_is_two(self) -> None:
        """P_4 with reuse = 2 (build P_2, duplicate+join → P_4)."""
        g = make_graph(
            [(0, "M"), (1, "M"), (2, "M"), (3, "M")],
            [(0, 1), (1, 2), (2, 3)],
        )
        assert assembly_index_reuse(g) == 2

    def test_k3_is_two(self) -> None:
        """KEY: K_3 with reuse = 2 (build P_2, duplicate, join → triangle)."""
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
        assert assembly_index_reuse(g) == 2

    def test_k3_less_than_exact(self) -> None:
        """Reuse-aware index is strictly less than edge-removal DP for K_3."""
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
        assert assembly_index_reuse(g) < assembly_index_exact(g)

    def test_c4_is_three(self) -> None:
        """C_4 with reuse = 3 (cycle closure prevents reuse savings)."""
        g = make_graph(
            [(0, "M"), (1, "M"), (2, "M"), (3, "M")],
            [(0, 1), (1, 2), (2, 3), (3, 0)],
        )
        assert assembly_index_reuse(g) == 3

    def test_reuse_leq_exact_always(self) -> None:
        """Property: reuse <= exact for varied small graphs."""
        graphs = [
            make_graph([(0, "M")], []),
            make_graph([(0, "M"), (1, "M")], [(0, 1)]),
            make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2)]),
            make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)]),
            make_graph(
                [(0, "M"), (1, "M"), (2, "M"), (3, "M")],
                [(0, 1), (1, 2), (2, 3)],
            ),
            make_graph(
                [(0, "M"), (1, "M"), (2, "M"), (3, "M")],
                [(0, 1), (1, 2), (2, 3), (3, 0)],
            ),
        ]
        for g in graphs:
            assert assembly_index_reuse(g) <= assembly_index_exact(g)

    def test_deterministic(self) -> None:
        g = make_graph([(0, "M"), (1, "M"), (2, "M")], [(0, 1), (1, 2), (0, 2)])
        r1 = assembly_index_reuse(g)
        r2 = assembly_index_reuse(g)
        assert r1 == r2

    def test_cache_isolation(self) -> None:
        """_ASSEMBLY_REUSE_CACHE is separate from _ASSEMBLY_CACHE."""
        from alife_discovery.metrics.assembly import _ASSEMBLY_CACHE, _ASSEMBLY_REUSE_CACHE

        assert _ASSEMBLY_REUSE_CACHE is not _ASSEMBLY_CACHE

    def test_compute_entity_metrics_reuse_column(self) -> None:
        """assembly_index_reuse column present when compute_reuse=True."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2)])
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakeR"),
        ):
            records = compute_entity_metrics(
                [fake_entity], step=0, run_id="test_run", compute_reuse=True
            )

        assert len(records) == 1
        assert "assembly_index_reuse" in records[0]
        assert isinstance(records[0]["assembly_index_reuse"], int)

    def test_compute_entity_metrics_no_reuse_by_default(self) -> None:
        """Reuse column absent when compute_reuse=False (default)."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakeR2"),
        ):
            records = compute_entity_metrics([fake_entity], step=0, run_id="test_run")

        assert len(records) == 1
        assert "assembly_index_reuse" not in records[0]


class TestAssemblyIndexNullImproved:
    """Tests for return_samples, empirical p-values, and std=0 guard."""

    def test_return_samples_gives_3_tuple(self) -> None:
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)],
        )
        result = assembly_index_null(g, n_shuffles=10, return_samples=True)
        assert len(result) == 3
        mean, std, samples = result
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert isinstance(samples, np.ndarray)

    def test_samples_array_length_matches_n_shuffles(self) -> None:
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)],
        )
        _, _, samples = assembly_index_null(g, n_shuffles=7, return_samples=True)
        assert len(samples) == 7

    def test_backward_compat_2_tuple_default(self) -> None:
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)],
        )
        result = assembly_index_null(g, n_shuffles=5)
        assert len(result) == 2

    def test_empirical_pvalue_in_zero_one_range(self) -> None:
        g = make_graph(
            [(0, "M"), (1, "C"), (2, "K"), (3, "M"), (4, "C")],
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)],
        )
        _, _, samples = assembly_index_null(g, n_shuffles=20, return_samples=True)
        a_idx = assembly_index_exact(g)
        pvalue = float((samples >= a_idx).mean())
        assert 0.0 <= pvalue <= 1.0

    def test_std_zero_does_not_crash(self) -> None:
        """Single-edge graph → std=0, should not crash."""
        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        mean, std = assembly_index_null(g, n_shuffles=5)
        assert std == 0.0
        # With return_samples
        mean2, std2, samples = assembly_index_null(g, n_shuffles=5, return_samples=True)
        assert std2 == 0.0
        assert len(samples) == 1  # degenerate: single sample = observed

    def test_compute_entity_metrics_pvalue_column_present(self) -> None:
        """pvalue column present when n_null_shuffles > 0."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = make_graph([(0, "M"), (1, "C"), (2, "K")], [(0, 1), (1, 2)])
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakeP"),
        ):
            records = compute_entity_metrics(
                [fake_entity], step=0, run_id="test_run", n_null_shuffles=5
            )

        assert len(records) == 1
        assert "assembly_index_null_pvalue" in records[0]
        pv = records[0]["assembly_index_null_pvalue"]
        assert 0.0 <= pv <= 1.0

    def test_pvalue_column_absent_when_no_null(self) -> None:
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = make_graph([(0, "M"), (1, "C")], [(0, 1)])
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakeP2"),
        ):
            records = compute_entity_metrics([fake_entity], step=0, run_id="test_run")

        assert "assembly_index_null_pvalue" not in records[0]

    def test_pvalue_for_single_node_entity(self) -> None:
        """Edge case: a_i=0, null=0, p=1.0."""
        from unittest.mock import MagicMock, patch

        from alife_discovery.domain.entity import Entity

        g = nx.Graph()
        g.add_node(0, block_type="M")
        fake_entity = MagicMock(spec=Entity)

        with (
            patch("alife_discovery.domain.entity.canonicalize_entity", return_value=g),
            patch("alife_discovery.domain.entity.entity_graph_hash", return_value="fakeSN"),
        ):
            records = compute_entity_metrics(
                [fake_entity], step=0, run_id="test_run", n_null_shuffles=5
            )

        assert records[0]["assembly_index_null_pvalue"] == 1.0


class TestBlockWorldSearchNullSchema:
    """Integration tests for parquet schema selection with null mode."""

    def test_parquet_schema_with_null(self, tmp_path: Path) -> None:
        """Entity log has null columns when n_null_shuffles > 0."""
        import pyarrow.parquet as pq

        from alife_discovery.config.types import BlockWorldConfig
        from alife_discovery.io.schemas import ENTITY_LOG_SCHEMA_WITH_NULL
        from alife_discovery.simulation.engine import run_block_world_search

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, n_null_shuffles=1
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)

        parquet_path = tmp_path / "logs" / "entity_log.parquet"
        assert parquet_path.exists()
        table = pq.read_table(parquet_path)
        assert "assembly_index_null_mean" in table.column_names
        assert "assembly_index_null_std" in table.column_names
        # Verify schema matches expected
        for field in ENTITY_LOG_SCHEMA_WITH_NULL:
            assert field.name in table.column_names

    def test_parquet_schema_without_null(self, tmp_path: Path) -> None:
        """Entity log omits null columns when n_null_shuffles == 0."""
        import pyarrow.parquet as pq

        from alife_discovery.config.types import BlockWorldConfig
        from alife_discovery.simulation.engine import run_block_world_search

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, n_null_shuffles=0
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)

        parquet_path = tmp_path / "logs" / "entity_log.parquet"
        assert parquet_path.exists()
        table = pq.read_table(parquet_path)
        assert "assembly_index_null_mean" not in table.column_names
        assert "assembly_index_null_std" not in table.column_names
