"""Assembly Theory metrics for block-world entity graphs.

=============================================================================
FORMAL ASSEMBLY GRAMMAR SPEC
=============================================================================

1. COMPOSITION OPERATION
   Each assembly step introduces exactly one new edge e = (u, v) into the
   growing object. Starting from single nodes (a=0), each step joins two
   existing sub-objects by adding one bond edge. This is an edge-join grammar.

2. ASSEMBLY INDEX (edge-removal DP formulation)
   a(G) = minimum number of edge-join steps needed to build G from nodes.

   Recurrence (no sub-object reuse):
     a(G) = 0                             if |E(G)| = 0
     a(G) = min_{e in E(G)} cost(G, e)

   where cost(G, e) when removing e gives connected G' = 1 + a(G')
         cost(G, e) when removing e gives components C1, C2 = 1 + a(C1) + a(C2)

   NOTE: This DP does not implement sub-object reuse (conservative upper bound).
   Full AT with reuse is more complex and deferred; this suffices for ranking
   entities by complexity.

3. BASE CASE
   Single node (any block_type) -> a = 0. No joins needed.

4. KNOWN-ANSWER RECONCILIATION (under this DP)
   - Single node:  a = 0
   - P_2 (1 edge): a = 1
   - P_3 (2 edges): a = 2
   - P_n (n-1 edges, path): a = n-1
   - K_3 (triangle, 3 edges): a = 3
     (remove any edge -> P_3 with a=2, so a(K_3) = 1+2 = 3)
   - C_4 (4-cycle, 4 edges): a = 3
     (remove any edge -> P_3 with a=2, so a(C_4) = 1+2 = 3)
   NOTE: C_4=2 appears in literature only under binary-doubling grammar with
   reuse. The simpler DP used here gives C_4=3.

5. LABEL CONSERVATION
   Node labels (block_type in {M, C, K}) are preserved through composition.
   Two graphs are the same assembly type iff labeled-isomorphic.
=============================================================================
"""

from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
import numpy as np

from alife_discovery.config.constants import BLOCK_TYPES

# Global memoization cache: canonical_key -> assembly_index
_ASSEMBLY_CACHE: dict[str, int] = {}

# Separate cache for reuse-aware assembly index
_ASSEMBLY_REUSE_CACHE: dict[str, int] = {}


def _canonical_key(graph: nx.Graph) -> str:
    """Canonical string for a labeled graph, used for memoization.

    Uses WL hash + sorted adjacency fingerprint for collision resistance.
    The adjacency fingerprint encodes (node_label, sorted_neighbor_labels)
    for every node, making it strongly discriminating for small labeled graphs.
    """
    wl = nx.weisfeiler_lehman_graph_hash(graph, node_attr="block_type")
    adj_sig = sorted(
        (
            graph.nodes[n].get("block_type", "?"),
            tuple(sorted(graph.nodes[nb].get("block_type", "?") for nb in graph.neighbors(n))),
        )
        for n in graph.nodes()
    )
    adj_hash = hashlib.sha256(str(adj_sig).encode()).hexdigest()[:16]
    return f"n{graph.number_of_nodes()}_e{graph.number_of_edges()}_{wl}_{adj_hash}"


def assembly_index_exact(graph: nx.Graph) -> int:
    """Exact assembly index via edge-removal DP with memoization.

    a(G) = min over all edges e of: 1 + assembly_index(G - e)
    where G - e means remove edge e from G.
    If G - e is disconnected, the cost is 1 + sum(a(component)).

    Memoized globally by canonical graph key.
    """
    key = _canonical_key(graph)
    if key in _ASSEMBLY_CACHE:
        return _ASSEMBLY_CACHE[key]

    n_edges = graph.number_of_edges()

    if n_edges == 0:
        _ASSEMBLY_CACHE[key] = 0
        return 0

    if n_edges == 1:
        _ASSEMBLY_CACHE[key] = 1
        return 1

    min_cost = n_edges  # naive upper bound

    for u, v in list(graph.edges()):
        sub = graph.copy()
        sub.remove_edge(u, v)

        if nx.is_connected(sub):
            cost = 1 + assembly_index_exact(sub)
        else:
            cost = 1
            for comp_nodes in nx.connected_components(sub):
                comp = sub.subgraph(comp_nodes).copy()
                cost += assembly_index_exact(comp)

        if cost < min_cost:
            min_cost = cost

    _ASSEMBLY_CACHE[key] = min_cost
    return min_cost


def assembly_index_reuse(graph: nx.Graph) -> int:
    """AT-standard assembly index with sub-object reuse (vertex-primitive counting).

    Like ``assembly_index_exact``, each step adds one edge.  Unlike the
    edge-removal DP, previously constructed sub-objects can be duplicated at
    zero cost.  The assembly index equals the length of the shortest assembly
    pathway — the minimum number of *distinct* construction steps.

    Algorithm: for each "joining edge" *e* in E(G), consider removing *e* and
    partitioning the remaining edges into two connected sub-objects S1 and S2.
    The joining step (+1) adds *e* back to merge S1 and S2.  When S1 ≅ S2
    (by canonical key), the duplicate is free so cost = a(S1) + 1.

    Complexity: |E| × 2^(|E|−1) partitions with memoization. Tractable for
    entity graphs with ≤ 15 edges.

    Memoized globally by canonical graph key (separate from _ASSEMBLY_CACHE).
    """
    key = _canonical_key(graph)
    if key in _ASSEMBLY_REUSE_CACHE:
        return _ASSEMBLY_REUSE_CACHE[key]

    n_edges = graph.number_of_edges()

    if n_edges == 0:
        _ASSEMBLY_REUSE_CACHE[key] = 0
        return 0

    if n_edges == 1:
        _ASSEMBLY_REUSE_CACHE[key] = 1
        return 1

    edges = list(graph.edges())
    n_e = len(edges)
    min_cost = n_edges  # naive upper bound (one step per edge)

    def _subgraph_from_edges(edge_list: list[tuple[int, int]]) -> nx.Graph:
        """Build a subgraph from an edge list, copying node attributes."""
        sg = nx.Graph()
        sg.add_edges_from(edge_list)
        for n in sg.nodes():
            sg.nodes[n].update(graph.nodes[n])
        return sg

    # For each "joining edge" e, try all partitions of E(G)\{e} into (S1, S2).
    for i in range(n_e):
        remaining = edges[:i] + edges[i + 1 :]
        n_rem = len(remaining)

        for mask in range(2**n_rem):
            s1_edges = [remaining[j] for j in range(n_rem) if not (mask >> j & 1)]
            s2_edges = [remaining[j] for j in range(n_rem) if mask >> j & 1]

            if not s1_edges and not s2_edges:
                # G had only 1 edge (handled above), guard only
                cost = 1
            elif not s2_edges:
                # Degenerate: entire G-e as one sub-object, joined by edge e
                s1 = _subgraph_from_edges(s1_edges)
                if not nx.is_connected(s1):
                    continue
                cost = 1 + assembly_index_reuse(s1)
            elif not s1_edges:
                # Symmetric degenerate
                s2 = _subgraph_from_edges(s2_edges)
                if not nx.is_connected(s2):
                    continue
                cost = 1 + assembly_index_reuse(s2)
            else:
                s1 = _subgraph_from_edges(s1_edges)
                s2 = _subgraph_from_edges(s2_edges)
                if not nx.is_connected(s1) or not nx.is_connected(s2):
                    continue
                key_s1 = _canonical_key(s1)
                key_s2 = _canonical_key(s2)
                if key_s1 == key_s2:
                    cost = assembly_index_reuse(s1) + 1
                else:
                    cost = assembly_index_reuse(s1) + assembly_index_reuse(s2) + 1

            if cost < min_cost:
                min_cost = cost

    _ASSEMBLY_REUSE_CACHE[key] = min_cost
    return min_cost


def assembly_index_approx(graph: nx.Graph) -> int:
    """Greedy upper bound: number of edges (assemble one edge at a time)."""
    return graph.number_of_edges()


def assembly_index_null(
    graph: nx.Graph,
    n_shuffles: int = 20,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Return (mean, std) of assembly index over degree-preserving bond shuffles.

    Uses nx.double_edge_swap() (edge-switching Markov chain) to preserve the
    degree sequence exactly. Degenerate cases:
    - 0 or 1 nodes: returns (0.0, 0.0)
    - 0 or 1 edges: shuffle is identity → returns (observed_a_i, 0.0)

    Args:
        graph: The entity graph (node attr ``block_type`` required).
        n_shuffles: Number of independent shuffle trials.
        rng_seed: Seed for reproducibility (passed to random.Random for
            nx.double_edge_swap's internal RNG via numpy seed).

    Returns:
        (mean_a_i, std_a_i) across shuffled graphs.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")

    if graph.number_of_nodes() <= 1:
        return (0.0, 0.0)

    n_edges = graph.number_of_edges()
    if n_edges < 2:
        observed = float(assembly_index_exact(graph))
        return (observed, 0.0)

    nswap = max(4, n_edges)
    results: list[int] = []

    rng = np.random.default_rng(rng_seed)
    for _ in range(n_shuffles):
        g_copy = graph.copy()
        try:
            nx.double_edge_swap(g_copy, nswap=nswap, max_tries=nswap * 10, seed=rng)
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            # Swap failed (e.g., all edges incident to same node pair); use original
            g_copy = graph.copy()
        results.append(assembly_index_exact(g_copy))

    arr = np.array(results, dtype=float)
    return (float(arr.mean()), float(arr.std()))


def compute_entity_metrics(
    entities: list[Any],
    step: int,
    run_id: str,
    n_null_shuffles: int = 0,
    compute_reuse: bool = False,
) -> list[dict[str, Any]]:
    """Compute per-entity AT metrics for a snapshot.

    Returns list of dicts matching ENTITY_LOG_SCHEMA columns (or
    ENTITY_LOG_SCHEMA_WITH_NULL when n_null_shuffles > 0).

    Args:
        entities: Detected entity objects from detect_entities().
        step: Current simulation step.
        run_id: Identifier for this run.
        n_null_shuffles: When > 0, compute shuffle-bond null model and include
            ``assembly_index_null_mean`` and ``assembly_index_null_std`` in each record.
    """
    from alife_discovery.config.constants import MAX_ENTITY_SIZE
    from alife_discovery.domain.entity import canonicalize_entity, entity_graph_hash

    hash_counts: dict[str, int] = {}
    entity_data: list[tuple[str, nx.Graph, Any]] = []

    for entity in entities:
        g = canonicalize_entity(entity)
        h = entity_graph_hash(g)
        hash_counts[h] = hash_counts.get(h, 0) + 1
        entity_data.append((h, g, entity))

    records = []
    for h, g, _entity in entity_data:
        n_nodes = g.number_of_nodes()
        a_idx = assembly_index_exact(g) if n_nodes <= MAX_ENTITY_SIZE else assembly_index_approx(g)

        type_counts: dict[str, int] = {bt: 0 for bt in BLOCK_TYPES}
        for node in g.nodes():
            bt = g.nodes[node]["block_type"]  # fail-fast: missing attr is a bug
            if bt in type_counts:
                type_counts[bt] += 1

        record: dict[str, Any] = {
            "run_id": run_id,
            "step": step,
            "entity_hash": h,
            "assembly_index": a_idx,
            "copy_number_at_step": hash_counts[h],
            "entity_size": n_nodes,
            "n_membrane": type_counts["M"],
            "n_cytosol": type_counts["C"],
            "n_catalyst": type_counts["K"],
        }

        if compute_reuse:
            record["assembly_index_reuse"] = assembly_index_reuse(g)

        if n_null_shuffles > 0:
            null_mean, null_std = assembly_index_null(
                g,
                n_shuffles=n_null_shuffles,
                rng_seed=int(hashlib.sha256(h.encode()).hexdigest()[:8], 16),
            )
            record["assembly_index_null_mean"] = null_mean
            record["assembly_index_null_std"] = null_std

        records.append(record)

    return records
