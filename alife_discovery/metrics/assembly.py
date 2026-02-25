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

from typing import Any

import networkx as nx

# Global memoization cache: canonical_key -> assembly_index
_ASSEMBLY_CACHE: dict[str, int] = {}


def _canonical_key(graph: nx.Graph) -> str:
    """Canonical string for a labeled graph, used for memoization.

    Uses networkx Weisfeiler-Leman graph hash with node attribute 'block_type'.
    Includes node/edge count to reduce collisions.
    """
    wl = nx.weisfeiler_lehman_graph_hash(graph, node_attr="block_type")
    return f"n{graph.number_of_nodes()}_e{graph.number_of_edges()}_{wl}"


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


def assembly_index_approx(graph: nx.Graph) -> int:
    """Greedy upper bound: number of edges (assemble one edge at a time)."""
    return graph.number_of_edges()


def compute_entity_metrics(
    entities: list[Any],
    step: int,
    run_id: str,
) -> list[dict[str, Any]]:
    """Compute per-entity AT metrics for a snapshot.

    Returns list of dicts matching ENTITY_LOG_SCHEMA columns.
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
        a_idx = (
            assembly_index_exact(g)
            if n_nodes <= MAX_ENTITY_SIZE
            else assembly_index_approx(g)
        )

        type_counts: dict[str, int] = {"M": 0, "C": 0, "K": 0}
        for node in g.nodes():
            bt = g.nodes[node].get("block_type", "M")
            if bt in type_counts:
                type_counts[bt] += 1

        records.append({
            "run_id": run_id,
            "step": step,
            "entity_hash": h,
            "assembly_index": a_idx,
            "copy_number_at_step": hash_counts[h],
            "entity_size": n_nodes,
            "n_membrane": type_counts["M"],
            "n_cytosol": type_counts["C"],
            "n_catalyst": type_counts["K"],
        })

    return records
