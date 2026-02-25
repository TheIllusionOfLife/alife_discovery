"""Entity detection via bond-graph BFS and labeled-graph canonicalization.

Two entities are the same 'type' iff their bond graphs are labeled-isomorphic:
same graph structure AND same block_type labels at corresponding nodes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx

from alife_discovery.domain.block_world import BlockType, BlockWorld

CanonicalGraph: TypeAlias = nx.Graph
"""A NetworkX graph where each node has attribute 'block_type' (M/C/K)."""


@dataclass(frozen=True)
class Entity:
    """A connected component of the bond graph at a snapshot."""

    block_ids: frozenset[int]
    positions: dict[int, tuple[int, int]]  # block_id -> (x, y) at snapshot time
    block_types: dict[int, BlockType]  # block_id -> BlockType
    bond_edges: frozenset[frozenset[int]]  # bonds within this entity


def detect_entities(world: BlockWorld) -> list[Entity]:
    """Detect all entities (bond-graph connected components) in world.

    Single isolated blocks (no bonds) are included as size-1 entities.
    """
    visited: set[int] = set()
    entities: list[Entity] = []

    for block_id in world.blocks:
        if block_id in visited:
            continue
        # BFS from block_id
        component: set[int] = set()
        queue = [block_id]
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            for bond in world.bonds:
                if current in bond:
                    other = next(b for b in bond if b != current)
                    if other not in component:
                        queue.append(other)
        visited |= component
        positions = {bid: (world.blocks[bid].x, world.blocks[bid].y) for bid in component}
        block_types = {bid: world.blocks[bid].block_type for bid in component}
        bond_edges = frozenset(bond for bond in world.bonds if bond.issubset(component))
        entities.append(
            Entity(
                block_ids=frozenset(component),
                positions=positions,
                block_types=block_types,
                bond_edges=bond_edges,
            )
        )

    return entities


def canonicalize_entity(entity: Entity) -> CanonicalGraph:
    """Build canonical labeled graph from entity.

    Returns a NetworkX Graph with:
    - Nodes labeled 0..n-1 in canonical order
    - Each node has attribute 'block_type' (M/C/K)
    - Edges from bond_edges
    """
    block_ids = sorted(entity.block_ids)
    id_to_idx = {bid: i for i, bid in enumerate(block_ids)}

    g = nx.Graph()
    for bid in block_ids:
        g.add_node(id_to_idx[bid], block_type=entity.block_types[bid])

    for bond in entity.bond_edges:
        endpoints = sorted(bond)
        g.add_edge(id_to_idx[endpoints[0]], id_to_idx[endpoints[1]])

    return g


def entity_graph_hash(graph: CanonicalGraph) -> str:
    """Return SHA-256 hex of a WL-style canonical string of the graph.

    The canonical string encodes node labels and neighborhood structure.
    Two rounds of WL refinement are sufficient for small typed graphs.
    """
    nodes = sorted(graph.nodes())

    def node_signature(n: int) -> str:
        bt = graph.nodes[n]["block_type"]
        neighbor_types = sorted(graph.nodes[nb]["block_type"] for nb in graph.neighbors(n))
        return f"{bt}:[{','.join(neighbor_types)}]"

    # Iterative WL refinement (2 rounds sufficient for small graphs)
    labels: dict[int, str] = {n: node_signature(n) for n in nodes}
    for _ in range(2):
        new_labels: dict[int, str] = {}
        for n in nodes:
            neighbor_labels = sorted(labels[nb] for nb in graph.neighbors(n))
            new_labels[n] = (
                f"{graph.nodes[n]['block_type']}|{labels[n]}|{','.join(neighbor_labels)}"
            )
        labels = new_labels

    canonical_str = ";".join(sorted(labels.values()))
    canonical_str += f"|n={graph.number_of_nodes()}|e={graph.number_of_edges()}"
    return hashlib.sha256(canonical_str.encode()).hexdigest()
