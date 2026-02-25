"""Tests for alife_discovery.domain.entity module."""

from __future__ import annotations

from random import Random

import networkx as nx

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import BlockWorld
from alife_discovery.domain.entity import (
    canonicalize_entity,
    detect_entities,
    entity_graph_hash,
)


class TestDetectEntities:
    def test_no_bonds_each_block_is_entity(self) -> None:
        config = BlockWorldConfig(n_blocks=5)
        world = BlockWorld.create(config, Random(0))
        assert len(world.bonds) == 0
        entities = detect_entities(world)
        assert len(entities) == 5

    def test_all_bonded_single_entity(self) -> None:
        config = BlockWorldConfig(grid_width=5, grid_height=5, n_blocks=3)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        # Place blocks in a line and bond them
        world.blocks[ids[0]].x, world.blocks[ids[0]].y = 1, 1
        world.blocks[ids[1]].x, world.blocks[ids[1]].y = 1, 2
        world.blocks[ids[2]].x, world.blocks[ids[2]].y = 1, 3
        world.grid.clear()
        for i, pos in [(0, (1, 1)), (1, (1, 2)), (2, (1, 3))]:
            world.grid[pos] = ids[i]
        world.bonds.add(frozenset({ids[0], ids[1]}))
        world.bonds.add(frozenset({ids[1], ids[2]}))
        entities = detect_entities(world)
        assert len(entities) == 1
        assert len(entities[0].block_ids) == 3

    def test_single_block_entity_size_1(self) -> None:
        config = BlockWorldConfig(n_blocks=1)
        world = BlockWorld.create(config, Random(0))
        entities = detect_entities(world)
        assert len(entities) == 1
        assert len(entities[0].block_ids) == 1

    def test_entity_stores_bond_edges(self) -> None:
        config = BlockWorldConfig(grid_width=5, grid_height=5, n_blocks=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        # Place adjacently and bond
        world.blocks[ids[0]].x, world.blocks[ids[0]].y = 0, 0
        world.blocks[ids[1]].x, world.blocks[ids[1]].y = 0, 1
        world.grid.clear()
        world.grid[(0, 0)] = ids[0]
        world.grid[(0, 1)] = ids[1]
        bond = frozenset({ids[0], ids[1]})
        world.bonds.add(bond)
        entities = detect_entities(world)
        assert len(entities) == 1
        assert bond in entities[0].bond_edges

    def test_two_disjoint_components(self) -> None:
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=4)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        # Place in two pairs far apart
        world.blocks[ids[0]].x, world.blocks[ids[0]].y = 0, 0
        world.blocks[ids[1]].x, world.blocks[ids[1]].y = 0, 1
        world.blocks[ids[2]].x, world.blocks[ids[2]].y = 5, 5
        world.blocks[ids[3]].x, world.blocks[ids[3]].y = 5, 6
        world.grid.clear()
        world.grid[(0, 0)] = ids[0]
        world.grid[(0, 1)] = ids[1]
        world.grid[(5, 5)] = ids[2]
        world.grid[(5, 6)] = ids[3]
        world.bonds.add(frozenset({ids[0], ids[1]}))
        world.bonds.add(frozenset({ids[2], ids[3]}))
        entities = detect_entities(world)
        assert len(entities) == 2
        sizes = sorted(len(e.block_ids) for e in entities)
        assert sizes == [2, 2]


class TestCanonicalizeEntity:
    def test_returns_nx_graph(self) -> None:
        config = BlockWorldConfig(n_blocks=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        world.blocks[ids[0]].x, world.blocks[ids[0]].y = 0, 0
        world.blocks[ids[1]].x, world.blocks[ids[1]].y = 0, 1
        world.grid.clear()
        world.grid[(0, 0)] = ids[0]
        world.grid[(0, 1)] = ids[1]
        world.bonds.add(frozenset({ids[0], ids[1]}))
        entities = detect_entities(world)
        g = canonicalize_entity(entities[0])
        assert isinstance(g, nx.Graph)

    def test_node_count_matches_entity(self) -> None:
        config = BlockWorldConfig(n_blocks=3)
        world = BlockWorld.create(config, Random(0))
        entities = detect_entities(world)
        for e in entities:
            g = canonicalize_entity(e)
            assert g.number_of_nodes() == len(e.block_ids)


class TestEntityGraphHash:
    def test_hash_is_64_hex(self) -> None:
        config = BlockWorldConfig(n_blocks=3)
        world = BlockWorld.create(config, Random(0))
        entities = detect_entities(world)
        for e in entities:
            g = canonicalize_entity(e)
            h = entity_graph_hash(g)
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_same_structure_same_hash(self) -> None:
        g1 = nx.Graph()
        g1.add_node(0, block_type="M")
        g1.add_node(1, block_type="M")
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0, block_type="M")
        g2.add_node(1, block_type="M")
        g2.add_edge(0, 1)

        assert entity_graph_hash(g1) == entity_graph_hash(g2)

    def test_different_labels_different_hash(self) -> None:
        g1 = nx.Graph()
        g1.add_node(0, block_type="M")
        g1.add_node(1, block_type="M")

        g2 = nx.Graph()
        g2.add_node(0, block_type="M")
        g2.add_node(1, block_type="K")

        assert entity_graph_hash(g1) != entity_graph_hash(g2)

    def test_different_structure_different_hash(self) -> None:
        # Linear chain M-M-M
        g1 = nx.Graph()
        g1.add_node(0, block_type="M")
        g1.add_node(1, block_type="M")
        g1.add_node(2, block_type="M")
        g1.add_edge(0, 1)
        g1.add_edge(1, 2)

        # Triangle M-M-M
        g2 = nx.Graph()
        g2.add_node(0, block_type="M")
        g2.add_node(1, block_type="M")
        g2.add_node(2, block_type="M")
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)
        g2.add_edge(0, 2)

        assert entity_graph_hash(g1) != entity_graph_hash(g2)
