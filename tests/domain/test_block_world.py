"""Tests for alife_discovery.domain.block_world module."""

from __future__ import annotations

from collections import Counter
from random import Random

import pytest

from alife_discovery.config.constants import BLOCK_TYPES
from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import (
    BlockRuleTable,
    BlockWorld,
    generate_block_rule_table,
)


class TestBlockWorldCreate:
    def test_correct_block_count(self) -> None:
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=5)
        world = BlockWorld.create(config, Random(0))
        assert len(world.blocks) == 5

    def test_no_overlapping_positions(self) -> None:
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=20)
        world = BlockWorld.create(config, Random(42))
        positions = [(b.x, b.y) for b in world.blocks.values()]
        assert len(positions) == len(set(positions))

    def test_type_fractions_approximately_correct(self) -> None:
        config = BlockWorldConfig(n_blocks=30, block_type_fractions=(0.5, 0.3, 0.2))
        world = BlockWorld.create(config, Random(0))
        counts = Counter(b.block_type for b in world.blocks.values())
        assert counts["M"] == 15  # 0.5 * 30
        assert counts["C"] == 9  # 0.3 * 30
        assert counts["K"] == 6  # 0.2 * 30

    def test_no_initial_bonds(self) -> None:
        world = BlockWorld.create(BlockWorldConfig(), Random(0))
        assert len(world.bonds) == 0

    def test_grid_occupancy_matches_blocks(self) -> None:
        world = BlockWorld.create(BlockWorldConfig(), Random(0))
        assert len(world.grid) == len(world.blocks)
        for pos, bid in world.grid.items():
            block = world.blocks[bid]
            assert (block.x, block.y) == pos

    def test_blocks_within_grid_bounds(self) -> None:
        config = BlockWorldConfig(grid_width=10, grid_height=8, n_blocks=15)
        world = BlockWorld.create(config, Random(0))
        for block in world.blocks.values():
            assert 0 <= block.x < 10
            assert 0 <= block.y < 8


class TestBlockWorldBonds:
    def test_bonds_are_frozensets_of_two(self) -> None:
        world = BlockWorld.create(BlockWorldConfig(), Random(0))
        ids = list(world.blocks.keys())[:2]
        world.bonds.add(frozenset(ids))
        for bond in world.bonds:
            assert isinstance(bond, frozenset)
            assert len(bond) == 2

    def test_bond_break_with_noise_1(self) -> None:
        """noise_level=1.0 must break ALL bonds every step."""
        config = BlockWorldConfig(n_blocks=10)
        world = BlockWorld.create(config, Random(0))
        # Manually add bonds between adjacent blocks
        ids = list(world.blocks.keys())
        for i in range(len(ids) - 1):
            world.bonds.add(frozenset({ids[i], ids[i + 1]}))
        rule_table = generate_block_rule_table(0)
        world.step(rule_table, noise_level=1.0, rng=Random(0))
        assert len(world.bonds) == 0

    def test_bond_motion_invariant(self) -> None:
        """Bond breaks if block moves away from bonded neighbor."""
        config = BlockWorldConfig(grid_width=5, grid_height=5, n_blocks=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = world.blocks[ids[0]], world.blocks[ids[1]]
        # Place blocks adjacently manually
        b0.x, b0.y = 2, 2
        b1.x, b1.y = 2, 3
        world.grid.clear()
        world.grid[(2, 2)] = ids[0]
        world.grid[(2, 3)] = ids[1]
        world.bonds.add(frozenset({ids[0], ids[1]}))
        # Force b0 to move to (2, 1) (away from b1 at 2,3)
        del world.grid[(2, 2)]
        b0.x, b0.y = 2, 1
        world.grid[(2, 1)] = ids[0]
        # Now call _prune_broken_bonds
        world._prune_broken_bonds(ids[0])
        assert frozenset({ids[0], ids[1]}) not in world.bonds


class TestBlockWorldStep:
    def test_drift_moves_block(self) -> None:
        """Single block in 5x5 grid must move within 20 steps."""
        config = BlockWorldConfig(grid_width=5, grid_height=5, n_blocks=1)
        world = BlockWorld.create(config, Random(0))
        block_id = list(world.blocks.keys())[0]
        original_pos = (world.blocks[block_id].x, world.blocks[block_id].y)
        rule_table = generate_block_rule_table(0)
        rng = Random(1)
        moved = False
        for _ in range(20):
            world.step(rule_table, noise_level=0.0, rng=rng)
            new_pos = (world.blocks[block_id].x, world.blocks[block_id].y)
            if new_pos != original_pos:
                moved = True
                break
        assert moved

    def test_drift_blocked_by_occupied_cell(self) -> None:
        """In a 2x1 grid with 2 blocks, no overlap after stepping."""
        config = BlockWorldConfig(grid_width=2, grid_height=1, n_blocks=2)
        world = BlockWorld.create(config, Random(0))
        rule_table = generate_block_rule_table(0)
        rng = Random(0)
        for _ in range(10):
            world.step(rule_table, noise_level=0.0, rng=rng)
        all_positions = [(b.x, b.y) for b in world.blocks.values()]
        assert len(all_positions) == len(set(all_positions))

    def test_step_preserves_block_count(self) -> None:
        config = BlockWorldConfig(n_blocks=15)
        world = BlockWorld.create(config, Random(0))
        rule_table = generate_block_rule_table(0)
        rng = Random(42)
        for _ in range(50):
            world.step(rule_table, noise_level=0.1, rng=rng)
        assert len(world.blocks) == 15
        assert len(world.grid) == 15

    def test_step_no_position_overlaps(self) -> None:
        config = BlockWorldConfig(n_blocks=15)
        world = BlockWorld.create(config, Random(0))
        rule_table = generate_block_rule_table(0)
        rng = Random(42)
        for _ in range(50):
            world.step(rule_table, noise_level=0.1, rng=rng)
            positions = [(b.x, b.y) for b in world.blocks.values()]
            assert len(positions) == len(set(positions))


class TestGenerateRuleTable:
    def test_rule_table_is_deterministic(self) -> None:
        t1 = generate_block_rule_table(42)
        t2 = generate_block_rule_table(42)
        assert t1 == t2

    def test_rule_table_different_seeds_differ(self) -> None:
        t1 = generate_block_rule_table(0)
        t2 = generate_block_rule_table(1)
        assert t1 != t2

    def test_rule_table_values_in_0_1(self) -> None:
        table = generate_block_rule_table(0)
        for v in table.values():
            assert 0.0 <= v <= 1.0


class TestBlockWorldConfig:
    def test_default_config(self) -> None:
        config = BlockWorldConfig()
        assert config.grid_width == 20
        assert config.grid_height == 20
        assert config.n_blocks == 30

    def test_invalid_grid_dimensions(self) -> None:
        with pytest.raises(ValueError, match="grid dimensions must be >= 1"):
            BlockWorldConfig(grid_width=0)

    def test_invalid_n_blocks(self) -> None:
        with pytest.raises(ValueError, match="n_blocks must be >= 1"):
            BlockWorldConfig(n_blocks=0)

    def test_n_blocks_exceeds_grid(self) -> None:
        with pytest.raises(ValueError, match="n_blocks cannot exceed grid cells"):
            BlockWorldConfig(grid_width=2, grid_height=2, n_blocks=5)

    def test_invalid_fractions_length(self) -> None:
        with pytest.raises(ValueError, match="block_type_fractions must have 3 elements"):
            BlockWorldConfig(block_type_fractions=(0.5, 0.5))

    def test_invalid_fractions_sum(self) -> None:
        with pytest.raises(ValueError, match="block_type_fractions must sum to 1.0"):
            BlockWorldConfig(block_type_fractions=(0.5, 0.3, 0.3))

    def test_invalid_noise_level(self) -> None:
        with pytest.raises(ValueError, match="noise_level must be in"):
            BlockWorldConfig(noise_level=1.5)

    def test_invalid_observation_range(self) -> None:
        with pytest.raises(ValueError, match="observation_range must be >= 1"):
            BlockWorldConfig(observation_range=0)

    def test_invalid_steps(self) -> None:
        with pytest.raises(ValueError, match="steps must be >= 1"):
            BlockWorldConfig(steps=0)


class TestObservationRange:
    """Tests for observation_range wiring: bond formation and pruning at r > 1."""

    def _place(self, world: BlockWorld, bid: int, x: int, y: int) -> None:
        """Teleport a block to (x, y), clearing old grid entry."""
        block = world.blocks[bid]
        if (block.x, block.y) in world.grid:
            del world.grid[(block.x, block.y)]
        block.x = x
        block.y = y
        world.grid[(x, y)] = bid

    def _uniform_rule_table(self, prob: float) -> BlockRuleTable:
        """Build a rule table with uniform bond probability across all keys."""
        table: BlockRuleTable = {}
        for st in BLOCK_TYPES:
            for nc in range(5):
                for dt in list(BLOCK_TYPES) + ["Empty"]:
                    table[(st, nc, dt)] = prob
        return table

    def test_bond_form_radius2_reaches_distance2_block(self) -> None:
        """radius=2: block at (0,0) should bond with block at (0,2) — Manhattan dist 2."""
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=2, observation_range=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        self._place(world, b0, 0, 0)
        self._place(world, b1, 0, 2)

        # Bond probability 1.0 for all keys ensures bonding is certain if in range
        rng = Random(42)
        world._try_bond_form(b0, self._uniform_rule_table(1.0), rng)
        assert frozenset({b0, b1}) in world.bonds

    def test_bond_form_radius1_cannot_reach_distance2_block(self) -> None:
        """radius=1: block at (0,0) must NOT bond with block at (0,2) — too far."""
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=2, observation_range=1)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        self._place(world, b0, 0, 0)
        self._place(world, b1, 0, 2)
        rng = Random(42)
        world._try_bond_form(b0, self._uniform_rule_table(1.0), rng)
        assert frozenset({b0, b1}) not in world.bonds

    def test_prune_breaks_bond_when_outside_range2(self) -> None:
        """radius=2: bond with dist-3 pair is pruned after the block is placed far away."""
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=2, observation_range=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        # Start adjacent so bond is legitimate
        self._place(world, b0, 0, 0)
        self._place(world, b1, 0, 1)
        world.bonds.add(frozenset({b0, b1}))
        # Now move b0 to distance 3 from b1
        self._place(world, b0, 0, 4)
        world._prune_broken_bonds(b0)
        assert frozenset({b0, b1}) not in world.bonds

    def test_prune_keeps_bond_within_range2(self) -> None:
        """radius=2: bond between dist-2 pair survives pruning."""
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=2, observation_range=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        self._place(world, b0, 0, 0)
        self._place(world, b1, 0, 2)
        world.bonds.add(frozenset({b0, b1}))
        world._prune_broken_bonds(b0)
        assert frozenset({b0, b1}) in world.bonds

    def test_neighbor_count_capped_at_4(self) -> None:
        """With 6 neighbors in radius=2, rule table is queried with neighbor_count ≤ 4."""
        from alife_discovery.config.constants import BLOCK_TYPES

        class RecordingRuleTable(dict):
            """dict subclass that records the neighbor_count from each .get() key."""

            def __init__(self) -> None:
                super().__init__()
                self.observed_counts: list[int] = []

            def get(self, key: object, default: float = 0.0) -> float:  # type: ignore[override]
                assert isinstance(key, tuple)
                _, nc, _ = key
                self.observed_counts.append(nc)
                return super().get(key, default)  # type: ignore[return-value]

        # 7-block config (center + 6 arranged around it within radius 2)
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=7, observation_range=2)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        center_id = ids[0]
        neighbor_ids = ids[1:7]

        # Place center at (5,5) and 6 neighbors within Manhattan dist ≤ 2
        self._place(world, center_id, 5, 5)
        positions = [(5, 4), (5, 6), (4, 5), (6, 5), (5, 3), (5, 7)]
        for bid, pos in zip(neighbor_ids, positions, strict=True):
            self._place(world, bid, pos[0], pos[1])

        rule_table: RecordingRuleTable = RecordingRuleTable()
        for st in BLOCK_TYPES:
            for nc in range(5):
                for dt in list(BLOCK_TYPES) + ["Empty"]:
                    rule_table[(st, nc, dt)] = 0.0

        world._try_bond_form(center_id, rule_table, Random(0))

        # Exactly one lookup, and neighbor_count must be capped at ≤ 4
        assert len(rule_table.observed_counts) == 1
        assert rule_table.observed_counts[0] <= 4

    def test_synchronous_stale_bond_uses_range(self) -> None:
        """Synchronous step with radius=2 prunes bonds that exceed observation_range."""
        from alife_discovery.config.types import UpdateMode

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=2, observation_range=2, steps=1
        )
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        # Place blocks at distance 5 — beyond observation_range=2
        self._place(world, b0, 0, 0)
        self._place(world, b1, 0, 5)
        # Manually inject a bond that should not exist at this distance
        world.bonds.add(frozenset({b0, b1}))
        world.step(
            self._uniform_rule_table(0.0),
            noise_level=0.0,
            rng=Random(0),
            update_mode=UpdateMode.SYNCHRONOUS,
        )
        assert frozenset({b0, b1}) not in world.bonds

    def test_prune_bond_survives_toroidal_wrap(self) -> None:
        """Bond across the toroidal edge (toroidal dist=1) must not be pruned at range=1."""
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=2, observation_range=1)
        world = BlockWorld.create(config, Random(0))
        ids = list(world.blocks.keys())
        b0, b1 = ids[0], ids[1]
        # (0,0) and (9,0): raw distance=9, toroidal distance=1
        self._place(world, b0, 0, 0)
        self._place(world, b1, 9, 0)
        world.bonds.add(frozenset({b0, b1}))
        world._prune_broken_bonds(b0)
        assert frozenset({b0, b1}) in world.bonds
