"""Tests for alife_discovery.domain.block_world module."""

from __future__ import annotations

from collections import Counter
from random import Random

import pytest

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import (
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
