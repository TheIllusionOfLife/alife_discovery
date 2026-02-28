"""Tests for drift_probability config and parameter sweep infrastructure."""

from __future__ import annotations

from random import Random

import pytest

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import BlockWorld, generate_block_rule_table


class TestDriftProbability:
    def test_drift_probability_config_validation_low(self) -> None:
        with pytest.raises(ValueError, match="drift_probability"):
            BlockWorldConfig(drift_probability=0.0)

    def test_drift_probability_config_validation_high(self) -> None:
        with pytest.raises(ValueError, match="drift_probability"):
            BlockWorldConfig(drift_probability=1.5)

    def test_drift_field_on_block_world(self) -> None:
        config = BlockWorldConfig(drift_probability=0.5)
        world = BlockWorld.create(config, Random(0))
        assert world.drift_probability == 0.5

    def test_drift_one_is_default_behavior(self) -> None:
        """Golden: drift_probability=1.0 produces identical results as default."""
        config1 = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, drift_probability=1.0
        )
        config2 = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=10, steps=20)
        rule_table = generate_block_rule_table(0)

        rng1 = Random(42)
        world1 = BlockWorld.create(config1, rng1)
        for _ in range(20):
            world1.step(rule_table, 0.01, rng1)

        rng2 = Random(42)
        world2 = BlockWorld.create(config2, rng2)
        for _ in range(20):
            world2.step(rule_table, 0.01, rng2)

        pos1 = {bid: (b.x, b.y) for bid, b in world1.blocks.items()}
        pos2 = {bid: (b.x, b.y) for bid, b in world2.blocks.items()}
        assert pos1 == pos2
        assert world1.bonds == world2.bonds

    def test_drift_probability_reduces_movement(self) -> None:
        """Statistical: lower drift_probability → fewer per-step position changes."""
        rule_table = generate_block_rule_table(0)

        def count_step_moves(drift_prob: float, seed: int = 42) -> int:
            config = BlockWorldConfig(
                grid_width=20,
                grid_height=20,
                n_blocks=10,
                steps=50,
                drift_probability=drift_prob,
            )
            rng = Random(seed)
            world = BlockWorld.create(config, rng)
            moves = 0
            for _ in range(50):
                prev_pos = {bid: (b.x, b.y) for bid, b in world.blocks.items()}
                world.step(rule_table, 0.01, rng)
                for bid, b in world.blocks.items():
                    if (b.x, b.y) != prev_pos[bid]:
                        moves += 1
            return moves

        moves_low = sum(count_step_moves(0.25, s) for s in range(5))
        moves_high = sum(count_step_moves(1.0, s) for s in range(5))
        # Lower drift probability should result in fewer per-step moves
        assert moves_low < moves_high

    def test_smoke_run_tiny_config(self, tmp_path: object) -> None:
        """Smoke test: run_block_world_search with drift_probability < 1."""
        from pathlib import Path

        from alife_discovery.simulation.engine import run_block_world_search

        config = BlockWorldConfig(
            grid_width=10,
            grid_height=10,
            n_blocks=10,
            steps=20,
            drift_probability=0.5,
        )
        summaries = run_block_world_search(n_rules=1, out_dir=Path(str(tmp_path)), config=config)
        assert len(summaries) == 1
