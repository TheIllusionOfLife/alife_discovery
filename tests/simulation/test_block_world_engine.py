"""Tests for block-world simulation engine (run_block_world_search)."""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow.parquet as pq

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.domain.block_world import BlockWorld, generate_block_rule_table
from alife_discovery.io.schemas import ENTITY_LOG_SCHEMA
from alife_discovery.simulation.engine import run_block_world_search


class TestRunBlockWorldSearch:
    def test_produces_entity_log_parquet(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=2, out_dir=tmp_path, config=None)
        entity_log = tmp_path / "logs" / "entity_log.parquet"
        assert entity_log.exists()

    def test_entity_log_has_correct_columns(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=None)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        expected_columns = {f.name for f in ENTITY_LOG_SCHEMA}
        assert set(table.column_names) == expected_columns

    def test_entity_hash_values_are_64_hex(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=None)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        hashes = table.column("entity_hash").to_pylist()
        assert len(hashes) > 0
        for h in hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_assembly_index_non_negative(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=None)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        indices = table.column("assembly_index").to_pylist()
        assert all(idx >= 0 for idx in indices)

    def test_entity_size_at_least_1(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=None)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        sizes = table.column("entity_size").to_pylist()
        assert all(s >= 1 for s in sizes)

    def test_returns_summary_dicts(self, tmp_path: Path) -> None:
        results = run_block_world_search(n_rules=2, out_dir=tmp_path, config=None)
        assert len(results) == 2
        for r in results:
            assert "run_id" in r
            assert "n_entities_final" in r

    def test_block_type_counts_sum_to_entity_size(self, tmp_path: Path) -> None:
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=None)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        for i in range(table.num_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            assert row["n_membrane"] + row["n_cytosol"] + row["n_catalyst"] == row["entity_size"]

    def test_observation_range_changes_bond_count(self, tmp_path: Path) -> None:
        """observation_range lever produces deterministically different bond counts (golden)."""
        # Golden values pinned at rule_seed=sim_seed=7, steps=50, 20x20 grid, 30 blocks.
        # Update these constants if BlockWorld physics change intentionally.
        EXPECTED_BONDS_RANGE_1 = 2
        EXPECTED_BONDS_RANGE_2 = 8

        def run(observation_range: int) -> int:
            cfg = BlockWorldConfig(
                grid_width=20,
                grid_height=20,
                n_blocks=30,
                observation_range=observation_range,
                steps=50,
                rule_seed=7,
                sim_seed=7,
            )
            rule_table = generate_block_rule_table(cfg.rule_seed)
            rng = random.Random(cfg.sim_seed)
            world = BlockWorld.create(cfg, rng)
            for _ in range(cfg.steps):
                world.step(rule_table, cfg.noise_level, rng, update_mode=cfg.update_mode)
            return len(world.bonds)

        assert run(1) == EXPECTED_BONDS_RANGE_1
        assert run(2) == EXPECTED_BONDS_RANGE_2


class TestStepTimeseries:
    """Tests for step_timeseries.parquet generation."""

    def test_timeseries_written_when_flag_set(self, tmp_path: Path) -> None:
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, write_timeseries=True
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        ts_path = tmp_path / "logs" / "step_timeseries.parquet"
        assert ts_path.exists()

    def test_timeseries_not_written_by_default(self, tmp_path: Path) -> None:
        config = BlockWorldConfig(grid_width=10, grid_height=10, n_blocks=10, steps=20)
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        ts_path = tmp_path / "logs" / "step_timeseries.parquet"
        assert not ts_path.exists()

    def test_timeseries_schema_columns(self, tmp_path: Path) -> None:
        from alife_discovery.io.schemas import STEP_TIMESERIES_SCHEMA

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, write_timeseries=True
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        table = pq.read_table(tmp_path / "logs" / "step_timeseries.parquet")
        expected = {f.name for f in STEP_TIMESERIES_SCHEMA}
        assert set(table.column_names) == expected

    def test_timeseries_step_values_match_snapshot_interval(self, tmp_path: Path) -> None:
        from alife_discovery.config.constants import ENTITY_SNAPSHOT_INTERVAL

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, write_timeseries=True
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        table = pq.read_table(tmp_path / "logs" / "step_timeseries.parquet")
        steps = table.column("step").to_pylist()
        # Steps should correspond to snapshot interval or last step
        for s in steps:
            assert (s + 1) % ENTITY_SNAPSHOT_INTERVAL == 0 or s == config.steps - 1

    def test_timeseries_sorted_by_run_id_step(self, tmp_path: Path) -> None:
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20, write_timeseries=True
        )
        run_block_world_search(n_rules=2, out_dir=tmp_path, config=config)
        table = pq.read_table(tmp_path / "logs" / "step_timeseries.parquet")
        run_ids = table.column("run_id").to_pylist()
        steps = table.column("step").to_pylist()
        pairs = list(zip(run_ids, steps, strict=True))
        assert pairs == sorted(pairs)


class TestFullSchema:
    """Test ENTITY_LOG_SCHEMA_FULL: reuse + null columns together."""

    def test_full_schema_columns_present(self, tmp_path: Path) -> None:
        """compute_reuse_index=True + n_null_shuffles>0 produces all columns."""
        from alife_discovery.io.schemas import ENTITY_LOG_SCHEMA_FULL

        config = BlockWorldConfig(
            grid_width=10,
            grid_height=10,
            n_blocks=10,
            steps=20,
            compute_reuse_index=True,
            n_null_shuffles=10,
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        expected = {f.name for f in ENTITY_LOG_SCHEMA_FULL}
        assert set(table.column_names) == expected

    def test_full_schema_reuse_leq_exact(self, tmp_path: Path) -> None:
        """Reuse AI should be <= exact AI for all rows."""
        config = BlockWorldConfig(
            grid_width=10,
            grid_height=10,
            n_blocks=10,
            steps=20,
            compute_reuse_index=True,
            n_null_shuffles=10,
        )
        run_block_world_search(n_rules=1, out_dir=tmp_path, config=config)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        ai_exact = table.column("assembly_index").to_pylist()
        ai_reuse = table.column("assembly_index_reuse").to_pylist()
        for exact, reuse in zip(ai_exact, ai_reuse, strict=True):
            assert reuse <= exact
