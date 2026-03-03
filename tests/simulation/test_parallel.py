"""Tests for parallel rule-level simulation runner."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.simulation.parallel import run_rules_parallel


class TestRunRulesParallel:
    """Verify parallel results match sequential for same seeds."""

    def test_parallel_matches_sequential(self, tmp_path: Path) -> None:
        """Same config → identical entity records regardless of n_workers."""
        from alife_discovery.simulation.engine import run_block_world_search

        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20
        )

        # Sequential (engine)
        seq_dir = tmp_path / "seq"
        run_block_world_search(n_rules=3, out_dir=seq_dir, config=config)
        seq_table = pq.read_table(seq_dir / "logs" / "entity_log.parquet")

        # Parallel
        par_dir = tmp_path / "par"
        run_rules_parallel(n_rules=3, out_dir=par_dir, config=config, n_workers=2)
        par_table = pq.read_table(par_dir / "logs" / "entity_log.parquet")

        # Same number of rows
        assert seq_table.num_rows == par_table.num_rows

        # Sort both by (run_id, step, entity_hash) and compare key columns
        sort_keys = [("run_id", "ascending"), ("step", "ascending"), ("entity_hash", "ascending")]
        seq_sorted = seq_table.sort_by(sort_keys)
        par_sorted = par_table.sort_by(sort_keys)

        for col in ["run_id", "step", "entity_hash", "assembly_index", "entity_size"]:
            seq_col = seq_sorted.column(col).to_pylist()
            par_col = par_sorted.column(col).to_pylist()
            assert seq_col == par_col, f"Mismatch in column {col}"

    def test_parallel_single_worker(self, tmp_path: Path) -> None:
        """n_workers=1 should still produce valid output."""
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20
        )
        run_rules_parallel(n_rules=2, out_dir=tmp_path, config=config, n_workers=1)
        assert (tmp_path / "logs" / "entity_log.parquet").exists()

    def test_parallel_with_null_shuffles(self, tmp_path: Path) -> None:
        """Null shuffle columns present in parallel output."""
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20,
            n_null_shuffles=5,
        )
        run_rules_parallel(n_rules=2, out_dir=tmp_path, config=config, n_workers=2)
        table = pq.read_table(tmp_path / "logs" / "entity_log.parquet")
        assert "assembly_index_null_mean" in table.column_names

    def test_parallel_zero_rules_raises(self, tmp_path: Path) -> None:
        """n_rules < 1 raises ValueError."""
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20
        )
        with pytest.raises(ValueError, match="n_rules must be >= 1"):
            run_rules_parallel(n_rules=0, out_dir=tmp_path, config=config)

    def test_parallel_returns_summaries(self, tmp_path: Path) -> None:
        """Returns list of run summary dicts."""
        config = BlockWorldConfig(
            grid_width=10, grid_height=10, n_blocks=10, steps=20
        )
        summaries = run_rules_parallel(
            n_rules=3, out_dir=tmp_path, config=config, n_workers=2
        )
        assert len(summaries) == 3
        assert all("run_id" in s for s in summaries)
