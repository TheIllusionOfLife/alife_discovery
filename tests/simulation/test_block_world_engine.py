"""Tests for block-world simulation engine (run_block_world_search)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

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
