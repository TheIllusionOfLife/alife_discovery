"""Tests for mechanism_analysis.py edge cases."""

from __future__ import annotations

import pyarrow as pa

from scripts.mechanism_analysis import (
    _bond_survival_stats,
    _entity_lifetime_stats,
    _growth_transition_stats,
)


class TestMechanismAnalysisEdgeCases:
    def test_empty_entity_log_lifetime(self, tmp_path: object) -> None:
        """Empty entity table should not crash."""
        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("step", pa.int64()),
                ("entity_hash", pa.string()),
                ("entity_size", pa.int64()),
                ("assembly_index", pa.int64()),
            ]
        )
        table = pa.table(
            {
                "run_id": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int64()),
                "entity_hash": pa.array([], type=pa.string()),
                "entity_size": pa.array([], type=pa.int64()),
                "assembly_index": pa.array([], type=pa.int64()),
            },
            schema=schema,
        )
        lines = _entity_lifetime_stats(table)
        assert any("No data" in line for line in lines)

    def test_empty_timeseries_bond_survival(self) -> None:
        """Empty timeseries should not crash."""
        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("step", pa.int64()),
                ("n_bonds", pa.int64()),
            ]
        )
        table = pa.table(
            {
                "run_id": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int64()),
                "n_bonds": pa.array([], type=pa.int64()),
            },
            schema=schema,
        )
        lines = _bond_survival_stats(table)
        assert any("No data" in line for line in lines)

    def test_empty_entity_log_growth_transitions(self) -> None:
        """Empty entity table should report no transitions."""
        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("step", pa.int64()),
                ("entity_hash", pa.string()),
                ("entity_size", pa.int64()),
            ]
        )
        table = pa.table(
            {
                "run_id": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int64()),
                "entity_hash": pa.array([], type=pa.string()),
                "entity_size": pa.array([], type=pa.int64()),
            },
            schema=schema,
        )
        lines = _growth_transition_stats(table)
        assert any("No size transitions" in line for line in lines)
