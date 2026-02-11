import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.rules import ObservationPhase
from src.run_search import SearchConfig, run_batch_search
from src.world import WorldConfig


def test_run_batch_search_writes_json_and_parquet(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=10,
        halt_window=3,
        base_rule_seed=10,
        base_sim_seed=20,
    )

    assert len(results) == 2

    rules_dir = tmp_path / "rules"
    logs_dir = tmp_path / "logs"
    assert rules_dir.exists()
    assert logs_dir.exists()

    json_files = list(rules_dir.glob("*.json"))
    assert len(json_files) == 2

    sim_table = pq.read_table(logs_dir / "simulation_log.parquet")
    metric_table = pq.read_table(logs_dir / "metrics_summary.parquet")

    sim_columns = {"rule_id", "step", "agent_id", "x", "y", "state", "action"}
    metric_columns = {
        "rule_id",
        "step",
        "state_entropy",
        "compression_ratio",
        "predictability_hamming",
        "morans_i",
        "cluster_count",
        "quasi_periodicity_peaks",
        "phase_transition_max_delta",
        "neighbor_mutual_information",
        "action_entropy_mean",
        "action_entropy_variance",
        "block_ncd",
    }
    assert sim_columns.issubset(sim_table.column_names)
    assert metric_columns.issubset(metric_table.column_names)


def test_run_batch_search_deterministic_rule_ids(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=2,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=tmp_path,
        steps=5,
        base_rule_seed=7,
        base_sim_seed=11,
    )

    rules_dir = tmp_path / "rules"
    file_names = sorted(path.stem for path in rules_dir.glob("*.json"))
    assert file_names == ["phase2_rs7_ss11", "phase2_rs8_ss12"]


def test_run_batch_search_steps_conflict_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_batch_search(
            n_rules=1,
            phase=ObservationPhase.PHASE1_DENSITY,
            out_dir=tmp_path,
            steps=200,
            world_config=WorldConfig(steps=123),
        )


def test_run_batch_search_phase2_executes(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=tmp_path,
        steps=10,
        base_rule_seed=1,
        base_sim_seed=2,
    )
    assert len(results) == 1

    rules_dir = tmp_path / "rules"
    payload = json.loads(next(rules_dir.glob("*.json")).read_text())
    assert payload["metadata"]["observation_phase"] == 2


def test_run_batch_search_accepts_search_config_with_optional_filters(tmp_path: Path) -> None:
    results = run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        config=SearchConfig(
            steps=10,
            halt_window=3,
            filter_short_period=True,
            short_period_max_period=2,
            short_period_history_size=6,
            filter_low_activity=True,
            low_activity_window=3,
            low_activity_min_unique_ratio=0.2,
        ),
    )
    assert len(results) == 1
    payload = json.loads(next((tmp_path / "rules").glob("*.json")).read_text())
    assert "short_period" in payload["filter_results"]
    assert "low_activity" in payload["filter_results"]


def test_run_batch_search_persists_grid_dimensions_in_metadata(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=5,
        world_config=WorldConfig(grid_width=11, grid_height=13, steps=5),
    )

    payload = json.loads(next((tmp_path / "rules").glob("*.json")).read_text())
    metadata = payload["metadata"]
    assert metadata["grid_width"] == 11
    assert metadata["grid_height"] == 13


def test_run_batch_search_termination_metadata_is_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=out_a,
        steps=15,
        base_rule_seed=30,
        base_sim_seed=40,
    )
    run_batch_search(
        n_rules=3,
        phase=ObservationPhase.PHASE2_PROFILE,
        out_dir=out_b,
        steps=15,
        base_rule_seed=30,
        base_sim_seed=40,
    )

    payloads_a = {
        path.stem: json.loads(path.read_text()) for path in sorted((out_a / "rules").glob("*.json"))
    }
    payloads_b = {
        path.stem: json.loads(path.read_text()) for path in sorted((out_b / "rules").glob("*.json"))
    }

    assert payloads_a.keys() == payloads_b.keys()
    for rule_id in payloads_a:
        metadata_a = payloads_a[rule_id]["metadata"]
        metadata_b = payloads_b[rule_id]["metadata"]
        assert metadata_a["terminated_at"] == metadata_b["terminated_at"]
        assert metadata_a["termination_reason"] == metadata_b["termination_reason"]
