from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rules import ObservationPhase
from src.run_search import run_batch_search
from src.visualize import render_rule_animation


def test_render_rule_animation_creates_gif(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    output_path = tmp_path / "preview.gif"
    render_rule_animation(
        simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
        metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
        rule_json_path=rule_json,
        output_path=output_path,
        fps=2,
    )
    assert output_path.exists()


def test_render_rule_animation_rejects_paths_outside_base_dir(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=3,
        base_rule_seed=1,
        base_sim_seed=1,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    outside_output = tmp_path.parent / "outside.gif"
    with pytest.raises(ValueError):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
            rule_json_path=rule_json,
            output_path=outside_output,
            fps=2,
            base_dir=tmp_path,
        )


def test_render_rule_animation_rejects_missing_rule_id(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=3,
        base_rule_seed=1,
        base_sim_seed=1,
    )
    rule_json = next((tmp_path / "rules").glob("*.json"))
    rule_json.write_text("{}")
    with pytest.raises(ValueError, match="rule_id"):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=tmp_path / "logs" / "metrics_summary.parquet",
            rule_json_path=rule_json,
            output_path=tmp_path / "preview.gif",
            fps=2,
        )


def test_render_rule_animation_rejects_missing_metric_steps(tmp_path: Path) -> None:
    run_batch_search(
        n_rules=1,
        phase=ObservationPhase.PHASE1_DENSITY,
        out_dir=tmp_path,
        steps=4,
        base_rule_seed=2,
        base_sim_seed=3,
    )
    metric_path = tmp_path / "logs" / "metrics_summary.parquet"
    table = pq.read_table(metric_path)
    rows = table.to_pylist()
    max_step = max(int(row["step"]) for row in rows)
    filtered = [row for row in rows if int(row["step"]) != max_step]
    pq.write_table(pa.Table.from_pylist(filtered), metric_path)

    rule_json = next((tmp_path / "rules").glob("*.json"))
    with pytest.raises(ValueError, match="Missing metrics for steps"):
        render_rule_animation(
            simulation_log_path=tmp_path / "logs" / "simulation_log.parquet",
            metrics_summary_path=metric_path,
            rule_json_path=rule_json,
            output_path=tmp_path / "preview.gif",
            fps=2,
        )
