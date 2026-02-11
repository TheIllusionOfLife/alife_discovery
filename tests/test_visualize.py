from pathlib import Path

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
