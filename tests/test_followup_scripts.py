import json
from pathlib import Path

from objectless_alife.config import ExperimentConfig
from objectless_alife.rules import ObservationPhase
from objectless_alife.run_search import run_experiment
from scripts.no_filter_analysis import main as no_filter_main
from scripts.phenotype_taxonomy import main as taxonomy_main
from scripts.ranking_stability import main as ranking_main
from scripts.synchronous_ablation import main as sync_main
from scripts.te_null_analysis import main as te_main


def _make_small_dataset(data_dir: Path) -> None:
    run_experiment(
        ExperimentConfig(
            phases=(
                ObservationPhase.PHASE1_DENSITY,
                ObservationPhase.PHASE2_PROFILE,
                ObservationPhase.CONTROL_DENSITY_CLOCK,
            ),
            n_rules=2,
            n_seed_batches=1,
            out_dir=data_dir,
            steps=5,
            rule_seed_start=0,
            sim_seed_start=0,
        )
    )


def test_no_filter_analysis_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "nf"
    no_filter_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "1",
            "--steps",
            "4",
            "--seed-batches",
            "1",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "phase_pairwise" in payload


def test_synchronous_ablation_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "sync"
    sync_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "1",
            "--steps",
            "4",
            "--seed-batches",
            "1",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "phase_pairwise" in payload


def test_ranking_stability_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "stability"
    ranking_main(
        [
            "--out-dir",
            str(out_dir),
            "--n-rules",
            "2",
            "--steps",
            "4",
            "--n-seed-batches",
            "2",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "pairwise_kendall_tau" in payload


def test_te_null_analysis_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "te"
    te_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
            "--n-shuffles",
            "10",
        ]
    )
    payload = json.loads((out_dir / "summary.json").read_text())
    assert "conditions" in payload


def test_phenotype_taxonomy_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_tax"
    _make_small_dataset(data_dir)
    out_dir = tmp_path / "taxonomy"
    taxonomy_main(
        [
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--top-k",
            "2",
        ]
    )
    payload = json.loads((out_dir / "taxonomy.json").read_text())
    assert "rows" in payload
