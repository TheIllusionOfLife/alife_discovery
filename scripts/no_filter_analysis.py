"""Run paper-scale filtered vs no-filter comparison across conditions.

Usage:
    uv run python scripts/no_filter_analysis.py --out-dir data/post_hoc/no_filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from objectless_alife.aggregation import run_experiment
from objectless_alife.config import ExperimentConfig
from objectless_alife.rules import ObservationPhase
from scripts._analysis_common import (
    collect_phase_pairwise_comparisons,
    write_metric_summary_outputs,
)


def _survival_rate(runs_path: Path) -> float:
    table = pq.read_table(runs_path, columns=["survived"])
    survived_scalar = pc.sum(pc.cast(table.column("survived"), pa.int64()))
    survived = int(survived_scalar.as_py()) if survived_scalar is not None else 0
    total = table.num_rows
    return 0.0 if total == 0 else survived / total


def _run_condition(
    *,
    out_dir: Path,
    enable_viability_filters: bool,
    n_rules: int,
    steps: int,
    seed_batches: int,
    rule_seed_start: int,
    sim_seed_start: int,
) -> Path:
    mode_dir = out_dir / ("filtered" if enable_viability_filters else "no_filter")
    run_experiment(
        ExperimentConfig(
            phases=(
                ObservationPhase.PHASE1_DENSITY,
                ObservationPhase.PHASE2_PROFILE,
                ObservationPhase.CONTROL_DENSITY_CLOCK,
            ),
            n_rules=n_rules,
            n_seed_batches=seed_batches,
            out_dir=mode_dir,
            steps=steps,
            enable_viability_filters=enable_viability_filters,
            rule_seed_start=rule_seed_start,
            sim_seed_start=sim_seed_start,
        )
    )
    return mode_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filtered vs no-filter supplementary analysis")
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/no_filter"))
    parser.add_argument("--n-rules", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed-batches", type=int, default=1)
    parser.add_argument("--rule-seed-start", type=int, default=0)
    parser.add_argument("--sim-seed-start", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_rules = 10
        args.steps = 20
        args.seed_batches = 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = _run_condition(
        out_dir=out_dir,
        enable_viability_filters=True,
        n_rules=args.n_rules,
        steps=args.steps,
        seed_batches=args.seed_batches,
        rule_seed_start=args.rule_seed_start,
        sim_seed_start=args.sim_seed_start,
    )
    no_filter_dir = _run_condition(
        out_dir=out_dir,
        enable_viability_filters=False,
        n_rules=args.n_rules,
        steps=args.steps,
        seed_batches=args.seed_batches,
        rule_seed_start=args.rule_seed_start,
        sim_seed_start=args.sim_seed_start,
    )

    comparisons = collect_phase_pairwise_comparisons(
        dir_a=filtered_dir,
        dir_b=no_filter_dir,
        label_a_suffix="filtered",
        label_b_suffix="no_filter",
    )

    summary = {
        "n_rules": args.n_rules,
        "steps": args.steps,
        "seed_batches": args.seed_batches,
        "filtered_survival_rate": _survival_rate(filtered_dir / "logs" / "experiment_runs.parquet"),
        "no_filter_survival_rate": _survival_rate(
            no_filter_dir / "logs" / "experiment_runs.parquet"
        ),
        "phase_pairwise": comparisons,
    }
    write_metric_summary_outputs(out_dir=out_dir, summary=summary, comparisons=comparisons)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
