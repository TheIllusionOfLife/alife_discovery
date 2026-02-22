"""Sequential vs synchronous update ablation.

Usage:
    uv run python scripts/synchronous_ablation.py --out-dir data/post_hoc/synchronous_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from objectless_alife.aggregation import run_experiment
from objectless_alife.config import ExperimentConfig, UpdateMode
from objectless_alife.rules import ObservationPhase
from scripts._analysis_common import (
    collect_phase_pairwise_comparisons,
    write_metric_summary_outputs,
)


def _run_mode(
    *,
    out_dir: Path,
    update_mode: UpdateMode,
    n_rules: int,
    steps: int,
    seed_batches: int,
    rule_seed_start: int,
    sim_seed_start: int,
) -> Path:
    mode_dir = out_dir / update_mode.value
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
            update_mode=update_mode,
            rule_seed_start=rule_seed_start,
            sim_seed_start=sim_seed_start,
        )
    )
    return mode_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sequential vs synchronous update ablation")
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/synchronous_ablation"))
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

    sequential_dir = _run_mode(
        out_dir=out_dir,
        update_mode=UpdateMode.SEQUENTIAL,
        n_rules=args.n_rules,
        steps=args.steps,
        seed_batches=args.seed_batches,
        rule_seed_start=args.rule_seed_start,
        sim_seed_start=args.sim_seed_start,
    )
    synchronous_dir = _run_mode(
        out_dir=out_dir,
        update_mode=UpdateMode.SYNCHRONOUS,
        n_rules=args.n_rules,
        steps=args.steps,
        seed_batches=args.seed_batches,
        rule_seed_start=args.rule_seed_start,
        sim_seed_start=args.sim_seed_start,
    )

    comparisons = collect_phase_pairwise_comparisons(
        dir_a=sequential_dir,
        dir_b=synchronous_dir,
        label_a_suffix="sequential",
        label_b_suffix="synchronous",
    )

    summary = {
        "n_rules": args.n_rules,
        "steps": args.steps,
        "seed_batches": args.seed_batches,
        "phase_pairwise": comparisons,
    }
    write_metric_summary_outputs(out_dir=out_dir, summary=summary, comparisons=comparisons)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
