"""Sequential vs synchronous update ablation.

Usage:
    uv run python scripts/synchronous_ablation.py --out-dir data/post_hoc/synchronous_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from objectless_alife.config import ExperimentConfig, UpdateMode
from objectless_alife.rules import ObservationPhase
from objectless_alife.run_search import run_experiment
from objectless_alife.stats import run_pairwise_analysis


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

    comparisons: dict[str, dict] = {}
    for phase in ("phase_1", "phase_2", "phase_3"):
        comparisons[phase] = run_pairwise_analysis(
            metrics_a=sequential_dir / phase / "logs" / "metrics_summary.parquet",
            metrics_b=synchronous_dir / phase / "logs" / "metrics_summary.parquet",
            rules_a=sequential_dir / phase / "rules",
            rules_b=synchronous_dir / phase / "rules",
            label_a=f"{phase}_sequential",
            label_b=f"{phase}_synchronous",
        )

    summary = {
        "n_rules": args.n_rules,
        "steps": args.steps,
        "seed_batches": args.seed_batches,
        "phase_pairwise": comparisons,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    csv_rows: list[dict[str, str | float]] = []
    for phase, phase_result in comparisons.items():
        metric_tests = phase_result.get("metric_tests", {})
        for metric_name, metric_payload in metric_tests.items():
            csv_rows.append(
                {
                    "phase": phase,
                    "metric": metric_name,
                    "p_value": float(metric_payload.get("p_value", float("nan"))),
                    "p_value_corrected": float(
                        metric_payload.get("p_value_corrected", float("nan"))
                    ),
                    "effect_size_r": float(metric_payload.get("effect_size_r", float("nan"))),
                }
            )
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "metric", "p_value", "p_value_corrected", "effect_size_r"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
