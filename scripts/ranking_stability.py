"""Kendall tau ranking stability across simulation seed batches.

Usage:
    uv run python scripts/ranking_stability.py --out-dir data/post_hoc/ranking_stability
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import warnings
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from scipy.stats import kendalltau

from objectless_alife.config import SearchConfig
from objectless_alife.rules import ObservationPhase
from objectless_alife.run_search import run_batch_search
from objectless_alife.stats import load_final_step_metrics

RULE_ID_SEED_RE = re.compile(r"^phase\d+_rs(?P<rule_seed>\d+)_ss\d+$")
# Prevent seed-range overlap across batches when deriving per-batch base_sim_seed.
BATCH_SEED_SPACING = 100_000


def _alignment_id(rule_id: str, alignment_key: str) -> str:
    if alignment_key == "rule_id":
        return rule_id
    if alignment_key == "rule_seed":
        # Keep this regex in sync with rule_id formatting in simulation payloads.
        match = RULE_ID_SEED_RE.match(rule_id)
        if match is None:
            warnings.warn(
                f"_alignment_id: rule_id did not match RULE_ID_SEED_RE for "
                f"alignment_key={alignment_key!r}: {rule_id!r}",
                stacklevel=2,
            )
            return rule_id
        return f"rule_seed:{match.group('rule_seed')}"
    raise ValueError(f"Unhandled alignment key: {alignment_key}")


def _rank_map(metrics_path: Path, alignment_key: str) -> dict[str, int]:
    table = load_final_step_metrics(metrics_path)
    mi = pc.fill_null(
        pc.cast(table.column("neighbor_mutual_information"), pa.float64(), safe=False),
        pa.scalar(0.0),
    )
    null = pc.fill_null(
        pc.cast(table.column("mi_shuffle_null"), pa.float64(), safe=False),
        pa.scalar(0.0),
    )
    diff = pc.subtract(mi, null)
    finite = pc.if_else(pc.is_finite(diff), diff, pa.scalar(0.0))
    excess = pc.max_element_wise(finite, pa.scalar(0.0))
    rows = list(
        zip(
            table.column("rule_id").to_pylist(),
            excess.to_pylist(),
            strict=True,
        )
    )
    rows.sort(key=lambda pair: float(pair[1]), reverse=True)
    rank_map: dict[str, int] = {}
    # rows is sorted by descending excess; when multiple rule_id values collapse to the
    # same alignment_key (e.g. rule_seed), first-seen wins so we keep the highest-excess one.
    for rule_id, _ in rows:
        key = _alignment_id(rule_id, alignment_key)
        if key not in rank_map:
            rank_map[key] = len(rank_map)
    return rank_map


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kendall tau ranking stability")
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/ranking_stability"))
    parser.add_argument("--n-rules", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-seed-batches", type=int, default=3)
    parser.add_argument("--rule-seed-start", type=int, default=0)
    parser.add_argument("--sim-seed-start", type=int, default=0)
    parser.add_argument(
        "--alignment-key",
        type=str,
        choices=("rule_seed", "rule_id"),
        default="rule_seed",
        help="Key used to align rules across seed batches when computing Kendall tau.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    args = parser.parse_args(argv)
    if args.quick:
        args.n_rules = 10
        args.steps = 20
        args.n_seed_batches = 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = (
        ObservationPhase.PHASE1_DENSITY,
        ObservationPhase.PHASE2_PROFILE,
        ObservationPhase.CONTROL_DENSITY_CLOCK,
    )

    batch_rankings: dict[int, dict[int, dict[str, int]]] = {}
    for phase in phases:
        per_phase: dict[int, dict[str, int]] = {}
        for batch_idx in range(args.n_seed_batches):
            batch_dir = out_dir / f"phase_{phase.value}" / f"batch_{batch_idx}"
            run_batch_search(
                n_rules=args.n_rules,
                phase=phase,
                out_dir=batch_dir,
                base_rule_seed=args.rule_seed_start,
                base_sim_seed=args.sim_seed_start + (batch_idx * BATCH_SEED_SPACING),
                config=SearchConfig(steps=args.steps),
            )
            per_phase[batch_idx] = _rank_map(
                batch_dir / "logs" / "metrics_summary.parquet",
                alignment_key=args.alignment_key,
            )
        batch_rankings[phase.value] = per_phase

    phase_results: dict[str, list[dict[str, float | int | str | None]]] = {}
    for phase in phases:
        rows: list[dict[str, float | int | str | None]] = []
        per_phase = batch_rankings[phase.value]
        for a, b in itertools.combinations(sorted(per_phase.keys()), 2):
            rank_a = per_phase[a]
            rank_b = per_phase[b]
            shared = sorted(set(rank_a) & set(rank_b))
            overlap_fraction_a = len(shared) / len(rank_a) if rank_a else 0.0
            overlap_fraction_b = len(shared) / len(rank_b) if rank_b else 0.0
            if len(shared) < 2:
                tau: float | None = None
            else:
                series_a = [rank_a[rid] for rid in shared]
                series_b = [rank_b[rid] for rid in shared]
                tau = float(kendalltau(series_a, series_b).statistic)
            rows.append(
                {
                    "batch_a": a,
                    "batch_b": b,
                    "kendall_tau": tau,
                    "n_rules": len(shared),
                    "overlap_fraction_a": overlap_fraction_a,
                    "overlap_fraction_b": overlap_fraction_b,
                    "alignment_key": args.alignment_key,
                }
            )
        phase_results[str(phase.value)] = rows

    output = {
        "n_rules": args.n_rules,
        "n_seed_batches": args.n_seed_batches,
        "steps": args.steps,
        "alignment_key": args.alignment_key,
        "pairwise_kendall_tau": phase_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    csv_rows: list[dict[str, str | float | int | None]] = []
    for phase, rows in phase_results.items():
        for row in rows:
            csv_rows.append(
                {
                    "phase": phase,
                    "batch_a": int(row["batch_a"]),
                    "batch_b": int(row["batch_b"]),
                    "kendall_tau": "" if row["kendall_tau"] is None else row["kendall_tau"],
                    "n_rules": int(row["n_rules"]),
                    "overlap_fraction_a": float(row["overlap_fraction_a"]),
                    "overlap_fraction_b": float(row["overlap_fraction_b"]),
                    "alignment_key": str(row["alignment_key"]),
                }
            )
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "batch_a",
                "batch_b",
                "kendall_tau",
                "n_rules",
                "overlap_fraction_a",
                "overlap_fraction_b",
                "alignment_key",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
