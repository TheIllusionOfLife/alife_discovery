"""Transfer-entropy shuffle-null calibration analysis.

Usage:
    uv run python scripts/te_null_analysis.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/te_null
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from objectless_alife.metrics import neighbor_transfer_entropy, transfer_entropy_shuffle_null
from objectless_alife.run_search import select_top_rules_by_excess_mi

GRID_WIDTH = 20
GRID_HEIGHT = 20


def _sim_log_for_rule_ids(
    sim_log_path: Path, rule_ids: set[str]
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    table = pq.read_table(
        sim_log_path,
        columns=["rule_id", "step", "agent_id", "x", "y", "state"],
    )
    rows = table.to_pylist()
    out: dict[str, list[tuple[int, int, int, int, int]]] = {rid: [] for rid in rule_ids}
    for row in rows:
        rid = str(row["rule_id"])
        if rid not in out:
            continue
        out[rid].append(
            (
                int(row["step"]),
                int(row["agent_id"]),
                int(row["x"]),
                int(row["y"]),
                int(row["state"]),
            )
        )
    return out


def _rule_ids_for_phase(phase: int, seeds: list[int]) -> set[str]:
    return {f"phase{phase}_rs{seed}_ss{seed}" for seed in seeds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TE shuffle-null calibration analysis")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/te_null"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-shuffles", type=int, default=200)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_metrics = data_dir / "phase_2" / "logs" / "metrics_summary.parquet"
    p2_rules = data_dir / "phase_2" / "rules"
    top_seeds = select_top_rules_by_excess_mi(p2_metrics, p2_rules, top_k=args.top_k)

    control_log = data_dir / "control" / "logs" / "simulation_log.parquet"
    if not control_log.exists():
        control_log = data_dir / "phase_3" / "logs" / "simulation_log.parquet"
    conditions = {
        "phase_1": (1, data_dir / "phase_1" / "logs" / "simulation_log.parquet"),
        "phase_2": (2, data_dir / "phase_2" / "logs" / "simulation_log.parquet"),
        "control": (3, control_log),
    }
    summary: dict[str, dict[str, float | int]] = {}

    for label, (phase, sim_log_path) in conditions.items():
        rule_ids = _rule_ids_for_phase(phase, top_seeds)
        logs_by_rule = _sim_log_for_rule_ids(sim_log_path, rule_ids)
        te_vals: list[float] = []
        te_null_vals: list[float] = []
        te_excess_vals: list[float] = []
        for idx, tuples in enumerate(logs_by_rule.values()):
            if not tuples:
                continue
            te = neighbor_transfer_entropy(tuples, GRID_WIDTH, GRID_HEIGHT)
            te_null = transfer_entropy_shuffle_null(
                tuples,
                GRID_WIDTH,
                GRID_HEIGHT,
                n_shuffles=args.n_shuffles,
                rng=random.Random(idx),
            )
            te_excess = max(te - te_null, 0.0)
            te_vals.append(te)
            te_null_vals.append(te_null)
            te_excess_vals.append(te_excess)
        summary[label] = {
            "n_rules": len(te_vals),
            "te_median": statistics.median(te_vals) if te_vals else 0.0,
            "te_null_median": statistics.median(te_null_vals) if te_null_vals else 0.0,
            "te_excess_median": statistics.median(te_excess_vals) if te_excess_vals else 0.0,
        }

    payload = {
        "top_k": args.top_k,
        "n_shuffles": args.n_shuffles,
        "conditions": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
