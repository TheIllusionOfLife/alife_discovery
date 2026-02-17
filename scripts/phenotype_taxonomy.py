"""Build a lightweight phenotype taxonomy for high-MI Phase 2 rules.

Usage:
    uv run python scripts/phenotype_taxonomy.py \
      --data-dir data/stage_d \
      --out-dir data/post_hoc/phenotypes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from objectless_alife.stats import load_final_step_metrics


def _classify_row(row: dict[str, float | str]) -> str:
    mi_excess = float(row["mi_excess"])
    entropy = float(row["state_entropy"])
    adjacency = float(row["same_state_adjacency_fraction"])
    predictability = float(row["predictability_hamming"])
    if mi_excess >= 0.12 and adjacency >= 0.60:
        return "polarized_cluster"
    if entropy <= 0.30 and predictability <= 0.20:
        return "frozen_patch"
    if mi_excess >= 0.05 and entropy >= 0.60 and predictability >= 0.40:
        return "mixed_turbulent"
    return "low_signal"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phenotype taxonomy for high-MI Phase 2 rules")
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage_d"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/post_hoc/phenotypes"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Run a small sanity-sized preset")
    args = parser.parse_args(argv)
    if args.quick:
        args.top_k = 10

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.data_dir) / "phase_2" / "logs" / "metrics_summary.parquet"
    table = load_final_step_metrics(metrics_path)

    mi = pc.cast(table.column("neighbor_mutual_information"), pa.float64(), safe=False)
    null = pc.cast(table.column("mi_shuffle_null"), pa.float64(), safe=False)
    diff = pc.subtract(mi, null)
    finite_diff = pc.if_else(pc.is_finite(diff), diff, pa.scalar(0.0))
    mi_excess = pc.max_element_wise(finite_diff, pa.scalar(0.0))
    if "mi_excess" in table.column_names:
        enriched = table.set_column(
            table.column_names.index("mi_excess"),
            "mi_excess",
            mi_excess,
        )
    else:
        enriched = table.append_column("mi_excess", mi_excess)

    rows = enriched.select(
        [
            "rule_id",
            "state_entropy",
            "predictability_hamming",
            "same_state_adjacency_fraction",
            "mi_excess",
        ]
    ).to_pylist()
    rows.sort(key=lambda row: float(row["mi_excess"]), reverse=True)
    top_rows = rows[: args.top_k]

    classified: list[dict[str, float | str]] = []
    for row in top_rows:
        normalized = {
            "rule_id": str(row["rule_id"]),
            "state_entropy": (
                float(row["state_entropy"]) if row["state_entropy"] is not None else 0.0
            ),
            "predictability_hamming": float(row["predictability_hamming"])
            if row["predictability_hamming"] is not None
            else 0.0,
            "same_state_adjacency_fraction": float(row["same_state_adjacency_fraction"])
            if row["same_state_adjacency_fraction"] is not None
            else 0.0,
            "mi_excess": float(row["mi_excess"]),
        }
        normalized["phenotype"] = _classify_row(normalized)
        classified.append(normalized)

    counts: dict[str, int] = {}
    representatives: dict[str, str] = {}
    for row in classified:
        phenotype = str(row["phenotype"])
        counts[phenotype] = counts.get(phenotype, 0) + 1
        if phenotype not in representatives:
            representatives[phenotype] = str(row["rule_id"])

    payload = {
        "top_k": args.top_k,
        "counts": counts,
        "representative_rule_ids": representatives,
        "rows": classified,
    }
    (out_dir / "taxonomy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with (out_dir / "taxonomy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rule_id",
                "state_entropy",
                "predictability_hamming",
                "same_state_adjacency_fraction",
                "mi_excess",
                "phenotype",
            ],
        )
        writer.writeheader()
        writer.writerows(classified)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
