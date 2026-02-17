"""Shared helpers for follow-up pairwise analysis scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from objectless_alife.stats import run_pairwise_analysis

PHASE_KEYS = ("phase_1", "phase_2", "phase_3")


def collect_phase_pairwise_comparisons(
    *,
    dir_a: Path,
    dir_b: Path,
    label_a_suffix: str,
    label_b_suffix: str,
) -> dict[str, dict]:
    """Run pairwise tests for each canonical phase key."""
    comparisons: dict[str, dict] = {}
    for phase in PHASE_KEYS:
        comparisons[phase] = run_pairwise_analysis(
            metrics_a=dir_a / phase / "logs" / "metrics_summary.parquet",
            metrics_b=dir_b / phase / "logs" / "metrics_summary.parquet",
            rules_a=dir_a / phase / "rules",
            rules_b=dir_b / phase / "rules",
            label_a=f"{phase}_{label_a_suffix}",
            label_b=f"{phase}_{label_b_suffix}",
        )
    return comparisons


def write_metric_summary_outputs(
    *,
    out_dir: Path,
    summary: dict,
    comparisons: dict[str, dict],
) -> None:
    """Write standard JSON + CSV artifacts for metric comparison summaries."""
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    csv_rows: list[dict[str, str | float | int]] = []
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
