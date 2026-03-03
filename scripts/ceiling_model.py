#!/usr/bin/env python3
"""Birth-death size-ceiling model for entity growth.

Computes empirical birth (b_k) and death (d_k) rates per size class from
step-level timeseries data, derives k_max where b_k/d_k < 1, and fits a
compact approximation.

Addresses reviewer concern I5 (mechanism not formalized) by providing a
quantitative equation for the max-size ceiling (~6).

Usage:
    uv run python scripts/ceiling_model.py \
        --input data/assembly_audit/entity_log_combined.parquet \
        --out-dir data/ceiling_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Birth-Death Size Ceiling Model")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/assembly_audit/entity_log_combined.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/ceiling_model"))
    return p.parse_args()


def compute_transition_rates(
    sizes_by_run_step: dict[str, dict[int, list[int]]],
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute empirical birth and death rates per size class.

    b_k = P(entity of size k at step t has size k+1 at step t+1)
    d_k = P(entity of size k at step t has size k-1 at step t+1)

    Since we don't track individual entities across steps, we use the
    aggregate size distribution: b_k is estimated from the rate at which
    entities of size k appear at size k+1 in consecutive snapshots.

    This is a simplified model using size-class frequencies rather than
    individual entity tracking.
    """
    # Collect size distributions per (run, step)
    size_freq_per_step: dict[str, dict[int, dict[int, int]]] = {}
    for run_id, steps in sizes_by_run_step.items():
        size_freq_per_step[run_id] = {}
        for step, sizes in steps.items():
            freq: dict[int, int] = {}
            for s in sizes:
                freq[s] = freq.get(s, 0) + 1
            size_freq_per_step[run_id][step] = freq

    # Estimate transition rates from consecutive step pairs
    growth_count: dict[int, int] = {}  # k → times n(k+1) increased
    shrink_count: dict[int, int] = {}  # k → times n(k-1) increased
    total_count: dict[int, int] = {}  # k → total observations at size k

    for _run_id, step_freqs in size_freq_per_step.items():
        sorted_steps = sorted(step_freqs.keys())
        for i in range(len(sorted_steps) - 1):
            t0 = sorted_steps[i]
            t1 = sorted_steps[i + 1]
            freq0 = step_freqs[t0]
            freq1 = step_freqs[t1]
            all_sizes = set(freq0.keys()) | set(freq1.keys())
            for k in all_sizes:
                n0 = freq0.get(k, 0)
                if n0 == 0:
                    continue
                total_count[k] = total_count.get(k, 0) + n0
                # Growth: entities at k contributing to k+1
                n1_plus = freq1.get(k + 1, 0) - freq0.get(k + 1, 0)
                if n1_plus > 0:
                    growth_count[k] = growth_count.get(k, 0) + min(n1_plus, n0)
                # Shrink: entities at k contributing to k-1
                if k > 1:
                    n1_minus = freq1.get(k - 1, 0) - freq0.get(k - 1, 0)
                    if n1_minus > 0:
                        shrink_count[k] = shrink_count.get(k, 0) + min(n1_minus, n0)

    b_k: dict[int, float] = {}
    d_k: dict[int, float] = {}
    for k, total in sorted(total_count.items()):
        b_k[k] = growth_count.get(k, 0) / total if total > 0 else 0.0
        d_k[k] = shrink_count.get(k, 0) / total if total > 0 else 0.0

    return b_k, d_k


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.input)
    run_ids = table.column("run_id").to_pylist()
    steps = table.column("step").to_pylist()
    sizes = table.column("entity_size").to_pylist()

    # Group by (run_id, step)
    sizes_by_run_step: dict[str, dict[int, list[int]]] = {}
    for rid, step, sz in zip(run_ids, steps, sizes, strict=True):
        sizes_by_run_step.setdefault(rid, {}).setdefault(step, []).append(sz)

    b_k, d_k = compute_transition_rates(sizes_by_run_step)

    # Find k_max: largest k where b_k/d_k >= 1
    lines = ["=== Birth-Death Size Ceiling Model ===", ""]
    lines.append(f"{'k':>3}  {'b_k':>8}  {'d_k':>8}  {'b_k/d_k':>8}  {'grow?':>5}")
    lines.append("-" * 45)

    k_max = 1
    for k in sorted(b_k.keys()):
        bk = b_k[k]
        dk = d_k.get(k, 0.0)
        ratio = bk / dk if dk > 0 else float("inf") if bk > 0 else 0.0
        growing = ratio >= 1.0
        if growing and k > k_max:
            k_max = k
        ratio_str = f"{ratio:.4f}" if ratio < 1000 else "inf"
        grow_str = "yes" if growing else "no"
        lines.append(f"{k:>3}  {bk:>8.5f}  {dk:>8.5f}  {ratio_str:>8}  {grow_str:>5}")

    lines.append("")
    lines.append(f"Predicted k_max (largest k where b_k/d_k >= 1): {k_max}")
    lines.append("")

    # Compact approximation
    # b_k ≈ c(ρ) · p̄ · s^(k-1), d_k ≈ 1 - s^k
    # where s = bond survival probability, p̄ = mean bond formation prob
    # Fit s from observed b_k decay
    ks = sorted(k for k in b_k if b_k[k] > 0 and k > 1)
    if len(ks) >= 2:
        log_bk = np.array([np.log(b_k[k]) for k in ks])
        k_arr = np.array(ks, dtype=float)
        # Linear fit: log(b_k) ≈ log(c·p̄) + (k-1)·log(s)
        coeffs = np.polyfit(k_arr - 1, log_bk, 1)
        s_est = np.exp(coeffs[0])
        c_pbar_est = np.exp(coeffs[1])
        lines.append(f"Fitted bond survival parameter s ≈ {s_est:.4f}")
        lines.append(f"Fitted c·p̄ ≈ {c_pbar_est:.4f}")
        if coeffs[0] != 0 and np.log(s_est) != 0:
            lines.append(f"Predicted k_max from s: ~{int(-np.log(2) / np.log(s_est)) + 1}")
        else:
            lines.append("Predicted k_max: undefined (flat birth-rate fit)")
    else:
        lines.append("Insufficient data for compact approximation fit")

    report = "\n".join(lines)
    report_path = args.out_dir / "ceiling_model_report.txt"
    report_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
