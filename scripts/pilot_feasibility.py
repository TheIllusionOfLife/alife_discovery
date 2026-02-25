#!/usr/bin/env python3
"""PR #4 — Feasibility Pilot.

Validates two things before committing to full Experiment 1:
1. AT timing benchmark: is assembly_index_exact() tractable for entity sizes
   up to MAX_ENTITY_SIZE?
2. Pilot simulation: what entity/assembly-index distributions arise under
   default settings?

Usage:
    # Smoke test (fast)
    uv run python scripts/pilot_feasibility.py \
        --n-rules 5 --seeds 1 --steps 50 --out-dir tmp/pilot_smoke

    # Full pilot
    uv run python scripts/pilot_feasibility.py \
        --n-rules 100 --seeds 2 --out-dir data/pilot
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from alife_discovery.config.constants import MAX_ENTITY_SIZE
from alife_discovery.config.types import BlockWorldConfig
from alife_discovery.metrics.assembly import assembly_index_exact
from alife_discovery.simulation.engine import run_block_world_search

# ---------------------------------------------------------------------------
# Section 1: AT timing benchmark
# ---------------------------------------------------------------------------


def _path_graph(n: int) -> nx.Graph:
    """Labeled path graph P_n with block_type='M' on all nodes."""
    g = nx.path_graph(n)
    nx.set_node_attributes(g, "M", "block_type")
    return g


def _complete_graph(n: int) -> nx.Graph:
    """Labeled complete graph K_n with block_type='M' on all nodes."""
    g = nx.complete_graph(n)
    nx.set_node_attributes(g, "M", "block_type")
    return g


def run_at_benchmark(at_max_n: int, at_warn_ms: float) -> None:
    """Time assembly_index_exact() on P_n and K_n for n = 2 .. at_max_n.

    Stops benchmarking a graph family once it exceeds at_warn_ms, since the
    exponential DP cost makes larger graphs impractical to measure inline.
    """
    print("=== AT Timing Benchmark ===")
    header = (
        f"{'n':>3}  {'P_n a':>6}  {'P_n ms':>9}  {'K_n a':>6}  {'K_n ms':>9}"
    )
    print(header)
    print("-" * len(header))

    skip_path = False
    skip_complete = False

    for n in range(2, at_max_n + 1):
        path_str = "     ---        ---"
        complete_str = "     ---        ---"

        if not skip_path:
            pg = _path_graph(n)
            t0 = time.perf_counter()
            a_path = assembly_index_exact(pg)
            ms_path = (time.perf_counter() - t0) * 1000
            path_str = f"{a_path:>6}  {ms_path:>9.2f}"
            if ms_path > at_warn_ms:
                warnings.warn(
                    f"P_{n}: assembly_index_exact took {ms_path:.1f} ms"
                    f" (threshold {at_warn_ms} ms); skipping larger P_n",
                    RuntimeWarning,
                    stacklevel=1,
                )
                skip_path = True

        if not skip_complete:
            kg = _complete_graph(n)
            t0 = time.perf_counter()
            a_complete = assembly_index_exact(kg)
            ms_complete = (time.perf_counter() - t0) * 1000
            complete_str = f"{a_complete:>6}  {ms_complete:>9.2f}"
            if ms_complete > at_warn_ms:
                warnings.warn(
                    f"K_{n}: assembly_index_exact took {ms_complete:.1f} ms"
                    f" (threshold {at_warn_ms} ms); skipping larger K_n",
                    RuntimeWarning,
                    stacklevel=1,
                )
                skip_complete = True

        print(f"{n:>3}  {path_str}  {complete_str}")

        if skip_path and skip_complete:
            print(f"  (both graph families exceeded {at_warn_ms:.0f} ms — stopping benchmark)")
            break

    print()


# ---------------------------------------------------------------------------
# Section 2: Pilot simulation run
# ---------------------------------------------------------------------------


def run_pilot(
    n_rules: int,
    seeds: int,
    steps: int,
    out_dir: Path,
) -> list[Path]:
    """Run pilot simulations; return list of entity_log.parquet file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_paths: list[Path] = []

    for seed in range(seeds):
        seed_dir = out_dir / f"seed_{seed}"
        config = BlockWorldConfig(
            steps=steps,
            sim_seed=seed,
        )
        run_block_world_search(
            n_rules=n_rules,
            out_dir=seed_dir,
            config=config,
        )
        log_path = seed_dir / "logs" / "entity_log.parquet"
        if log_path.exists():
            log_paths.append(log_path)

    return log_paths


# ---------------------------------------------------------------------------
# Section 3: Summary report
# ---------------------------------------------------------------------------


def print_summary(
    n_rules: int,
    seeds: int,
    steps: int,
    log_paths: list[Path],
) -> None:
    """Print summary statistics from combined entity logs."""
    print(f"=== Pilot: {n_rules} rules × {seeds} seeds, {steps} steps ===")

    if not log_paths:
        print("No entity logs found — no entities were detected.")
        return

    tables = [pq.read_table(f) for f in log_paths]
    combined = pa.concat_tables(tables)

    if len(combined) == 0:
        print("Entity logs empty — no entities were detected.")
        return

    sizes = combined.column("entity_size").to_pylist()
    ai_col = combined.column("assembly_index").to_pylist()
    cn_col = combined.column("copy_number_at_step").to_pylist()

    # Size distribution
    size_dist: dict[int, int] = {}
    for s in sizes:
        size_dist[s] = size_dist.get(s, 0) + 1
    print(f"Entity size distribution: {dict(sorted(size_dist.items()))}")
    print(f"Max observed size: {max(sizes)}  (MAX_ENTITY_SIZE={MAX_ENTITY_SIZE})")

    # Assembly index distribution
    ai_sorted = sorted(ai_col)
    n = len(ai_sorted)
    ai_min = ai_sorted[0]
    ai_max = ai_sorted[-1]
    ai_p25 = ai_sorted[n // 4]
    ai_p50 = ai_sorted[n // 2]
    ai_p75 = ai_sorted[3 * n // 4]
    ai_mean = sum(ai_col) / n
    print(
        f"Assembly index:"
        f" min={ai_min} p25={ai_p25} p50={ai_p50}"
        f" p75={ai_p75} max={ai_max} mean={ai_mean:.2f}"
    )

    # P(discovery) estimate
    n_discovered = sum(
        1 for a, c in zip(ai_col, cn_col, strict=True) if a >= 2 and c >= 2
    )
    p_discovery = n_discovered / n
    print(
        f"P(discovery | a>=2, cn>=2) = {p_discovery:.3f}"
        f"  ({n_discovered}/{n})"
    )

    # Approx path usage
    n_approx = sum(1 for s in sizes if s > MAX_ENTITY_SIZE)
    print(
        f"Entities exceeding MAX_ENTITY_SIZE={MAX_ENTITY_SIZE}:"
        f" {n_approx} (approx path used)"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PR #4 Feasibility Pilot: AT benchmark + pilot run"
    )
    p.add_argument(
        "--n-rules", type=int, default=100,
        help="Rule table samples per seed (default: 100)",
    )
    p.add_argument(
        "--seeds", type=int, default=2,
        help="Number of sim seeds (default: 2)",
    )
    p.add_argument(
        "--steps", type=int, default=200,
        help="Steps per simulation (default: 200)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/pilot"),
        help="Output directory (default: data/pilot)",
    )
    p.add_argument(
        "--at-max-n", type=int, default=16,
        help="Max graph size for AT timing benchmark (default: 16)",
    )
    p.add_argument(
        "--at-warn-ms", type=float, default=1000.0,
        help="Warn if AT call exceeds this ms threshold (default: 1000)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_at_benchmark(args.at_max_n, args.at_warn_ms)

    log_paths = run_pilot(args.n_rules, args.seeds, args.steps, args.out_dir)

    print_summary(args.n_rules, args.seeds, args.steps, log_paths)


if __name__ == "__main__":
    main()
