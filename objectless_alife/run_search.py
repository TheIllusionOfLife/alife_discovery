"""CLI entrypoint for search execution.

After refactoring, this thin module owns only the CLI argument parsing
and mode dispatch.  All domain logic lives in the extracted modules:

- ``src.schemas``      – Parquet schemas & metric-name constants
- ``src.config``       – configuration dataclasses
- ``src.simulation``   – ``run_batch_search`` engine
- ``src.aggregation``  – experiment / density-sweep / multi-seed orchestration

For backward compatibility every public symbol is re-exported here so
that existing ``from objectless_alife.run_search import X`` imports keep working.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
# ---------------------------------------------------------------------------
from objectless_alife.aggregation import (  # noqa: F401
    _build_phase_comparison,
    _build_phase_summary,
    _collect_final_metric_rows,
    run_density_sweep,
    run_experiment,
    run_halt_window_sweep,
    run_multi_seed_robustness,
    select_top_rules_by_excess_mi,
)
from objectless_alife.config import (  # noqa: F401
    MAX_EXPERIMENT_WORK_UNITS,
    DensitySweepConfig,
    ExperimentConfig,
    HaltWindowSweepConfig,
    MultiSeedConfig,
    SearchConfig,
    SimulationResult,
)
from objectless_alife.rules import ObservationPhase
from objectless_alife.schemas import (  # noqa: F401
    AGGREGATE_SCHEMA_VERSION,
    DENSITY_PHASE_COMPARISON_SCHEMA,
    DENSITY_PHASE_SUMMARY_SCHEMA,
    DENSITY_SWEEP_RUNS_SCHEMA,
    DENSITY_SWEEP_SCHEMA_VERSION,
    HALT_WINDOW_SWEEP_SCHEMA,
    METRICS_SCHEMA,
    MULTI_SEED_SCHEMA,
    PHASE_SUMMARY_METRIC_NAMES,
    SIMULATION_SCHEMA,
)
from objectless_alife.simulation import _entropy_from_action_counts, run_batch_search  # noqa: F401

# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def _parse_phase(raw_phase: int) -> ObservationPhase:
    """Parse CLI phase value into ObservationPhase enum."""
    try:
        return ObservationPhase(raw_phase)
    except ValueError as exc:
        valid = ", ".join(str(p.value) for p in ObservationPhase)
        raise ValueError(f"phase must be one of {valid}") from exc


def _parse_phase_list(raw_phases: str) -> tuple[ObservationPhase, ...]:
    """Parse comma-delimited phase list and require exactly two entries."""
    parts = [part.strip() for part in raw_phases.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("phases must contain exactly two values")

    phases: list[ObservationPhase] = []
    for part in parts:
        try:
            phase_raw = int(part)
        except ValueError as exc:
            raise ValueError("phases must be integers 1, 2, 3, or 4") from exc
        phase = _parse_phase(phase_raw)
        phases.append(phase)

    if phases[0] == phases[1]:
        raise ValueError("phases must include two distinct values")

    return tuple(phases)


def _parse_grid_sizes(raw_grid_sizes: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-delimited grid sizes formatted as `WxH`."""
    parts = [part.strip() for part in raw_grid_sizes.split(",") if part.strip()]
    if not parts:
        raise ValueError("grid-sizes must not be empty")

    grid_sizes: list[tuple[int, int]] = []
    for part in parts:
        tokens = part.lower().split("x")
        if len(tokens) != 2:
            raise ValueError("grid-sizes entries must use WxH format")
        width_raw, height_raw = tokens
        try:
            width = int(width_raw)
            height = int(height_raw)
        except ValueError as exc:
            raise ValueError("grid-sizes entries must use integer WxH values") from exc
        if width < 1 or height < 1:
            raise ValueError("grid-sizes entries must be >= 1x1")
        grid_sizes.append((width, height))
    return tuple(grid_sizes)


def _parse_positive_int_csv(raw_values: str, label: str) -> tuple[int, ...]:
    """Parse comma-delimited positive integers."""
    parts = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} must not be empty")

    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"{label} must contain integers") from exc
        if value < 1:
            raise ValueError(f"{label} values must be >= 1")
        values.append(value)
    return tuple(values)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for search execution.

    Supports ``--config path/to/config.json`` for experiment reproducibility.
    CLI arguments override config-file values; config-file values override
    built-in defaults.
    """
    parser = argparse.ArgumentParser(description="Run objective-free ALife search")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file (CLI args override file values)",
    )
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--n-rules", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--halt-window", type=int, default=None)
    parser.add_argument("--rule-seed", type=int, default=None)
    parser.add_argument("--sim-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--density-sweep", action=argparse.BooleanOptionalAction, default=None)
    mode_group.add_argument("--experiment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grid-sizes", type=str, default=None)
    parser.add_argument("--agent-counts", type=str, default=None)
    parser.add_argument("--seed-batches", type=int, default=None)
    parser.add_argument("--phases", type=str, default=None)
    parser.add_argument(
        "--filter-short-period", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--short-period-max-period", type=int, default=None)
    parser.add_argument("--short-period-history-size", type=int, default=None)
    parser.add_argument(
        "--filter-low-activity", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--low-activity-window", type=int, default=None)
    parser.add_argument("--low-activity-min-unique-ratio", type=float, default=None)
    parser.add_argument("--block-ncd-window", type=int, default=None)
    parser.add_argument(
        "--fast-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip expensive null-model computations (shuffle_null_mi)",
    )
    args = parser.parse_args(argv)

    # Load config file defaults (CLI overrides file, file overrides built-in)
    file_cfg: dict[str, object] = {}
    if args.config is not None:
        file_cfg = json.loads(Path(args.config).read_text())

    def _get(cli_val: object, key: str, default: object) -> object:
        if cli_val is not None:
            return cli_val
        return file_cfg.get(key, default)

    def _get_bool(cli_val: bool | None, key: str, default: bool) -> bool:
        """CLI > file > default for boolean flags."""
        if cli_val is not None:
            return cli_val
        return bool(file_cfg.get(key, default))

    phase_raw = int(_get(args.phase, "phase", 1))
    n_rules = int(_get(args.n_rules, "n_rules", 100))
    steps = int(_get(args.steps, "steps", 200))
    halt_window = int(_get(args.halt_window, "halt_window", 10))
    rule_seed = int(_get(args.rule_seed, "rule_seed", 0))
    sim_seed = int(_get(args.sim_seed, "sim_seed", 0))
    out_dir = Path(str(_get(args.out_dir, "out_dir", "data")))
    grid_sizes_raw = str(_get(args.grid_sizes, "grid_sizes", "20x20"))
    agent_counts_raw = str(_get(args.agent_counts, "agent_counts", "30"))
    seed_batches = int(_get(args.seed_batches, "seed_batches", 1))
    phases_raw = str(_get(args.phases, "phases", "1,2"))
    filter_short_period = _get_bool(args.filter_short_period, "filter_short_period", False)
    short_period_max_period = int(_get(args.short_period_max_period, "short_period_max_period", 2))
    short_period_history_size = int(
        _get(args.short_period_history_size, "short_period_history_size", 8)
    )
    filter_low_activity = _get_bool(args.filter_low_activity, "filter_low_activity", False)
    low_activity_window = int(_get(args.low_activity_window, "low_activity_window", 5))
    low_activity_min_unique_ratio = float(
        _get(args.low_activity_min_unique_ratio, "low_activity_min_unique_ratio", 0.2)
    )
    block_ncd_window = int(_get(args.block_ncd_window, "block_ncd_window", 10))
    fast_metrics = _get_bool(args.fast_metrics, "fast_metrics", False)
    is_density_sweep = _get_bool(args.density_sweep, "density_sweep", False)
    is_experiment = _get_bool(args.experiment, "experiment", False)

    if is_density_sweep:
        density_sweep_config = DensitySweepConfig(
            grid_sizes=_parse_grid_sizes(grid_sizes_raw),
            agent_counts=_parse_positive_int_csv(agent_counts_raw, "agent-counts"),
            n_rules=n_rules,
            n_seed_batches=seed_batches,
            out_dir=out_dir,
            steps=steps,
            halt_window=halt_window,
            rule_seed_start=rule_seed,
            sim_seed_start=sim_seed,
            filter_short_period=filter_short_period,
            short_period_max_period=short_period_max_period,
            short_period_history_size=short_period_history_size,
            filter_low_activity=filter_low_activity,
            low_activity_window=low_activity_window,
            low_activity_min_unique_ratio=low_activity_min_unique_ratio,
            block_ncd_window=block_ncd_window,
        )
        results = run_density_sweep(density_sweep_config)
        summary = {
            "mode": "density_sweep",
            "phases": [1, 2],
            "density_points": len(density_sweep_config.grid_sizes)
            * len(density_sweep_config.agent_counts),
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    elif is_experiment:
        experiment_config = ExperimentConfig(
            phases=_parse_phase_list(phases_raw),
            n_rules=n_rules,
            n_seed_batches=seed_batches,
            out_dir=out_dir,
            steps=steps,
            halt_window=halt_window,
            rule_seed_start=rule_seed,
            sim_seed_start=sim_seed,
            filter_short_period=filter_short_period,
            short_period_max_period=short_period_max_period,
            short_period_history_size=short_period_history_size,
            filter_low_activity=filter_low_activity,
            low_activity_window=low_activity_window,
            low_activity_min_unique_ratio=low_activity_min_unique_ratio,
            block_ncd_window=block_ncd_window,
        )
        results = run_experiment(experiment_config)
        summary = {
            "experiment": True,
            "phases": [phase.value for phase in experiment_config.phases],
            "seed_batches": experiment_config.n_seed_batches,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    else:
        phase = _parse_phase(phase_raw)
        search_config = SearchConfig(
            steps=steps,
            halt_window=halt_window,
            filter_short_period=filter_short_period,
            short_period_max_period=short_period_max_period,
            short_period_history_size=short_period_history_size,
            filter_low_activity=filter_low_activity,
            low_activity_window=low_activity_window,
            low_activity_min_unique_ratio=low_activity_min_unique_ratio,
            block_ncd_window=block_ncd_window,
            skip_null_models=fast_metrics,
        )
        results = run_batch_search(
            n_rules=n_rules,
            phase=phase,
            out_dir=out_dir,
            base_rule_seed=rule_seed,
            base_sim_seed=sim_seed,
            config=search_config,
        )

        summary = {
            "experiment": False,
            "phase": phase.value,
            "total_rules": len(results),
            "survived": sum(1 for r in results if r.survived),
            "terminated": sum(1 for r in results if not r.survived),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
