"""Configuration dataclasses and safety constants for simulation experiments.

All frozen dataclasses that parameterise search, experiment, density-sweep,
multi-seed, and halt-window-sweep runs live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from objectless_alife.rules import ObservationPhase

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

MAX_EXPERIMENT_WORK_UNITS = 100_000_000

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Top-level result for one evaluated rule table."""

    rule_id: str
    survived: bool
    terminated_at: int | None
    termination_reason: str | None


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """Batch-search runtime parameters including optional dynamic filters."""

    steps: int = 200
    halt_window: int = 10
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    shuffle_null_n_shuffles: int = 200
    skip_null_models: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-scale runtime settings for multi-phase, multi-seed evaluation."""

    phases: tuple[ObservationPhase, ...] = (
        ObservationPhase.PHASE1_DENSITY,
        ObservationPhase.PHASE2_PROFILE,
    )
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    skip_null_models: bool = False


@dataclass(frozen=True)
class DensitySweepConfig:
    """Runtime settings for grid/agent density sweeps across both phases."""

    grid_sizes: tuple[tuple[int, int], ...] = ((20, 20),)
    agent_counts: tuple[int, ...] = (30,)
    n_rules: int = 100
    n_seed_batches: int = 1
    out_dir: Path = Path("data")
    steps: int = 200
    halt_window: int = 10
    rule_seed_start: int = 0
    sim_seed_start: int = 0
    filter_short_period: bool = False
    short_period_max_period: int = 2
    short_period_history_size: int = 8
    filter_low_activity: bool = False
    low_activity_window: int = 5
    low_activity_min_unique_ratio: float = 0.2
    block_ncd_window: int = 10
    skip_null_models: bool = False


@dataclass(frozen=True)
class MultiSeedConfig:
    """Settings for multi-seed robustness evaluation of selected rules."""

    rule_seeds: tuple[int, ...]
    n_sim_seeds: int = 20
    out_dir: Path = Path("data/multi_seed")
    steps: int = 200
    halt_window: int = 10
    phase: ObservationPhase = ObservationPhase.PHASE2_PROFILE
    shuffle_null_n_shuffles: int = 200


@dataclass(frozen=True)
class HaltWindowSweepConfig:
    """Settings for halt-window sensitivity analysis."""

    rule_seeds: tuple[int, ...]
    halt_windows: tuple[int, ...] = (5, 10, 20)
    out_dir: Path = Path("data/halt_window_sweep")
    steps: int = 200
    phase: ObservationPhase = ObservationPhase.PHASE2_PROFILE
    shuffle_null_n_shuffles: int = 200

    def __post_init__(self) -> None:
        if not self.rule_seeds:
            raise ValueError("rule_seeds must not be empty")
        if not self.halt_windows:
            raise ValueError("halt_windows must not be empty")
        if any(w < 1 for w in self.halt_windows):
            raise ValueError("halt_windows values must be >= 1")
