"""Simulation engine: batch search, per-step metrics, and Parquet persistence."""

from alife_discovery.simulation.engine import run_batch_search
from alife_discovery.simulation.persistence import flush_sim_columns
from alife_discovery.simulation.step import (
    _compute_step_metrics as _compute_step_metrics,
)
from alife_discovery.simulation.step import (
    _entropy_from_action_counts as _entropy_from_action_counts,
)
from alife_discovery.simulation.step import (
    _mean_and_pvariance as _mean_and_pvariance,
)
from alife_discovery.simulation.step import (
    compute_step_metrics,
    entropy_from_action_counts,
    mean_and_pvariance,
)

__all__ = [
    "compute_step_metrics",
    "entropy_from_action_counts",
    "flush_sim_columns",
    "mean_and_pvariance",
    "run_batch_search",
]
