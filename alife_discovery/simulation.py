"""Backward-compatibility shim: simulation module moved to simulation/ subpackage."""

from alife_discovery.simulation.engine import (
    run_batch_search as run_batch_search,
)
from alife_discovery.simulation.persistence import (
    flush_sim_columns as flush_sim_columns,
)
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
    compute_step_metrics as compute_step_metrics,
)
from alife_discovery.simulation.step import (
    entropy_from_action_counts as entropy_from_action_counts,
)
from alife_discovery.simulation.step import (
    mean_and_pvariance as mean_and_pvariance,
)
