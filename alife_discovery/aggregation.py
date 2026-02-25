"""Backward-compatibility shim: aggregation module moved to experiments/ subpackage."""

from alife_discovery.experiments.density_sweep import run_density_sweep as run_density_sweep
from alife_discovery.experiments.experiment import run_experiment as run_experiment
from alife_discovery.experiments.robustness import (
    run_halt_window_sweep as run_halt_window_sweep,
)
from alife_discovery.experiments.robustness import (
    run_multi_seed_robustness as run_multi_seed_robustness,
)
from alife_discovery.experiments.selection import (
    select_top_rules_by_delta_mi as select_top_rules_by_delta_mi,
)
from alife_discovery.experiments.summaries import (
    collect_final_metric_rows as collect_final_metric_rows,
)

# Private alias for backward compatibility with scripts using the old name
_collect_final_metric_rows = collect_final_metric_rows
