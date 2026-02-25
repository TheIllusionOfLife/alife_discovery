"""Compatibility shim for visualization APIs and CLI.

Rendering functionality now lives in ``alife_discovery.viz.render`` and
CLI parsing/dispatch in ``alife_discovery.viz.cli``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: F401 — re-exported for tests that patch via this module
from matplotlib import animation  # noqa: F401 — re-exported for tests that patch via this module

from alife_discovery.viz.cli import main as main
from alife_discovery.viz.render import (
    METRIC_LABELS as METRIC_LABELS,
)
from alife_discovery.viz.render import (
    PHASE_DESCRIPTIONS as PHASE_DESCRIPTIONS,
)
from alife_discovery.viz.render import (
    _build_grid_array as _build_grid_array,
)
from alife_discovery.viz.render import (
    _state_cmap as _state_cmap,
)
from alife_discovery.viz.render import (
    render_batch as render_batch,
)
from alife_discovery.viz.render import (
    render_filmstrip as render_filmstrip,
)
from alife_discovery.viz.render import (
    render_metric_distribution as render_metric_distribution,
)
from alife_discovery.viz.render import (
    render_metric_timeseries as render_metric_timeseries,
)
from alife_discovery.viz.render import (
    render_rule_animation as render_rule_animation,
)
from alife_discovery.viz.render import (
    render_snapshot_grid as render_snapshot_grid,
)
from alife_discovery.viz.render import (
    select_top_rules as select_top_rules,
)

if __name__ == "__main__":
    main()
