"""Backward-compatibility shim: viz_render module moved to viz/render.py."""

from alife_discovery.viz.render import (
    EMPTY_CELL_COLOR as EMPTY_CELL_COLOR,
)
from alife_discovery.viz.render import (
    EMPTY_CELL_COLOR_DARK as EMPTY_CELL_COLOR_DARK,
)
from alife_discovery.viz.render import (
    GRID_LINE_COLOR as GRID_LINE_COLOR,
)
from alife_discovery.viz.render import (
    GRID_LINE_COLOR_DARK as GRID_LINE_COLOR_DARK,
)
from alife_discovery.viz.render import (
    METRIC_COLORS as METRIC_COLORS,
)
from alife_discovery.viz.render import (
    METRIC_LABELS as METRIC_LABELS,
)
from alife_discovery.viz.render import (
    PHASE_COLORS as PHASE_COLORS,
)
from alife_discovery.viz.render import (
    PHASE_DESCRIPTIONS as PHASE_DESCRIPTIONS,
)
from alife_discovery.viz.render import (
    STATE_COLORS as STATE_COLORS,
)
from alife_discovery.viz.render import (
    _annotate_significance as _annotate_significance,
)
from alife_discovery.viz.render import (
    _build_grid_array as _build_grid_array,
)
from alife_discovery.viz.render import (
    _build_state_legend_handles as _build_state_legend_handles,
)
from alife_discovery.viz.render import (
    _draw_cell_grid as _draw_cell_grid,
)
from alife_discovery.viz.render import (
    _resolve_grid_dimension as _resolve_grid_dimension,
)
from alife_discovery.viz.render import (
    _resolve_within_base as _resolve_within_base,
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
