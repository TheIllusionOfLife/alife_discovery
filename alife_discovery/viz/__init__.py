"""Visualization layer: themes, renderers, CLI, and web export."""

from alife_discovery.viz.cli import main
from alife_discovery.viz.export_web import (
    export_batch,
    export_gallery,
    export_paired,
    export_single,
)
from alife_discovery.viz.render import (
    render_batch,
    render_filmstrip,
    render_metric_distribution,
    render_metric_timeseries,
    render_rule_animation,
    render_snapshot_grid,
    select_top_rules,
)
from alife_discovery.viz.theme import (
    DEFAULT_THEME,
    PAPER_THEME,
    REGISTERED_THEMES,
    Theme,
    get_theme,
)

__all__ = [
    "DEFAULT_THEME",
    "PAPER_THEME",
    "REGISTERED_THEMES",
    "Theme",
    "export_batch",
    "export_gallery",
    "export_paired",
    "export_single",
    "get_theme",
    "main",
    "render_batch",
    "render_filmstrip",
    "render_metric_distribution",
    "render_metric_timeseries",
    "render_rule_animation",
    "render_snapshot_grid",
    "select_top_rules",
]
