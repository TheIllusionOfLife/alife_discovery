"""Backward-compatibility shim: viz_cli module moved to viz/cli.py."""

from alife_discovery.viz.cli import (
    _build_batch_parser as _build_batch_parser,
)
from alife_discovery.viz.cli import (
    _build_figure_parser as _build_figure_parser,
)
from alife_discovery.viz.cli import (
    _build_filmstrip_parser as _build_filmstrip_parser,
)
from alife_discovery.viz.cli import (
    _build_single_parser as _build_single_parser,
)
from alife_discovery.viz.cli import (
    _handle_batch as _handle_batch,
)
from alife_discovery.viz.cli import (
    _handle_figure as _handle_figure,
)
from alife_discovery.viz.cli import (
    _handle_filmstrip as _handle_filmstrip,
)
from alife_discovery.viz.cli import (
    _handle_single as _handle_single,
)
from alife_discovery.viz.cli import (
    _parse_phase_dirs as _parse_phase_dirs,
)
from alife_discovery.viz.cli import (
    main as main,
)
