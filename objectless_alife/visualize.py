"""Compatibility shim for visualization APIs and CLI.

Rendering functionality now lives in ``objectless_alife.viz_render`` and
CLI parsing/dispatch in ``objectless_alife.viz_cli``.
"""

from __future__ import annotations

from objectless_alife import viz_render as _viz_render
from objectless_alife.viz_cli import main
from objectless_alife.viz_render import *  # noqa: F401,F403

# Preserve access to helper functions that intentionally start with underscore
# and are imported directly by tests and downstream tooling.
_build_grid_array = _viz_render._build_grid_array
_state_cmap = _viz_render._state_cmap

if __name__ == "__main__":
    main()
