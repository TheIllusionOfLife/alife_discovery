"""Backward-compatibility shim: CLI entrypoint moved to experiments/search.py."""

from alife_discovery.experiments.search import (
    _coerce_bool as _coerce_bool,
)
from alife_discovery.experiments.search import (
    _coerce_float as _coerce_float,
)
from alife_discovery.experiments.search import (
    _coerce_int as _coerce_int,
)
from alife_discovery.experiments.search import (
    _coerce_str as _coerce_str,
)
from alife_discovery.experiments.search import (
    _parse_grid_sizes as _parse_grid_sizes,
)
from alife_discovery.experiments.search import (
    _parse_phase as _parse_phase,
)
from alife_discovery.experiments.search import (
    _parse_phase_list as _parse_phase_list,
)
from alife_discovery.experiments.search import (
    _parse_positive_int_csv as _parse_positive_int_csv,
)
from alife_discovery.experiments.search import (
    _parse_state_uniform_mode as _parse_state_uniform_mode,
)
from alife_discovery.experiments.search import (
    _parse_update_mode as _parse_update_mode,
)
from alife_discovery.experiments.search import (
    main as main,
)

if __name__ == "__main__":
    main()
