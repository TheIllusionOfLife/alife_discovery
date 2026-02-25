"""Backward-compatibility shim: world module moved to domain/world.py."""

from alife_discovery.config.constants import CLOCK_PERIOD as CLOCK_PERIOD
from alife_discovery.domain.world import (
    NOOP_ACTION as NOOP_ACTION,
)
from alife_discovery.domain.world import (
    NUM_ACTIONS as NUM_ACTIONS,
)
from alife_discovery.domain.world import (
    Agent as Agent,
)
from alife_discovery.domain.world import (
    World as World,
)
from alife_discovery.domain.world import (
    WorldConfig as WorldConfig,
)
