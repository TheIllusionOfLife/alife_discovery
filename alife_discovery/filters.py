"""Backward-compatibility shim: filters module moved to domain/filters.py."""

from alife_discovery.config.constants import ACTION_SPACE_SIZE as ACTION_SPACE_SIZE
from alife_discovery.domain.filters import (
    HaltDetector as HaltDetector,
)
from alife_discovery.domain.filters import (
    LowActivityDetector as LowActivityDetector,
)
from alife_discovery.domain.filters import (
    ShortPeriodDetector as ShortPeriodDetector,
)
from alife_discovery.domain.filters import (
    StateUniformDetector as StateUniformDetector,
)
from alife_discovery.domain.filters import (
    TerminationReason as TerminationReason,
)
