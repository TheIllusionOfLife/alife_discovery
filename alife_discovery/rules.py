"""Backward-compatibility shim: rules module moved to domain/rules.py."""

from alife_discovery.config.constants import CLOCK_PERIOD as CLOCK_PERIOD
from alife_discovery.domain.rules import (
    ObservationPhase as ObservationPhase,
)
from alife_discovery.domain.rules import (
    compute_capacity_matched_index as compute_capacity_matched_index,
)
from alife_discovery.domain.rules import (
    compute_control_index as compute_control_index,
)
from alife_discovery.domain.rules import (
    compute_phase1_index as compute_phase1_index,
)
from alife_discovery.domain.rules import (
    compute_phase2_index as compute_phase2_index,
)
from alife_discovery.domain.rules import (
    dominant_neighbor_state as dominant_neighbor_state,
)
from alife_discovery.domain.rules import (
    generate_rule_table as generate_rule_table,
)
from alife_discovery.domain.rules import (
    rule_table_size as rule_table_size,
)
