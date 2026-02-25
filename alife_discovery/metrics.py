"""Backward-compatibility shim: metrics module moved to metrics/ subpackage."""

from alife_discovery.metrics.information import (
    block_ncd as block_ncd,
)
from alife_discovery.metrics.information import (
    block_shuffle_null_mi as block_shuffle_null_mi,
)
from alife_discovery.metrics.information import (
    compression_ratio as compression_ratio,
)
from alife_discovery.metrics.information import (
    fixed_marginal_null_mi as fixed_marginal_null_mi,
)
from alife_discovery.metrics.information import (
    neighbor_mutual_information as neighbor_mutual_information,
)
from alife_discovery.metrics.information import (
    neighbor_transfer_entropy as neighbor_transfer_entropy,
)
from alife_discovery.metrics.information import (
    normalized_hamming_distance as normalized_hamming_distance,
)
from alife_discovery.metrics.information import (
    serialize_snapshot as serialize_snapshot,
)
from alife_discovery.metrics.information import (
    shuffle_null_mi as shuffle_null_mi,
)
from alife_discovery.metrics.information import (
    spatial_scramble_mi as spatial_scramble_mi,
)
from alife_discovery.metrics.information import (
    state_entropy as state_entropy,
)
from alife_discovery.metrics.information import (
    transfer_entropy_excess as transfer_entropy_excess,
)
from alife_discovery.metrics.information import (
    transfer_entropy_shuffle_null as transfer_entropy_shuffle_null,
)
from alife_discovery.metrics.spatial import (
    cluster_count_by_state as cluster_count_by_state,
)
from alife_discovery.metrics.spatial import (
    morans_i_occupied as morans_i_occupied,
)
from alife_discovery.metrics.spatial import (
    neighbor_pair_count as neighbor_pair_count,
)
from alife_discovery.metrics.spatial import (
    same_state_adjacency_fraction as same_state_adjacency_fraction,
)
from alife_discovery.metrics.temporal import (
    action_entropy as action_entropy,
)
from alife_discovery.metrics.temporal import (
    action_entropy_variance as action_entropy_variance,
)
from alife_discovery.metrics.temporal import (
    phase_transition_max_delta as phase_transition_max_delta,
)
from alife_discovery.metrics.temporal import (
    quasi_periodicity_peak_count as quasi_periodicity_peak_count,
)
