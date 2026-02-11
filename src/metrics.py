from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Sequence


def state_entropy(states: Sequence[int]) -> float:
    """Compute Shannon entropy (base 2) for discrete agent states."""
    if not states:
        return 0.0

    counts = Counter(states)
    n = len(states)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def compression_ratio(payload: bytes) -> float:
    """Return zlib compressed_size / original_size for payload bytes."""
    if not payload:
        return 0.0
    compressed = zlib.compress(payload)
    return len(compressed) / len(payload)


def normalized_hamming_distance(before: Sequence[int], after: Sequence[int]) -> float:
    """Return normalized Hamming distance in [0, 1] between equal-length vectors."""
    if len(before) != len(after):
        raise ValueError("Vectors must have equal length")
    if not before:
        return 0.0
    distance = sum(1 for b, a in zip(before, after, strict=True) if b != a)
    return distance / len(before)


def serialize_snapshot(
    snapshot: tuple[tuple[int, int, int, int], ...], grid_width: int, grid_height: int
) -> bytes:
    """Serialize snapshot into a flat grid byte buffer with 255 for empty cells."""
    data = bytearray([255] * (grid_width * grid_height))
    for _, x, y, state in snapshot:
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            raise ValueError("Snapshot contains out-of-bounds coordinates")
        idx = y * grid_width + x
        data[idx] = state
    return bytes(data)
