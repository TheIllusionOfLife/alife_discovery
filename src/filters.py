from __future__ import annotations

from enum import Enum
from typing import Sequence


class TerminationReason(str, Enum):
    """Termination reason labels persisted in run metadata."""

    HALT = "halt"
    STATE_UNIFORM = "state_uniform"


class HaltDetector:
    """Detect N consecutive unchanged snapshots."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._last_snapshot: tuple[tuple[int, int, int, int], ...] | None = None
        self._unchanged_count = 0

    def observe(self, snapshot: tuple[tuple[int, int, int, int], ...]) -> bool:
        """Return True once snapshot has remained unchanged for `window` checks."""
        if self._last_snapshot is None:
            self._last_snapshot = snapshot
            return False

        if snapshot == self._last_snapshot:
            self._unchanged_count += 1
        else:
            self._unchanged_count = 0
            self._last_snapshot = snapshot

        return self._unchanged_count >= self.window


class StateUniformDetector:
    """Detect whether all agents currently share the same internal state."""

    def observe(self, states: Sequence[int]) -> bool:
        """Return True only when all states are equal and input is non-empty."""
        if not states:
            return False
        return len(set(states)) == 1
