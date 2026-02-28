"""Block-typed toroidal grid world with bond dynamics.

Bond-motion invariant: bonds break automatically when endpoints become
non-adjacent due to drift. Bonded clusters never move as rigid units.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Literal

from alife_discovery.config.constants import BLOCK_TYPES

if TYPE_CHECKING:
    from alife_discovery.config.types import BlockWorldConfig

BlockType = Literal["M", "C", "K"]

# Rule table type: (self_type, neighbor_count, dominant_type) -> bond_probability
BlockRuleTable = dict[tuple[str, int, str], float]

# Maximum neighbor_count value used as rule-table key (caps observed neighbors for any radius)
MAX_NEIGHBOR_COUNT = 4


@dataclass
class Block:
    """A single typed block in the world."""

    block_id: int
    x: int
    y: int
    block_type: BlockType


@dataclass
class BlockWorld:
    """Toroidal grid world with typed blocks and bond graph."""

    grid_width: int
    grid_height: int
    grid: dict[tuple[int, int], int]  # (x, y) -> block_id
    blocks: dict[int, Block]  # block_id -> Block
    bonds: set[frozenset[int]]  # each element is frozenset of 2 block IDs
    observation_range: int = 1  # Manhattan radius for bond formation/pruning
    catalyst_multiplier: float = 1.0  # bond prob multiplier when K neighbor present
    drift_probability: float = 1.0  # probability of attempting drift each step

    @classmethod
    def create(cls, config: BlockWorldConfig, rng: Random) -> BlockWorld:
        """Initialize world with randomly placed blocks, no initial bonds."""
        w, h = config.grid_width, config.grid_height

        # Sample unique positions
        all_positions = [(x, y) for x in range(w) for y in range(h)]
        positions = rng.sample(all_positions, config.n_blocks)

        # Compute block type counts from fractions
        n_types = len(BLOCK_TYPES)
        fractions = config.block_type_fractions
        raw_counts = [fractions[i] * config.n_blocks for i in range(n_types)]
        counts = [int(round(c)) for c in raw_counts]
        # Adjust rounding errors to match n_blocks exactly
        diff = config.n_blocks - sum(counts)
        if diff > 0:
            # Add to the largest fractional remainder
            remainders = [raw_counts[i] - int(raw_counts[i]) for i in range(n_types)]
            for _ in range(diff):
                idx = remainders.index(max(remainders))
                counts[idx] += 1
                remainders[idx] = -1.0  # don't pick again
        elif diff < 0:
            # Remove from the smallest fractional remainder
            remainders = [raw_counts[i] - int(raw_counts[i]) for i in range(n_types)]
            for _ in range(-diff):
                idx = remainders.index(min(remainders))
                counts[idx] -= 1
                remainders[idx] = 2.0  # don't pick again

        # Build type list
        type_list: list[BlockType] = []
        for i, bt in enumerate(BLOCK_TYPES):
            type_list.extend([bt] * counts[i])  # type: ignore[list-item]
        rng.shuffle(type_list)

        grid: dict[tuple[int, int], int] = {}
        blocks: dict[int, Block] = {}
        for block_id, (pos, bt) in enumerate(zip(positions, type_list, strict=True)):
            block = Block(block_id=block_id, x=pos[0], y=pos[1], block_type=bt)
            blocks[block_id] = block
            grid[pos] = block_id

        return cls(
            grid_width=w,
            grid_height=h,
            grid=grid,
            blocks=blocks,
            bonds=set(),
            observation_range=config.observation_range,
            catalyst_multiplier=config.catalyst_multiplier,
            drift_probability=config.drift_probability,
        )

    def neighbors_of(self, block_id: int, radius: int = 1) -> list[int]:
        """Return block_ids of blocks within Manhattan distance `radius` of block_id."""
        block = self.blocks[block_id]
        result: list[int] = []
        for nx_, ny_ in self._manhattan_neighbors(block.x, block.y, radius):
            if (nx_, ny_) in self.grid:
                neighbor_id = self.grid[(nx_, ny_)]
                if neighbor_id != block_id:
                    result.append(neighbor_id)
        return result

    def _toroidal_manhattan(self, a: Block, b: Block) -> int:
        """Toroidal Manhattan distance between two blocks."""
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        return min(dx, self.grid_width - dx) + min(dy, self.grid_height - dy)

    def _manhattan_neighbors(self, x: int, y: int, radius: int) -> list[tuple[int, int]]:
        """Return all cells within Manhattan distance `radius` (toroidal, excluding origin).

        Iterates only the diamond (not the enclosing square) for efficiency.
        Deduplicates wrapped cells, which can collide when radius > grid_dim/2.
        """
        seen: set[tuple[int, int]] = set()
        cells: list[tuple[int, int]] = []
        for dx in range(-radius, radius + 1):
            y_radius = radius - abs(dx)
            for dy in range(-y_radius, y_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                cell = ((x + dx) % self.grid_width, (y + dy) % self.grid_height)
                if cell not in seen:
                    seen.add(cell)
                    cells.append(cell)
        return cells

    def _von_neumann_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return radius=1 Manhattan neighbors (toroidal). Drift always uses radius=1."""
        return self._manhattan_neighbors(x, y, 1)

    def step(
        self,
        rule_table: BlockRuleTable,
        noise_level: float,
        rng: Random,
        update_mode: object = None,
    ) -> None:
        """Advance one simulation step.

        Sequential mode (default): process one block at a time in shuffled order.
        Synchronous mode: compute all drift targets from frozen positions, apply
        simultaneously, then compute bonds from new positions.
        """
        from alife_discovery.config.types import UpdateMode

        mode = update_mode if update_mode is not None else UpdateMode.SEQUENTIAL
        if mode == UpdateMode.SYNCHRONOUS:
            self._step_synchronous(rule_table, noise_level, rng)
        else:
            self._step_sequential(rule_table, noise_level, rng)

    def _step_sequential(self, rule_table: BlockRuleTable, noise_level: float, rng: Random) -> None:
        """Random-sequential update: one block at a time."""
        order = list(self.blocks.keys())
        rng.shuffle(order)
        for block_id in order:
            self._try_drift(block_id, rng)
            self._try_bond_form(block_id, rule_table, rng)
        self._step_bond_break(noise_level, rng)

    def _step_synchronous(
        self, rule_table: BlockRuleTable, noise_level: float, rng: Random
    ) -> None:
        """Synchronous update: drift targets computed from frozen positions, then applied."""
        order = list(self.blocks.keys())
        rng.shuffle(order)
        # Phase 1: compute drift targets from frozen occupancy snapshot
        frozen_grid = dict(self.grid)
        new_positions: dict[int, tuple[int, int]] = {}
        claimed: dict[tuple[int, int], int] = {}  # target -> first claimant
        for block_id in order:
            block = self.blocks[block_id]
            cells = self._von_neumann_cells(block.x, block.y)
            rng.shuffle(cells)
            moved = False
            for nx_, ny_ in cells:
                if (nx_, ny_) not in frozen_grid and (nx_, ny_) not in claimed:
                    new_positions[block_id] = (nx_, ny_)
                    claimed[(nx_, ny_)] = block_id
                    moved = True
                    break
            if not moved:
                new_positions[block_id] = (block.x, block.y)
        # Phase 2: apply moves
        self.grid.clear()
        for block_id, (nx_, ny_) in new_positions.items():
            block = self.blocks[block_id]
            block.x = nx_
            block.y = ny_
            self.grid[(nx_, ny_)] = block_id
        # Prune bonds where endpoints now exceed observation_range (toroidal distance)
        stale: set[frozenset[int]] = set()
        for bond in self.bonds:
            endpoints = list(bond)
            a, b = self.blocks[endpoints[0]], self.blocks[endpoints[1]]
            if self._toroidal_manhattan(a, b) > self.observation_range:
                stale.add(bond)
        self.bonds -= stale
        # Phase 3: bond formation then breaking (from new positions)
        rng.shuffle(order)
        for block_id in order:
            self._try_bond_form(block_id, rule_table, rng)
        self._step_bond_break(noise_level, rng)

    def _try_drift(self, block_id: int, rng: Random) -> None:
        """Attempt to move block to a random adjacent empty cell."""
        if self.drift_probability < 1.0 and rng.random() >= self.drift_probability:
            return
        block = self.blocks[block_id]
        cells = self._von_neumann_cells(block.x, block.y)
        rng.shuffle(cells)
        for nx_, ny_ in cells:
            if (nx_, ny_) not in self.grid:
                # Move block
                del self.grid[(block.x, block.y)]
                block.x = nx_
                block.y = ny_
                self.grid[(nx_, ny_)] = block_id
                # Break bonds that are now non-adjacent
                self._prune_broken_bonds(block_id)
                return

    def _prune_broken_bonds(self, block_id: int) -> None:
        """Remove bonds where endpoints exceed observation_range after drift."""
        block = self.blocks[block_id]
        reachable = set(self._manhattan_neighbors(block.x, block.y, self.observation_range))
        stale: set[frozenset[int]] = set()
        for bond in self.bonds:
            if block_id not in bond:
                continue
            other_id = next(b for b in bond if b != block_id)
            other = self.blocks[other_id]
            if (other.x, other.y) not in reachable:
                stale.add(bond)
        self.bonds -= stale

    def _try_bond_form(self, block_id: int, rule_table: BlockRuleTable, rng: Random) -> None:
        """Form bonds with blocks within observation_range based on rule table."""
        block = self.blocks[block_id]
        neighbor_ids = self.neighbors_of(block_id, radius=self.observation_range)
        if not neighbor_ids:
            return
        # Cap neighbor_count to keep rule table at 60 entries for any radius
        neighbor_count = min(len(neighbor_ids), MAX_NEIGHBOR_COUNT)
        neighbor_types = [self.blocks[n].block_type for n in neighbor_ids]
        type_counts = Counter(neighbor_types)
        dominant_type = type_counts.most_common(1)[0][0]
        prob = rule_table.get((block.block_type, neighbor_count, dominant_type), 0.0)
        if self.catalyst_multiplier > 1.0:
            has_k_neighbor = any(self.blocks[n].block_type == "K" for n in neighbor_ids)
            if has_k_neighbor:
                prob = min(prob * self.catalyst_multiplier, 1.0)
        for neighbor_id in neighbor_ids:
            bond = frozenset({block_id, neighbor_id})
            if bond not in self.bonds:
                if rng.random() < prob:
                    self.bonds.add(bond)

    def _step_bond_break(self, noise_level: float, rng: Random) -> None:
        """Break each bond once per step with probability noise_level.

        Called once per step (not per block) to prevent the double-break bug where
        processing both endpoints of a bond would give effective rate 2p - p² instead of p.
        """
        to_break: set[frozenset[int]] = set()
        for bond in self.bonds:
            if rng.random() < noise_level:
                to_break.add(bond)
        self.bonds -= to_break


def generate_block_rule_table(rule_seed: int) -> BlockRuleTable:
    """Generate a random bond-formation rule table."""
    rng = Random(rule_seed)
    table: BlockRuleTable = {}
    dominant_types = list(BLOCK_TYPES) + ["Empty"]
    for self_type in BLOCK_TYPES:
        for neighbor_count in range(MAX_NEIGHBOR_COUNT + 1):  # 0..MAX_NEIGHBOR_COUNT
            for dominant_type in dominant_types:
                table[(self_type, neighbor_count, dominant_type)] = rng.random()
    return table
