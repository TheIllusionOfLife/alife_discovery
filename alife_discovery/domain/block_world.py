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

    @classmethod
    def create(cls, config: BlockWorldConfig, rng: Random) -> BlockWorld:
        """Initialize world with randomly placed blocks, no initial bonds."""
        w, h = config.grid_width, config.grid_height

        # Sample unique positions
        all_positions = [(x, y) for x in range(w) for y in range(h)]
        positions = rng.sample(all_positions, config.n_blocks)

        # Compute block type counts from fractions
        fractions = config.block_type_fractions
        raw_counts = [fractions[i] * config.n_blocks for i in range(3)]
        counts = [int(round(c)) for c in raw_counts]
        # Adjust rounding errors to match n_blocks exactly
        diff = config.n_blocks - sum(counts)
        if diff > 0:
            # Add to the largest fractional remainder
            remainders = [raw_counts[i] - int(raw_counts[i]) for i in range(3)]
            for _ in range(diff):
                idx = remainders.index(max(remainders))
                counts[idx] += 1
                remainders[idx] = -1.0  # don't pick again
        elif diff < 0:
            # Remove from the smallest fractional remainder
            remainders = [raw_counts[i] - int(raw_counts[i]) for i in range(3)]
            for _ in range(-diff):
                idx = remainders.index(min(remainders))
                counts[idx] -= 1
                remainders[idx] = 2.0  # don't pick again

        # Build type list
        type_list: list[BlockType] = []
        for i, bt in enumerate(("M", "C", "K")):
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
        )

    def neighbors_of(self, block_id: int) -> list[int]:
        """Return block_ids of blocks in Von Neumann neighborhood of block_id."""
        block = self.blocks[block_id]
        result: list[int] = []
        for nx_, ny_ in self._von_neumann_cells(block.x, block.y):
            if (nx_, ny_) in self.grid:
                neighbor_id = self.grid[(nx_, ny_)]
                if neighbor_id != block_id:
                    result.append(neighbor_id)
        return result

    def _von_neumann_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return 4 Von Neumann neighbor cells (toroidal)."""
        return [
            (x, (y - 1) % self.grid_height),
            (x, (y + 1) % self.grid_height),
            ((x - 1) % self.grid_width, y),
            ((x + 1) % self.grid_width, y),
        ]

    def step(self, rule_table: BlockRuleTable, noise_level: float, rng: Random) -> None:
        """Advance one simulation step (random sequential order)."""
        order = list(self.blocks.keys())
        rng.shuffle(order)
        for block_id in order:
            self._try_drift(block_id, rng)
            self._try_bond_form(block_id, rule_table, rng)
            self._try_bond_break(block_id, noise_level, rng)

    def _try_drift(self, block_id: int, rng: Random) -> None:
        """Attempt to move block to a random adjacent empty cell."""
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
        """Remove bonds where endpoints are no longer spatially adjacent."""
        block = self.blocks[block_id]
        adjacent_cells = set(self._von_neumann_cells(block.x, block.y))
        stale: set[frozenset[int]] = set()
        for bond in self.bonds:
            if block_id not in bond:
                continue
            other_id = next(b for b in bond if b != block_id)
            other = self.blocks[other_id]
            if (other.x, other.y) not in adjacent_cells:
                stale.add(bond)
        self.bonds -= stale

    def _try_bond_form(self, block_id: int, rule_table: BlockRuleTable, rng: Random) -> None:
        """Form bonds with adjacent unbound blocks based on rule table."""
        block = self.blocks[block_id]
        neighbor_ids = self.neighbors_of(block_id)
        if not neighbor_ids:
            return
        # Compute observation features
        neighbor_count = len(neighbor_ids)
        neighbor_types = [self.blocks[n].block_type for n in neighbor_ids]
        type_counts = Counter(neighbor_types)
        dominant_type = type_counts.most_common(1)[0][0]
        prob = rule_table.get((block.block_type, neighbor_count, dominant_type), 0.0)
        # Try forming bond with each unbound adjacent block
        for neighbor_id in neighbor_ids:
            bond = frozenset({block_id, neighbor_id})
            if bond not in self.bonds:
                if rng.random() < prob:
                    self.bonds.add(bond)

    def _try_bond_break(self, block_id: int, noise_level: float, rng: Random) -> None:
        """Break existing bonds of this block with probability noise_level."""
        to_break: set[frozenset[int]] = set()
        for bond in self.bonds:
            if block_id in bond:
                if rng.random() < noise_level:
                    to_break.add(bond)
        self.bonds -= to_break


def generate_block_rule_table(rule_seed: int) -> BlockRuleTable:
    """Generate a random bond-formation rule table."""
    rng = Random(rule_seed)
    table: BlockRuleTable = {}
    dominant_types = list(BLOCK_TYPES) + ["Empty"]
    for self_type in BLOCK_TYPES:
        for neighbor_count in range(5):  # 0-4
            for dominant_type in dominant_types:
                table[(self_type, neighbor_count, dominant_type)] = rng.random()
    return table
