"""Spatial hashing for O(1) neighbor lookups in 2D and 3D.

Divides space into uniform grid cells and hashes each object into the
cell it occupies. Neighbor lookups only require inspecting the 4 (2D)
or 8 (3D) cells around a query point, giving O(1) lookup amortized
over many queries.

This is a port of Matthias Muller's O(1) spatial hashing system (used in
zalo/MathUtilities) and is useful for:
- Player-player collision detection in pressure metrics
- Finding nearest neighbors for clustering
- Ball-player proximity queries
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

import numpy as np


class SpatialHash2D:
    """Uniform-grid spatial hash for 2D point queries.

    Example:
        >>> sh = SpatialHash2D(cell_size=10.0)
        >>> sh.insert("player_1", (5.0, 3.0))
        >>> sh.insert("player_2", (7.0, 2.5))
        >>> sh.query_neighbors(5.5, 3.0, radius=2.0)
        ['player_1', 'player_2']
    """

    def __init__(self, cell_size: float = 10.0) -> None:
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list[Hashable]] = defaultdict(list)

    def _cell_coords(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.cell_size)), int(np.floor(y / self.cell_size)))

    def insert(self, obj: Hashable, position: Sequence[float]) -> None:
        """Insert an object at the given (x, y) position."""
        if len(position) < 2:
            raise ValueError(f"position must have at least 2 elements, got {len(position)}")
        cx, cy = self._cell_coords(float(position[0]), float(position[1]))
        self.grid[(cx, cy)].append(obj)

    def query_cell(self, x: float, y: float) -> list[Hashable]:
        """Return all objects in the cell containing (x, y)."""
        return self.grid.get(self._cell_coords(x, y), [])

    def query_neighbors(self, x: float, y: float, radius: float) -> list[Hashable]:
        """Return all objects within `radius` of (x, y).

        Inspects the 3x3 neighborhood of cells around the query point,
        giving O(1) amortized lookup.
        """
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        cx, cy = self._cell_coords(x, y)
        cell_extent = int(np.ceil(radius / self.cell_size))
        seen: set[Hashable] = set()
        result: list[Hashable] = []
        for dx in range(-cell_extent, cell_extent + 1):
            for dy in range(-cell_extent, cell_extent + 1):
                for obj in self.grid.get((cx + dx, cy + dy), []):
                    if obj not in seen:
                        seen.add(obj)
                        result.append(obj)
        return result

    def clear(self) -> None:
        self.grid.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self.grid.values())


class SpatialHash3D:
    """Uniform-grid spatial hash for 3D point queries.

    Same interface as SpatialHash2D, but operates on (x, y, z) and
    inspects a 3x3x3 neighborhood (27 cells).
    """

    def __init__(self, cell_size: float = 10.0) -> None:
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int, int], list[Hashable]] = defaultdict(list)

    def _cell_coords(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(np.floor(x / self.cell_size)),
            int(np.floor(y / self.cell_size)),
            int(np.floor(z / self.cell_size)),
        )

    def insert(self, obj: Hashable, position: Sequence[float]) -> None:
        if len(position) < 3:
            raise ValueError(f"position must have 3 elements, got {len(position)}")
        cx, cy, cz = self._cell_coords(float(position[0]), float(position[1]), float(position[2]))
        self.grid[(cx, cy, cz)].append(obj)

    def query_cell(self, x: float, y: float, z: float) -> list[Hashable]:
        return self.grid.get(self._cell_coords(x, y, z), [])

    def query_neighbors(self, x: float, y: float, z: float, radius: float) -> list[Hashable]:
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        cx, cy, cz = self._cell_coords(x, y, z)
        cell_extent = int(np.ceil(radius / self.cell_size))
        seen: set[Hashable] = set()
        result: list[Hashable] = []
        for dx in range(-cell_extent, cell_extent + 1):
            for dy in range(-cell_extent, cell_extent + 1):
                for dz in range(-cell_extent, cell_extent + 1):
                    for obj in self.grid.get((cx + dx, cy + dy, cz + dz), []):
                        if obj not in seen:
                            seen.add(obj)
                            result.append(obj)
        return result

    def clear(self) -> None:
        self.grid.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self.grid.values())


def bulk_insert_2d(
    sh: SpatialHash2D, positions: Iterable[Sequence[float]], objs: Iterable[Hashable] | None = None
) -> list[Hashable]:
    """Insert many points at once. Returns list of inserted objects."""
    inserted: list[Hashable] = []
    if objs is None:
        positions_list = list(positions)
        objs_iter: Iterable[int] | list[Hashable] = range(len(positions_list))
    else:
        positions_list = list(positions)
        objs_iter = list(objs)
        if len(objs_iter) != len(positions_list):
            raise ValueError("positions and objs must have the same length")
    for obj, pos in zip(objs_iter, positions_list, strict=False):
        sh.insert(obj, pos)
        inserted.append(obj)
    return inserted
