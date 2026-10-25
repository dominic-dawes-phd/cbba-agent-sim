"""Shortest-path distances on the static grid (hard rule 8).

`PathPlanner.steps` is the only distance function in the codebase. The oracle, numeric CBBA, and
the text shown to LLMs all use it, so no condition gets better distance information.
"""

import math
from collections import deque

from sim.world import Cell, World

# Agents move between 4-connected cells.
MOVE_OFFSETS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def travel_ticks(steps: int, speed: int) -> int:
    """Ticks to walk `steps` cells at `speed` cells per tick, never less than one."""
    return max(1, math.ceil(steps / speed))


class PathPlanner:
    """BFS distances to a target cell, cached per target.

    Furniture is impassable, so reaching an object means reaching any free cell within one cell of
    it (diagonals included). Walls and furniture never move, so the cache is valid all episode.
    """

    def __init__(self, world: World) -> None:
        self._width = world.width
        self._height = world.height
        self._blocked = world.blocked_cells()
        self._distance_maps: dict[Cell, dict[Cell, int]] = {}

    def steps(self, start: Cell, target: Cell) -> int | None:
        """Path length in cells from `start` to next to `target`, or None if unreachable."""
        return self._distances_to(target).get(start)

    def next_cell(self, start: Cell, target: Cell) -> Cell | None:
        """The neighbour of `start` one step closer to `target`, or None if there or unreachable."""
        distances = self._distances_to(target)
        here = distances.get(start)
        if here is None or here == 0:
            return None
        for dx, dy in MOVE_OFFSETS:
            neighbour = (start[0] + dx, start[1] + dy)
            if distances.get(neighbour) == here - 1:
                return neighbour
        return None

    def _distances_to(self, target: Cell) -> dict[Cell, int]:
        cached = self._distance_maps.get(target)
        if cached is not None:
            return cached
        distances: dict[Cell, int] = {}
        frontier: deque[Cell] = deque()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                goal = (target[0] + dx, target[1] + dy)
                if self._is_free(goal):
                    distances[goal] = 0
                    frontier.append(goal)
        while frontier:
            cell = frontier.popleft()
            for dx, dy in MOVE_OFFSETS:
                neighbour = (cell[0] + dx, cell[1] + dy)
                if neighbour not in distances and self._is_free(neighbour):
                    distances[neighbour] = distances[cell] + 1
                    frontier.append(neighbour)
        self._distance_maps[target] = distances
        return distances

    def _is_free(self, cell: Cell) -> bool:
        in_bounds = 0 <= cell[0] < self._width and 0 <= cell[1] < self._height
        return in_bounds and cell not in self._blocked
