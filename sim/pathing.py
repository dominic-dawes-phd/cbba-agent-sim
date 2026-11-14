"""Shortest-path distances on the grid (hard rule 8).

`PathPlanner.steps_to_any` is the only distance function in the codebase. Called without `avoid`
it sees only walls and furniture; the oracle, numeric CBBA, and the text shown to LLMs all use
that static form, so no condition gets better distance information. Execution passes `avoid`
(other agents' bodies and reserved workspaces) to route around them, which costs ticks but changes
no one's distance information.
"""

import math
from collections import deque

from sim.config import PLANNER_CACHE_SIZE
from sim.world import Cell, World

# Agents move between 4-connected cells.
MOVE_OFFSETS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

NO_CELLS: frozenset[Cell] = frozenset()


def travel_ticks(steps: int, speed: int) -> int:
    """Ticks to walk `steps` cells at `speed` cells per tick, never less than one."""
    return max(1, math.ceil(steps / speed))


class PathPlanner:
    """BFS distances to a set of target cells.

    Furniture is impassable, so reaching an object means reaching any free cell within one cell of
    one of its cells (diagonals included). Static distance maps are cached per target set, the
    oldest dropped beyond `PLANNER_CACHE_SIZE` (explorers target many different floor cells); maps
    that avoid extra cells change every tick and are recomputed.
    """

    def __init__(self, world: World) -> None:
        self._width = world.width
        self._height = world.height
        self._blocked = world.blocked_cells()
        self._distance_maps: dict[tuple[Cell, ...], dict[Cell, int]] = {}

    def steps(self, start: Cell, target: Cell) -> int | None:
        """Static path length from `start` to next to one target cell, or None if unreachable."""
        return self.steps_to_any(start, (target,))

    def steps_to_any(
        self, start: Cell, targets: tuple[Cell, ...], avoid: frozenset[Cell] = NO_CELLS
    ) -> int | None:
        """Path length from `start` to next to any of `targets`, or None if unreachable."""
        return self._distances_to(targets, avoid).get(start)

    def next_cell(
        self, start: Cell, targets: tuple[Cell, ...], avoid: frozenset[Cell] = NO_CELLS
    ) -> Cell | None:
        """The neighbour of `start` one step closer to `targets`, or None if there or stuck."""
        distances = self._distances_to(targets, avoid)
        here = distances.get(start)
        if here is None or here == 0:
            return None
        for dx, dy in MOVE_OFFSETS:
            neighbour = (start[0] + dx, start[1] + dy)
            if distances.get(neighbour) == here - 1:
                return neighbour
        return None

    def _distances_to(self, targets: tuple[Cell, ...], avoid: frozenset[Cell]) -> dict[Cell, int]:
        cached = self._distance_maps.get(targets) if not avoid else None
        if cached is not None:
            return cached
        distances: dict[Cell, int] = {}
        frontier: deque[Cell] = deque()
        for target in targets:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    goal = (target[0] + dx, target[1] + dy)
                    if goal not in distances and self._is_free(goal, avoid):
                        distances[goal] = 0
                        frontier.append(goal)
        while frontier:
            cell = frontier.popleft()
            for dx, dy in MOVE_OFFSETS:
                neighbour = (cell[0] + dx, cell[1] + dy)
                if neighbour not in distances and self._is_free(neighbour, avoid):
                    distances[neighbour] = distances[cell] + 1
                    frontier.append(neighbour)
        if not avoid:
            if len(self._distance_maps) >= PLANNER_CACHE_SIZE:
                del self._distance_maps[next(iter(self._distance_maps))]
            self._distance_maps[targets] = distances
        return distances

    def _is_free(self, cell: Cell, avoid: frozenset[Cell]) -> bool:
        in_bounds = 0 <= cell[0] < self._width and 0 <= cell[1] < self._height
        return in_bounds and cell not in self._blocked and cell not in avoid
