"""Deterministic perception (hard rule 5): range, view cone, line of sight, containment.

The walls that block sight here are the same walls that attenuate radio in `sim.channel`.
"""

import math
from dataclasses import dataclass

from sim.world import Cell, World, chebyshev_distance

# A view angle at or above this sees in every direction.
FULL_CIRCLE = 360.0


@dataclass(frozen=True)
class Viewpoint:
    """Where an agent is looking from. Heading is in degrees: 0 east, 90 south (y grows down)."""

    cell: Cell
    heading: float
    view_range: float
    view_angle: float


def line_cells(start: Cell, end: Cell) -> list[Cell]:
    """Bresenham line from `start` to `end`, both included."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    step_x = 1 if x1 > x0 else -1
    step_y = 1 if y1 > y0 else -1
    error = dx + dy
    cells = [(x0, y0)]
    while (x0, y0) != (x1, y1):
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y
        cells.append((x0, y0))
    return cells


def walls_between(walls: set[Cell], start: Cell, end: Cell) -> int:
    """Number of wall cells strictly between `start` and `end`."""
    return sum(1 for cell in line_cells(start, end)[1:-1] if cell in walls)


def cell_visible(walls: set[Cell], viewpoint: Viewpoint, target: Cell) -> bool:
    """Whether a cell is within range, inside the view cone, and in line of sight."""
    dx, dy = target[0] - viewpoint.cell[0], target[1] - viewpoint.cell[1]
    if dx == 0 and dy == 0:
        return True
    if math.hypot(dx, dy) > viewpoint.view_range:
        return False
    bearing = math.degrees(math.atan2(dy, dx))
    if abs((bearing - viewpoint.heading + 180) % 360 - 180) > viewpoint.view_angle / 2:
        return False
    return walls_between(walls, viewpoint.cell, target) == 0


class VisibleCells:
    """Map cells (floor and furniture, not walls) each viewpoint can see, cached per viewpoint.

    Walls never move, so a viewpoint always sees the same cells. With a full 360-degree view the
    heading does not matter and is left out of the cache key.
    """

    def __init__(self, world: World) -> None:
        self._walls = world.walls
        self._map_cells = frozenset(world.rooms)
        self._cache: dict[Viewpoint, frozenset[Cell]] = {}

    def of(self, viewpoint: Viewpoint) -> frozenset[Cell]:
        """Cells visible from `viewpoint`."""
        key = viewpoint
        if viewpoint.view_angle >= FULL_CIRCLE:
            key = Viewpoint(viewpoint.cell, 0.0, viewpoint.view_range, viewpoint.view_angle)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        reach = math.ceil(viewpoint.view_range)
        x0, y0 = viewpoint.cell
        cells = frozenset(
            (x0 + dx, y0 + dy)
            for dx in range(-reach, reach + 1)
            for dy in range(-reach, reach + 1)
            if (x0 + dx, y0 + dy) in self._map_cells
            and cell_visible(self._walls, viewpoint, (x0 + dx, y0 + dy))
        )
        self._cache[key] = cells
        return cells


def visible_object_ids(world: World, viewpoint: Viewpoint) -> set[str]:
    """Objects the viewer can see. Contents of closed containers are hidden."""
    visible = set()
    for obj in world.objects.values():
        if not any(cell_visible(world.walls, viewpoint, cell) for cell in obj.cells()):
            continue
        if obj.parent_relation == "in" and not world.objects[obj.parent_id].is_open:
            continue
        visible.add(obj.obj_id)
    return visible


def observed_attrs(world: World, obj_id: str) -> dict[str, str]:
    """Perceivable attributes: open/closed, fill, appliance cycle and countdown, who reserved it.

    Per-slot fill (`slots`, e.g. "3/3,1/3") is visible only when the inside is: an open container
    or a surface. A trashcan's whole-object `fill` is a gauge readable with the lid closed.
    """
    obj = world.objects[obj_id]
    attrs = dict(obj.visual_attrs)
    if obj.is_container:
        attrs["state"] = "open" if obj.is_open else "closed"
    if obj.slot_capacity is not None and obj.is_open:
        attrs["slots"] = ",".join(
            f"{world.slot_count(obj_id, slot)}/{obj.slot_capacity}" for slot in obj.cells()
        )
    if obj.capacity is not None:
        attrs["fill"] = f"{world.contents_count(obj_id)}/{obj.capacity}"
    if obj.washing_time > 0:
        attrs["cycle"] = obj.cycle
        if obj.cycle == "running":
            attrs["cycle_left"] = str(world.cycle_ticks_left[obj_id])
    holders = sorted(
        {world.reservations[cell][0] for cell in obj.cells() if cell in world.reservations}
    )
    if holders:
        attrs["reserved_by"] = ",".join(holders)
    return attrs


def observed_relations(world: World, visible: set[str]) -> list[tuple[str, str, str]]:
    """Relations among visible objects, sorted so identical states give identical text.

    Vocabulary: on, in, held_by, in_room, next_to. next_to is only emitted for adjacent pairs with
    no parent link, never for all pairs.
    """
    relations = []
    ordered = sorted(visible)
    for obj_id in ordered:
        obj = world.objects[obj_id]
        if obj.held_by is not None:
            relations.append((obj_id, "held_by", obj.held_by))
        elif obj.parent_id is not None:
            relations.append((obj_id, obj.parent_relation, obj.parent_id))
        else:
            relations.append((obj_id, "in_room", world.rooms[obj.cell]))
    for index, first_id in enumerate(ordered):
        first = world.objects[first_id]
        for second_id in ordered[index + 1 :]:
            second = world.objects[second_id]
            if first.held_by is not None or second.held_by is not None:
                continue
            linked = first.parent_id == second_id or second.parent_id == first_id
            if chebyshev_distance(first.cell, second.cell) == 1 and not linked:
                relations.append((first_id, "next_to", second_id))
    return relations
