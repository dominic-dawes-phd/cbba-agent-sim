"""Deterministic perception (hard rule 5): range, view cone, line of sight, containment.

The walls that block sight here are the same walls that attenuate radio in `sim.channel`.
"""

import math
from dataclasses import dataclass

from sim.world import Cell, World, chebyshev_distance


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


def visible_object_ids(world: World, viewpoint: Viewpoint) -> set[str]:
    """Objects the viewer can see. Contents of closed containers are hidden."""
    visible = set()
    for obj in world.objects.values():
        if not cell_visible(world.walls, viewpoint, obj.cell):
            continue
        if obj.parent_relation == "in" and not world.objects[obj.parent_id].is_open:
            continue
        visible.add(obj.obj_id)
    return visible


def observed_attrs(world: World, obj_id: str) -> dict[str, str]:
    """Perceivable attributes of an object, including the open/closed state of containers."""
    obj = world.objects[obj_id]
    attrs = dict(obj.visual_attrs)
    if obj.is_container:
        attrs["state"] = "open" if obj.is_open else "closed"
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
