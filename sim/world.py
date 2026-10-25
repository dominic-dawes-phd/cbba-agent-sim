"""World ground truth: grid, walls, rooms, and the object scene graph.

Agents never read this directly (hard rule 5). Perception turns it into beliefs, and primitives
check their preconditions against it when they apply.
"""

from dataclasses import dataclass, field

Cell = tuple[int, int]


def chebyshev_distance(first: Cell, second: Cell) -> int:
    """Grid distance where diagonal neighbours count as adjacent."""
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


@dataclass
class WorldObject:
    """One object or piece of furniture.

    Furniture is static and blocks movement. A movable object sits on or in a piece of furniture
    (`parent_id` set), lies on the floor (`parent_id` None), or is held by an agent (`held_by`).
    """

    obj_id: str
    obj_type: str
    cell: Cell
    parent_id: str | None
    parent_relation: str | None  # "on" | "in"
    visual_attrs: dict[str, str]  # perceivable, e.g. {"dirty": "yes"}
    hidden_attrs: dict[str, str]  # language-only facts, never perceived, e.g. {"fragile": "yes"}
    hands_required: int = 1
    is_furniture: bool = False
    is_surface: bool = False
    is_container: bool = False
    is_open: bool = True
    held_by: str | None = None


@dataclass
class World:
    """Ground-truth state. Cells are (x, y) with y increasing downward."""

    width: int
    height: int
    walls: set[Cell]
    rooms: dict[Cell, str]  # every non-wall cell -> room name
    objects: dict[str, WorldObject] = field(default_factory=dict)
    tick: int = 0

    def blocked_cells(self) -> set[Cell]:
        """Cells no agent can enter: walls plus furniture."""
        furniture_cells = {obj.cell for obj in self.objects.values() if obj.is_furniture}
        return self.walls | furniture_cells

    def hold(self, obj_id: str, agent_id: str, agent_cell: Cell) -> None:
        """Moves an object into an agent's gripper."""
        obj = self.objects[obj_id]
        obj.parent_id = None
        obj.parent_relation = None
        obj.held_by = agent_id
        obj.cell = agent_cell

    def attach(self, obj_id: str, parent_id: str) -> None:
        """Places an object in a container or on a surface."""
        obj = self.objects[obj_id]
        parent = self.objects[parent_id]
        obj.parent_id = parent_id
        obj.parent_relation = "in" if parent.is_container else "on"
        obj.held_by = None
        obj.cell = parent.cell

    def drop(self, obj_id: str, agent_cell: Cell) -> str:
        """Drop rule: lands on an adjacent surface if one exists, else on the floor.

        Returns:
            Where the object ended up, as text for the agent ("on table_1" or "on the floor").
        """
        surfaces = sorted(
            other.obj_id
            for other in self.objects.values()
            if other.is_surface and chebyshev_distance(other.cell, agent_cell) <= 1
        )
        if surfaces:
            self.attach(obj_id, surfaces[0])
            return f"on {surfaces[0]}"
        obj = self.objects[obj_id]
        obj.parent_id = None
        obj.parent_relation = None
        obj.held_by = None
        obj.cell = agent_cell
        return "on the floor"
