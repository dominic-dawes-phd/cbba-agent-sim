"""Small worlds and agents for unit tests."""

from sim.agent import AgentState, CapabilityCard
from sim.world import Cell, World, WorldObject


def make_world(rows: list[str]) -> World:
    """Builds a world from rows of '#' (wall) and '.' (floor in room 'room')."""
    walls = set()
    rooms = {}
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))
            else:
                rooms[(x, y)] = "room"
    return World(width=len(rows[0]), height=len(rows), walls=walls, rooms=rooms)


def add_furniture(
    world: World,
    obj_id: str,
    cell: Cell,
    surface: bool = False,
    container: bool = False,
    is_open: bool = True,
) -> WorldObject:
    obj = WorldObject(
        obj_id,
        obj_id.split("_")[0],
        cell,
        None,
        None,
        {},
        {},
        is_furniture=True,
        is_surface=surface,
        is_container=container,
        is_open=is_open,
    )
    world.objects[obj_id] = obj
    return obj


def add_item(
    world: World,
    obj_id: str,
    cell: Cell,
    parent_id: str | None = None,
    hands: int = 1,
    hidden: dict[str, str] | None = None,
) -> WorldObject:
    relation = None
    if parent_id is not None:
        relation = "in" if world.objects[parent_id].is_container else "on"
        cell = world.objects[parent_id].cell
    obj = WorldObject(
        obj_id,
        obj_id.split("_")[0],
        cell,
        parent_id,
        relation,
        {},
        dict(hidden or {}),
        hands_required=hands,
    )
    world.objects[obj_id] = obj
    return obj


def make_agent(
    agent_id: str,
    cell: Cell,
    heading: float = 0.0,
    hands: int = 1,
    speed: int = 1,
    view_angle: float = 360.0,
    success: dict[str, float] | None = None,
    can_scan: bool = False,
) -> AgentState:
    card = CapabilityCard(
        hands=hands,
        speed=speed,
        view_range=10.0,
        view_angle=view_angle,
        can_scan=can_scan,
        success=dict(success or {}),
    )
    return AgentState(agent_id, card, cell, heading, {})
