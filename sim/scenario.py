"""Scenario loading: an ASCII map plus a YAML mission (objects, agents, tasks).

Map file format: legend lines `<char>: <room name>`, a line `---`, then the grid. `#` is a wall and
every other character is a floor cell of the room its legend names.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from sim.agent import CapabilityCard, ToolCall
from sim.world import Cell, World, WorldObject

WALL_CHAR = "#"
LEGEND_SEPARATOR = "---"
# Allowed goal relations. Quote "on" in YAML, which otherwise parses it as the boolean True.
GOAL_RELATIONS = ("in", "on", "is")


@dataclass(frozen=True)
class GoalCondition:
    """A ground-truth fact: (obj, "in" | "on", parent) or (obj, "is", "open" | "closed")."""

    obj_id: str
    relation: str
    value: str


@dataclass(frozen=True)
class TaskSpec:
    """One mission task.

    Attributes:
        after: Task ids that must be done (in ground truth) before this task counts as done.
        language_only: The task carries a constraint stated only in its description (the rho knob).
        steps: Primitive steps, used by the scripted test policy only.
    """

    task_id: str
    description: str
    goals: tuple[GoalCondition, ...]
    after: tuple[str, ...]
    language_only: bool
    steps: tuple[ToolCall, ...]


@dataclass(frozen=True)
class AgentSpec:
    """An agent's identity, capabilities, and starting pose."""

    agent_id: str
    card: CapabilityCard
    cell: Cell
    heading: float


@dataclass
class Scenario:
    """Everything a mission file defines."""

    world: World
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    scripted_assignment: dict[str, list[str]]


def load_map(path: Path) -> World:
    """Builds an empty world (walls and rooms only) from an ASCII map file."""
    legend: dict[str, str] = {}
    grid_lines: list[str] = []
    in_grid = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not in_grid:
            if line == LEGEND_SEPARATOR:
                in_grid = True
            elif line:
                char, room = line.split(":", 1)
                legend[char.strip()] = room.strip()
            continue
        if line:
            grid_lines.append(line)
    walls: set[Cell] = set()
    rooms: dict[Cell, str] = {}
    for y, row in enumerate(grid_lines):
        for x, char in enumerate(row):
            if char == WALL_CHAR:
                walls.add((x, y))
            else:
                rooms[(x, y)] = legend[char]
    width = max(len(row) for row in grid_lines)
    return World(width=width, height=len(grid_lines), walls=walls, rooms=rooms)


def load_scenario(map_path: Path, mission_path: Path) -> Scenario:
    """Loads the map and places the mission's furniture, objects, agents, and tasks."""
    world = load_map(map_path)
    with mission_path.open(encoding="utf-8") as handle:
        mission = yaml.safe_load(handle)
    for entry in mission["objects"]:
        parent_id = entry.get("parent")
        cell = world.objects[parent_id].cell if parent_id else tuple(entry["cell"])
        parent_relation = None
        if parent_id:
            parent_relation = "in" if world.objects[parent_id].is_container else "on"
        world.objects[entry["id"]] = WorldObject(
            obj_id=entry["id"],
            obj_type=entry["type"],
            cell=cell,
            parent_id=parent_id,
            parent_relation=parent_relation,
            visual_attrs=dict(entry.get("visual", {})),
            hidden_attrs=dict(entry.get("hidden", {})),
            hands_required=int(entry.get("hands", 1)),
            is_furniture=bool(entry.get("furniture", False)),
            is_surface=bool(entry.get("surface", False)),
            is_container=bool(entry.get("container", False)),
            is_open=bool(entry.get("open", True)),
        )
    agents = [
        AgentSpec(
            agent_id=entry["id"],
            card=CapabilityCard(
                hands=int(entry["hands"]),
                speed=int(entry["speed"]),
                view_range=float(entry["view_range"]),
                view_angle=float(entry["view_angle"]),
                can_scan=bool(entry.get("can_scan", False)),
                success={name: float(prob) for name, prob in entry.get("success", {}).items()},
            ),
            cell=tuple(entry["cell"]),
            heading=float(entry["heading"]),
        )
        for entry in mission["agents"]
    ]
    for entry in mission["tasks"]:
        for goal in entry["goals"]:
            if goal[1] not in GOAL_RELATIONS:
                raise ValueError(f"Task {entry['id']} has goal relation {goal[1]!r}")
    tasks = [
        TaskSpec(
            task_id=entry["id"],
            description=entry["description"],
            goals=tuple(GoalCondition(*goal) for goal in entry["goals"]),
            after=tuple(entry.get("after", [])),
            language_only=bool(entry.get("language_only", False)),
            steps=tuple(ToolCall(step[0], tuple(step[1:])) for step in entry.get("steps", [])),
        )
        for entry in mission["tasks"]
    ]
    assignment = {
        agent_id: list(task_ids)
        for agent_id, task_ids in mission.get("scripted_assignment", {}).items()
    }
    return Scenario(world, agents, tasks, assignment)


def goal_holds(world: World, goal: GoalCondition) -> bool:
    """Whether a goal condition holds in ground truth."""
    obj = world.objects[goal.obj_id]
    if goal.relation == "is":
        return obj.is_open == (goal.value == "open")
    return obj.parent_id == goal.value and obj.parent_relation == goal.relation
