"""Scenario loading: an ASCII map plus a YAML mission (objects, agents, tasks).

Map file format: legend lines `<char>: <room name>`, a line `---`, then the grid. `#` is a wall and
every other character is a floor cell of the room its legend names.

Mission objects: furniture gives `cell` or `cells: [corner, corner]` (a rectangle whose cells are
its slots, numbered row by row) plus optional `slot_capacity`, `capacity`, and `washing_time`. A
movable object gives `parent` and optionally `slot` (default 0).
"""

import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from sim.agent import CapabilityCard, ToolCall
from sim.config import TASK_TEMPLATES_PATH
from sim.pathing import PathPlanner
from sim.world import Cell, World, WorldObject

WALL_CHAR = "#"
LEGEND_SEPARATOR = "---"
# Allowed goal relations. Quote "on" in YAML, which otherwise parses it as the boolean True.
# (obj, "cycle", "done") is an appliance's wash-cycle state. (obj, "in_kind", "cabinet|drying_rack")
# holds when the object sits in or on furniture of any listed type. (obj, "washed", "yes") holds
# once a wash cycle that held the dish has ended, in whichever dishwasher.
GOAL_RELATIONS = ("in", "on", "is", "cycle", "in_kind", "washed")
# Mission task-entry keys that are not template parameters. `reveal_on` lists the objects whose
# sight reveals the task when tasks are discovered rather than posted.
TASK_ENTRY_KEYS = ("id", "kind", "after", "reveal_on")
# Headings a uniformly spawned agent may start with.
SPAWN_HEADINGS = (0.0, 90.0, 180.0, 270.0)


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
        steps: Primitive steps, used by the scripted test policy only.
        reveal_on: Objects whose sight reveals the task in discovered mode. Empty means its
            movable goal objects (the loop resolves the default against the world).
    """

    task_id: str
    description: str
    goals: tuple[GoalCondition, ...]
    after: tuple[str, ...]
    steps: tuple[ToolCall, ...]
    kind: str | None = None
    reveal_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSpec:
    """An agent's identity, capabilities, and starting pose."""

    agent_id: str
    card: CapabilityCard
    cell: Cell
    heading: float


@dataclass
class Scenario:
    """Everything a mission file defines.

    Attributes:
        tasks: The mission's tasks. Tasks discovered during the episode are appended.
        templates: Task templates by kind, for building discovered tasks.
    """

    world: World
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    scripted_assignment: dict[str, list[str]]
    templates: dict[str, dict]


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
        footprint = rectangle(entry["cells"]) if "cells" in entry else ()
        parent_relation = None
        if parent_id:
            parent = world.objects[parent_id]
            cell = parent.cells()[int(entry.get("slot", 0))]
            parent_relation = "in" if parent.is_container else "on"
        else:
            cell = footprint[0] if footprint else tuple(entry["cell"])
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
            footprint=footprint,
            capacity=entry.get("capacity"),
            slot_capacity=entry.get("slot_capacity"),
            washing_time=int(entry.get("washing_time", 0)),
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
    validate_spawns(world, agents)
    with TASK_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        templates = yaml.safe_load(handle)
    tasks = [task_from_entry(entry, templates) for entry in mission["tasks"]]
    assignment = {
        agent_id: list(task_ids)
        for agent_id, task_ids in mission.get("scripted_assignment", {}).items()
    }
    return Scenario(world, agents, tasks, assignment, templates)


def validate_spawns(world: World, agents: list[AgentSpec]) -> None:
    """Rejects start cells an agent could not really start on.

    Each start cell must be floor (not a wall, outside the map, or furniture), not shared, and able
    to reach every piece of furniture, so no agent is sealed in a pocket. Reachability uses the
    same static distance function as everything else (hard rule 8).

    Raises:
        ValueError: Naming the first agent and the reason.
    """
    blocked = world.blocked_cells()
    planner = PathPlanner(world)
    furniture = sorted(obj_id for obj_id, obj in world.objects.items() if obj.is_furniture)
    seen: dict[Cell, str] = {}
    for spec in agents:
        if spec.cell not in world.rooms:
            raise ValueError(f"Agent {spec.agent_id} starts at {spec.cell}, a wall or off the map")
        if spec.cell in blocked:
            raise ValueError(f"Agent {spec.agent_id} starts at {spec.cell}, on furniture")
        if spec.cell in seen:
            raise ValueError(
                f"Agents {seen[spec.cell]} and {spec.agent_id} both start at {spec.cell}"
            )
        seen[spec.cell] = spec.agent_id
        for obj_id in furniture:
            if planner.steps_to_any(spec.cell, world.objects[obj_id].cells()) is None:
                raise ValueError(f"Agent {spec.agent_id} at {spec.cell} cannot reach {obj_id}")


def draw_spawns(
    world: World, agents: list[AgentSpec], rng: random.Random, min_separation: int
) -> list[AgentSpec]:
    """Seeded uniform start cells and headings, replacing the mission's.

    Candidates are the free floor cells connected to the mission's first start cell (which is
    validated, so every candidate reaches all furniture). Cells are tried in a seeded shuffled order
    and kept if they are at least `min_separation` static steps from every cell kept so far. The
    result passes `validate_spawns`.

    Raises:
        ValueError: If the floor has too few cells far enough apart.
    """
    planner = PathPlanner(world)
    blocked = world.blocked_cells()
    anchor = agents[0].cell
    candidates = [
        cell
        for cell in sorted(world.rooms)
        if cell not in blocked and planner.steps(cell, anchor) is not None
    ]
    rng.shuffle(candidates)
    chosen: list[Cell] = []
    for cell in candidates:
        if all(planner.steps(cell, other) >= min_separation for other in chosen):
            chosen.append(cell)
        if len(chosen) == len(agents):
            break
    if len(chosen) < len(agents):
        raise ValueError(f"Only {len(chosen)} start cells are {min_separation} steps apart")
    spawned = [
        dataclasses.replace(spec, cell=cell, heading=rng.choice(SPAWN_HEADINGS))
        for spec, cell in zip(agents, chosen, strict=True)
    ]
    validate_spawns(world, spawned)
    return spawned


def rectangle(corners: list[list[int]]) -> tuple[Cell, ...]:
    """Cells of the rectangle between two corner cells, row by row (this is the slot order)."""
    (x0, y0), (x1, y1) = corners
    return tuple(
        (x, y)
        for y in range(min(y0, y1), max(y0, y1) + 1)
        for x in range(min(x0, x1), max(x0, x1) + 1)
    )


def task_from_entry(entry: dict, templates: dict[str, dict]) -> TaskSpec:
    """Builds a task from a mission entry: explicit goals and steps, or a `kind` template.

    A templated entry names the kind and supplies its parameters. Parameters that are lists (the
    `over` parameter of a template's `each` block) are unrolled into one copy of the block's steps
    and goals per element, so a task like "unpack k dishes" becomes a flat step list.
    """
    if "kind" in entry:
        template = templates[entry["kind"]]
        params = {key: value for key, value in entry.items() if key not in TASK_ENTRY_KEYS}
        steps = [fill(step, params) for step in template.get("before", [])]
        goals = [fill(goal, params) for goal in template.get("goals", [])]
        each = template.get("each")
        if each is not None:
            for item in params[each["over"]]:
                item_params = {**params, "item": item}
                steps += [fill(step, item_params) for step in each["steps"]]
                goals += [fill(goal, item_params) for goal in each["goals"]]
        steps += [fill(step, params) for step in template.get("after", [])]
        words = {
            key: ", ".join(value) if isinstance(value, list) else value
            for key, value in params.items()
        }
        description = template["description"].format(**words)
    else:
        steps = entry.get("steps", [])
        goals = entry["goals"]
        description = entry["description"]
    for goal in goals:
        if goal[1] not in GOAL_RELATIONS:
            raise ValueError(f"Task {entry['id']} has goal relation {goal[1]!r}")
    return TaskSpec(
        task_id=entry["id"],
        description=description,
        goals=tuple(GoalCondition(*goal) for goal in goals),
        after=tuple(entry.get("after", [])),
        steps=tuple(ToolCall(step[0], tuple(step[1:])) for step in steps),
        kind=entry.get("kind"),
        reveal_on=tuple(entry.get("reveal_on", ())),
    )


def fill(row: list[str], values: dict[str, str]) -> list[str]:
    """Substitutes `{name}` placeholders in one step or goal row."""
    return [cell.format(**values) for cell in row]


def goal_holds(world: World, goal: GoalCondition) -> bool:
    """Whether a goal condition holds in ground truth. Deleted (bagged) objects hold none."""
    obj = world.objects.get(goal.obj_id)
    if obj is None:
        return False
    if goal.relation == "is" and goal.value == "tied":
        return obj.visual_attrs.get("tied") == "yes"
    if goal.relation == "is":
        return obj.is_open == (goal.value == "open")
    if goal.relation == "cycle":
        return obj.cycle == goal.value
    if goal.relation == "washed":
        return obj.visual_attrs.get("dirty") == "no"
    if goal.relation == "in_kind":
        return obj.parent_id is not None and (
            world.objects[obj.parent_id].obj_type in goal.value.split("|")
        )
    return obj.parent_id == goal.value and obj.parent_relation == goal.relation
