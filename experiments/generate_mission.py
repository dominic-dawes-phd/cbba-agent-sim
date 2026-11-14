"""Seeded household mission generator: python -m experiments.generate_mission <settings.yaml>.

Builds a multi-room house (recursive splitting with 2-wide doors and an outside strip), furnishes
each room from a per-type template, scatters work across rooms so travel dominates, and writes a
map, a mission, and greedy, scripted, and oracle episode configs to
configs/generated/<name>_s<seed>/. The tick budget is measured by replaying the oracle's plan. Same
settings and seed give byte-identical files.

Every generated mission passes the loader's spawn validation (start cells on reachable floor).
"""

import argparse
import dataclasses
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments.run_single import run_config
from sim.config import load_config
from sim.pathing import PathPlanner
from sim.world import Cell, World, WorldObject, chebyshev_distance

GENERATED_DIR = Path("configs/generated")
# LLM-executor episode configs written per house: file label -> condition.
LLM_EPISODES = {
    "greedy_llm": "greedy",
    "numeric_cbba": "numeric_cbba",
    "llm_cbba": "llm_cbba",
    "llm_only": "llm_only",
}
WALL = "#"
OUTSIDE_CHAR = "z"
ROOM_CHARS = "abcdefghijklmnopqrstuvwxy"
BUDGET_ROUNDING = 50
# Furniture dirty dishes may be left on.
DIRTY_SURFACE_KINDS = ("counter", "table", "shelf", "entry_table")

# Furniture per room type: (kind, length in cells along a wall). Kinds map to KIND_PROPERTIES.
ROOM_FURNITURE: dict[str, tuple[tuple[str, int], ...]] = {
    "kitchen": (
        ("counter", 3),
        ("dishwasher", 1),
        ("cabinet", 3),
        ("trashcan", 1),
        ("drying_rack", 1),
        ("pantry", 2),
    ),
    "dining": (("table", 3), ("entry_table", 2), ("shelf", 2)),
    "living": (("table", 2), ("shelf", 3)),
    "bedroom": (("shelf", 2),),
    "study": (("table", 2), ("shelf", 2)),
    "laundry": (("shelf", 2),),
    "bathroom": (("shelf", 1),),
}
# Mission-file properties per furniture kind (cabinet slot_capacity is set from the dish count).
KIND_PROPERTIES: dict[str, dict[str, str | int | bool]] = {
    "counter": {"type": "counter", "surface": True, "slot_capacity": 3},
    "dishwasher": {"type": "dishwasher", "container": True, "open": False, "washing_time": 60},
    "cabinet": {"type": "cabinet", "container": True, "open": False},
    "trashcan": {"type": "trashcan", "container": True, "open": False, "capacity": 5},
    "drying_rack": {"type": "drying_rack", "surface": True},
    "pantry": {"type": "pantry", "container": True, "open": False, "slot_capacity": 2},
    "table": {"type": "table", "surface": True},
    "entry_table": {"type": "table", "surface": True},
    "shelf": {"type": "shelf", "surface": True},
    "bin": {"type": "bin", "container": True},
    "dock": {"type": "dock"},
}
DISH_TYPES = ("plate", "cup")
FOOD_TYPES = ("milk", "bread", "apple", "rice", "eggs")
TIDY_TYPES = ("book", "toy", "sock", "remote", "towel")


@dataclass
class Rect:
    """An inclusive rectangle of cells."""

    x0: int
    y0: int
    x1: int
    y1: int

    def cells(self) -> list[Cell]:
        """Every cell, row by row."""
        return [(x, y) for y in range(self.y0, self.y1 + 1) for x in range(self.x0, self.x1 + 1)]

    def area(self) -> int:
        """Number of cells."""
        return (self.x1 - self.x0 + 1) * (self.y1 - self.y0 + 1)


@dataclass(frozen=True)
class Split:
    """A dividing wall: vertical at x = coord (axis "x") or horizontal at y = coord (axis "y")."""

    axis: str
    coord: int
    low: int
    high: int


@dataclass
class Room:
    """A named room, the map character that marks it, and its floor rectangle."""

    name: str
    char: str
    rect: Rect


@dataclass
class House:
    """The generated layout before it is written out."""

    grid: list[list[str]]
    rooms: list[Room]
    doors: set[Cell]
    furniture: dict[str, tuple[str, list[Cell], str]]  # id -> (kind, cells, room name)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", type=Path, help="generator settings YAML")
    args = parser.parse_args()
    with args.settings.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    for seed in settings["seeds"]:
        out_dir = generate({**settings, "seed": seed})
        print(f"Wrote {out_dir}")


def generate(settings: dict, out_root: Path = GENERATED_DIR) -> Path:
    """Generates one mission (for `settings['seed']`) and returns its output directory."""
    rng = random.Random(f"{settings['seed']}:generator")
    house = build_house(settings, rng)
    mission = build_mission(settings, house, rng)
    out_dir = out_root / f"{settings['name']}_s{settings['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "map.txt").write_text(map_text(house), encoding="utf-8", newline="\n")
    write_yaml(out_dir / "mission.yaml", mission)
    budget = measure_budget(settings, out_dir)
    for condition in ("greedy", "scripted", "oracle"):
        write_yaml(
            out_dir / f"episode_{condition}.yaml",
            episode_config(settings, out_dir, condition, budget),
        )
    for label, condition in LLM_EPISODES.items():
        write_yaml(
            out_dir / f"episode_{label}.yaml",
            episode_config(settings, out_dir, condition, budget, label=label, executor="llm"),
        )
    return out_dir


def build_house(settings: dict, rng: random.Random) -> House:
    """Splits the house into rooms, carves doors, and places furniture."""
    width, height = settings["width"], settings["height"]
    outside_x0 = width - 1 - settings["outside_width"]
    grid = [[WALL] * width for _ in range(height)]
    leaves = [Rect(1, 1, outside_x0 - 2, height - 2)]
    splits: list[Split] = []
    min_side = settings["min_room_side"]
    while len(leaves) < settings["rooms"]:
        candidates = sorted(
            (leaf for leaf in leaves if _can_split(leaf, min_side)),
            key=Rect.area,
            reverse=True,
        )
        if not candidates:
            break
        leaf = candidates[0]
        leaves.remove(leaf)
        first, second, split = _split(leaf, min_side, rng)
        leaves += [first, second]
        splits.append(split)
    leaves.sort(key=Rect.area, reverse=True)
    rooms = []
    type_counts: dict[str, int] = {}
    for index, leaf in enumerate(leaves):
        room_type = settings["room_types"][index % len(settings["room_types"])]
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        rooms.append(Room(f"{room_type}_{type_counts[room_type]}", ROOM_CHARS[index], leaf))
        for x, y in leaf.cells():
            grid[y][x] = ROOM_CHARS[index]
    outside = Rect(outside_x0, 1, width - 2, height - 2)
    for x, y in outside.cells():
        grid[y][x] = OUTSIDE_CHAR
    rooms.append(Room("outside", OUTSIDE_CHAR, outside))
    doors: set[Cell] = set()
    door_width = settings["door_width"]
    for split in splits:
        doors |= _carve_door(grid, split, door_width, rng)
    east_rooms = [room for room in rooms[:-1] if room.rect.x1 == outside_x0 - 2]
    outdoor = Split("x", outside_x0 - 1, east_rooms[0].rect.y0, east_rooms[0].rect.y1)
    doors |= _carve_door(grid, outdoor, door_width, rng)
    house = House(grid, rooms, doors, {})
    for room in rooms[:-1]:
        for kind, length in ROOM_FURNITURE[room.name.rsplit("_", 1)[0]]:
            _place_furniture(house, room, kind, length, gap=1, rng=rng)
    _place_furniture(house, rooms[-1], "bin", 1, gap=1, rng=rng)
    return house


def _can_split(leaf: Rect, min_side: int) -> bool:
    width, height = leaf.x1 - leaf.x0 + 1, leaf.y1 - leaf.y0 + 1
    return max(width, height) >= 2 * min_side + 1


def _split(leaf: Rect, min_side: int, rng: random.Random) -> tuple[Rect, Rect, Split]:
    """Cuts a rectangle along its longer side, leaving a one-cell wall between the halves."""
    width, height = leaf.x1 - leaf.x0 + 1, leaf.y1 - leaf.y0 + 1
    if width >= height:
        x = rng.randint(leaf.x0 + min_side, leaf.x1 - min_side)
        return (
            Rect(leaf.x0, leaf.y0, x - 1, leaf.y1),
            Rect(x + 1, leaf.y0, leaf.x1, leaf.y1),
            Split("x", x, leaf.y0, leaf.y1),
        )
    y = rng.randint(leaf.y0 + min_side, leaf.y1 - min_side)
    return (
        Rect(leaf.x0, leaf.y0, leaf.x1, y - 1),
        Rect(leaf.x0, y + 1, leaf.x1, leaf.y1),
        Split("y", y, leaf.x0, leaf.x1),
    )


def _carve_door(grid: list[list[str]], split: Split, width: int, rng: random.Random) -> set[Cell]:
    """Opens a door of `width` cells in a dividing wall where both sides are floor."""
    options = []
    for start in range(split.low, split.high - width + 2):
        span = range(start, start + width)
        if split.axis == "x":
            sides = [(split.coord + side, along) for along in span for side in (-1, 1)]
        else:
            sides = [(along, split.coord + side) for along in span for side in (-1, 1)]
        if all(grid[y][x] != WALL for x, y in sides):
            options.append(start)
    start = rng.choice(options)
    door = set()
    for along in range(start, start + width):
        x, y = (split.coord, along) if split.axis == "x" else (along, split.coord)
        neighbour = (x - 1, y) if split.axis == "x" else (x, y - 1)
        grid[y][x] = grid[neighbour[1]][neighbour[0]]
        door.add((x, y))
    return door


def _place_furniture(
    house: House, room: Room, kind: str, length: int, gap: int, rng: random.Random
) -> str | None:
    """Places a piece against the room's top or bottom wall, clear of doors and other pieces.

    Returns:
        The new furniture id, or None if no spot fits.
    """
    rect = room.rect
    taken = [cell for _, cells, _ in house.furniture.values() for cell in cells]
    along_rows = [
        [(x + offset, row) for offset in range(length)]
        for row in (rect.y0, rect.y1)
        for x in range(rect.x0, rect.x1 - length + 2)
    ]
    along_columns = [
        [(column, y + offset) for offset in range(length)]
        for column in (rect.x0, rect.x1)
        for y in range(rect.y0, rect.y1 - length + 2)
    ]
    spots = along_rows + along_columns
    rng.shuffle(spots)
    for cells in spots:
        clear_of_doors = all(
            chebyshev_distance(cell, door) > 1 for cell in cells for door in house.doors
        )
        clear_of_pieces = all(
            chebyshev_distance(cell, other) > gap for cell in cells for other in taken
        )
        if clear_of_doors and clear_of_pieces:
            count = sum(1 for other_kind, _, _ in house.furniture.values() if other_kind == kind)
            obj_id = f"{kind}_{count + 1}"
            house.furniture[obj_id] = (kind, cells, room.name)
            return obj_id
    return None


def build_mission(settings: dict, house: House, rng: random.Random) -> dict:
    """Docks, agents, movable objects, and tasks, as a mission-file dict."""
    agents = []
    indoor = house.rooms[:-1]
    for index, agent in enumerate(settings["agents"]):
        # Spread docks over the rooms; if a room is full, try the next one.
        dock_id = None
        for offset in range(len(indoor)):
            room = indoor[(index + offset) % len(indoor)]
            dock_id = _place_furniture(house, room, "dock", 1, gap=2, rng=rng)
            if dock_id is not None:
                break
        if dock_id is None:
            raise ValueError(f"No room anywhere for {agent['id']}'s dock")
        dock_cells, room_name = house.furniture[dock_id][1], house.furniture.pop(dock_id)[2]
        house.furniture[f"dock_{agent['id']}"] = ("dock", dock_cells, room_name)
        start = _free_neighbour(dock_cells[0], house)
        agents.append({**agent, "cell": list(start), "heading": 0})
    world = _world_of(house)
    planner = PathPlanner(world)
    objects: list[dict] = []
    tasks: list[dict] = []
    clusters = _dirty_clusters(settings, house, rng)
    dish_count = sum(stack for _, tiles in clusters for _, stack in tiles)
    furniture_by_kind: dict[str, list[str]] = {}
    for obj_id, (kind, _, _) in sorted(house.furniture.items()):
        furniture_by_kind.setdefault(kind, []).append(obj_id)
    dishwashers = furniture_by_kind["dishwasher"]
    dish_count += settings["unpack_bowls"] * len(dishwashers)
    cabinet_cells = sum(len(house.furniture[obj_id][1]) for obj_id in furniture_by_kind["cabinet"])
    cabinet_capacity = math.ceil(dish_count / cabinet_cells)
    for obj_id, (kind, cells, _) in sorted(house.furniture.items()):
        entry = {"id": obj_id, **KIND_PROPERTIES[kind], "furniture": True}
        entry["cells"] = [list(cells[0]), list(cells[-1])]
        if kind == "cabinet":
            entry["slot_capacity"] = cabinet_capacity
        if kind == "entry_table":
            entry["visual"] = {"entry": "yes"}
        objects.append(entry)

    # Unpacking: clean bowls in every dishwasher, to the cabinet in the same kitchen.
    unpack_of: dict[str, str] = {}
    for dishwasher in dishwashers:
        room = house.furniture[dishwasher][2]
        cabinet = _same_room(house, room, "cabinet") or furniture_by_kind["cabinet"][0]
        bowls = [f"bowl_{dishwasher}_{index + 1}" for index in range(settings["unpack_bowls"])]
        for bowl in bowls:
            objects.append({"id": bowl, "type": "bowl", "parent": dishwasher})
            objects[-1]["visual"] = {"dirty": "no"}
        task_id = f"t_unpack_{dishwasher}"
        # The bowls are inside, so the task is revealed by seeing the dishwasher itself.
        tasks.append(
            {
                "id": task_id,
                "kind": "unpack_dishwasher",
                "appliance": dishwasher,
                "items": bowls,
                "dest": cabinet,
                "reveal_on": [dishwasher],
            }
        )
        unpack_of[dishwasher] = task_id

    # Each cluster of dirty dishes is one task: all loaded, then washed in one cycle, at the
    # dishwasher nearest its surface.
    dish_index = 0
    washes_at: dict[str, int] = {}
    for surface, tiles in clusters:
        dishes = []
        for slot, stack in tiles:
            for _ in range(stack):
                dish_index += 1
                dish = f"{rng.choice(DISH_TYPES)}_{dish_index}"
                dishes.append(dish)
                objects.append(
                    {
                        "id": dish,
                        "type": dish.rsplit("_", 1)[0],
                        "parent": surface,
                        "slot": slot,
                        "visual": {"dirty": "yes"},
                    }
                )
        dishwasher = _nearest(planner, world, surface, dishwashers, house)
        washes_at[dishwasher] = washes_at.get(dishwasher, 0) + 1
        tasks.append(
            {
                "id": f"t_wash_{dishwasher}_{washes_at[dishwasher]}",
                "kind": "clean_dishes",
                "items": dishes,
                "appliance": dishwasher,
                "after": [unpack_of[dishwasher]],
            }
        )

    # Groceries on entry tables, one pack task per item, into the nearest pantry.
    for entry_table in furniture_by_kind.get("entry_table", []):
        for index in range(settings["groceries_per_entry"]):
            food = f"{rng.choice(FOOD_TYPES)}_{entry_table}_{index + 1}"
            objects.append(
                {
                    "id": food,
                    "type": food.split("_")[0],
                    "parent": entry_table,
                    "slot": index % len(house.furniture[entry_table][1]),
                }
            )
            pantry = _nearest(planner, world, entry_table, furniture_by_kind["pantry"], house)
            tasks.append(
                {
                    "id": f"t_pack_{food}",
                    "kind": "pack_groceries",
                    "source": entry_table,
                    "items": [food],
                    "dest": pantry,
                }
            )

    # Trash in every kitchen can, taken to the outdoor bin.
    for can in furniture_by_kind["trashcan"]:
        for index in range(settings["scraps_per_trashcan"]):
            scrap = f"scrap_{can}_{index + 1}"
            objects.append({"id": scrap, "type": "food_scrap", "parent": can})
            objects[-1]["visual"] = {"trash": "yes"}
        bag = f"{can}_bag"
        objects.append({"id": bag, "type": "trash_bag", "parent": can, "visual": {"tied": "no"}})
        tasks.append(
            {
                "id": f"t_trash_{can}",
                "kind": "take_out_trash",
                "can": can,
                "bag": bag,
                "bin": furniture_by_kind["bin"][0],
                "reveal_on": [can],  # the bag is inside; the can's fill gauge shows the need
            }
        )

    # Stray objects on floors, each to a shelf in another room.
    floor = _floor_cells(house)
    shelves = furniture_by_kind["shelf"]
    for index in range(settings["tidy_items"]):
        item = f"{rng.choice(TIDY_TYPES)}_{index + 1}"
        cell = rng.choice(floor)
        floor.remove(cell)
        objects.append({"id": item, "type": item.rsplit("_", 1)[0], "cell": list(cell)})
        room = world.rooms[cell]
        elsewhere = [shelf for shelf in shelves if house.furniture[shelf][2] != room] or shelves
        tasks.append(
            {
                "id": f"t_tidy_{item}",
                "kind": "tidy",
                "item": item,
                "dest": rng.choice(elsewhere),
                "relation": "on",
            }
        )

    assignment = {agent["id"]: [] for agent in agents}
    for index, task in enumerate(tasks):
        assignment[agents[index % len(agents)]["id"]].append(task["id"])
    return {
        "objects": objects,
        "agents": agents,
        "tasks": tasks,
        "scripted_assignment": assignment,
    }


def _world_of(house: House) -> World:
    """A World with walls, rooms, and furniture only, for distances during generation."""
    legend = {room.char: room.name for room in house.rooms}
    walls = set()
    rooms = {}
    for y, row in enumerate(house.grid):
        for x, char in enumerate(row):
            if char == WALL:
                walls.add((x, y))
            else:
                rooms[(x, y)] = legend[char]
    world = World(width=len(house.grid[0]), height=len(house.grid), walls=walls, rooms=rooms)
    for obj_id, (kind, cells, _) in house.furniture.items():
        world.objects[obj_id] = _furniture_object(obj_id, kind, cells)
    return world


def _furniture_object(obj_id: str, kind: str, cells: list[Cell]) -> WorldObject:
    return WorldObject(
        obj_id, kind, cells[0], None, None, {}, {}, is_furniture=True, footprint=tuple(cells)
    )


def _free_neighbour(cell: Cell, house: House) -> Cell:
    """A floor cell next to `cell` that no furniture occupies (an agent's start beside its dock)."""
    taken = {other for _, cells, _ in house.furniture.values() for other in cells}
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        spot = (cell[0] + dx, cell[1] + dy)
        if house.grid[spot[1]][spot[0]] != WALL and spot not in taken and spot not in house.doors:
            return spot
    raise ValueError(f"No free cell next to {cell}")


def _same_room(house: House, room: str, kind: str) -> str | None:
    for obj_id, (other_kind, _, other_room) in sorted(house.furniture.items()):
        if other_kind == kind and other_room == room:
            return obj_id
    return None


def _dirty_clusters(
    settings: dict, house: House, rng: random.Random
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Where the dirty dishes lie: clusters of adjacent surface tiles, each tile a stack of dishes.

    Adjacent dirty tiles are one task, so a cluster never touches another one. Each cluster starts
    on a free tile of a random surface and grows to adjacent free tiles.

    Returns:
        (surface id, [(slot index, stack size)]) per cluster.

    Raises:
        ValueError: If no surface has a tile clear of every earlier cluster.
    """
    surfaces = sorted(
        obj_id for obj_id, (kind, _, _) in house.furniture.items() if kind in DIRTY_SURFACE_KINDS
    )
    taken: list[Cell] = []
    clusters: list[tuple[str, list[tuple[int, int]]]] = []
    for _ in range(settings["dirty_clusters"]):
        placed = None
        for surface in rng.sample(surfaces, len(surfaces)):
            cells = house.furniture[surface][1]
            free = [
                index
                for index, cell in enumerate(cells)
                if all(chebyshev_distance(cell, other) > 1 for other in taken)
            ]
            if free:
                placed = (surface, cells, free)
                break
        if placed is None:
            raise ValueError("No surface has a tile clear of every earlier cluster")
        surface, cells, free = placed
        tiles = [rng.choice(free)]
        wanted = rng.randint(*settings["cluster_tiles"])
        while len(tiles) < wanted:
            adjacent = [
                index
                for index in free
                if index not in tiles
                and any(chebyshev_distance(cells[index], cells[tile]) == 1 for tile in tiles)
            ]
            if not adjacent:
                break
            tiles.append(rng.choice(adjacent))
        taken += [cells[tile] for tile in tiles]
        clusters.append((surface, [(tile, rng.randint(*settings["stack_size"])) for tile in tiles]))
    return clusters


def _nearest(
    planner: PathPlanner, world: World, source: str, options: list[str], house: House
) -> str:
    """The option with the fewest static steps from the source furniture.

    Furniture cells are not walkable, so the distance is measured from the free cells around it.

    Raises:
        ValueError: If no option can be reached from the source.
    """
    blocked = world.blocked_cells()
    standing = [
        (cell[0] + dx, cell[1] + dy)
        for cell in world.objects[source].cells()
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (cell[0] + dx, cell[1] + dy) in world.rooms
        and (cell[0] + dx, cell[1] + dy) not in blocked
    ]
    best = None
    for option in options:
        for cell in standing:
            steps = planner.steps_to_any(cell, tuple(house.furniture[option][1]))
            if steps is not None and (best is None or steps < best[0]):
                best = (steps, option)
    if best is None:
        raise ValueError(f"None of {options} can be reached from {source}")
    return best[1]


def _floor_cells(house: House) -> list[Cell]:
    """Free floor cells away from doors and furniture (where stray objects may lie)."""
    near_furniture = {
        (x + dx, y + dy)
        for _, cells, _ in house.furniture.values()
        for x, y in cells
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    cells = []
    for room in house.rooms[:-1]:
        for cell in room.rect.cells():
            if cell not in near_furniture and cell not in house.doors:
                cells.append(cell)
    return cells


def map_text(house: House) -> str:
    """The map file: legend lines, a separator, then the grid."""
    legend = [f"{room.char}: {room.name}" for room in house.rooms]
    rows = ["".join(row) for row in house.grid]
    return "\n".join([*legend, "---", *rows]) + "\n"


def write_yaml(path: Path, data: dict) -> None:
    """Writes YAML deterministically (insertion order, fixed width) so seeds reproduce exactly."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, width=100)


def episode_config(
    settings: dict,
    out_dir: Path,
    condition: str,
    budget: int,
    label: str | None = None,
    executor: str = "coded",
) -> dict:
    """An episode config for the generated mission, copying the base's sweep settings.

    Task visibility and spawning come from the generator settings, not the base. The oracle (C0)
    runs on known tasks without drops, with the generator's chosen planning method. An LLM episode
    points at the generator's model file, so its run.json records the model hash.

    Args:
        settings: The generator settings.
        out_dir: The generated mission's folder.
        condition: The episode's condition.
        budget: The tick budget.
        label: The config's name suffix, when it is not the condition (G with the LLM executor).
        executor: "coded" or "llm".
    """
    with Path(settings["episode_base"]).open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    config = {
        **base,
        "name": f"{out_dir.name}_{label or condition}",
        "env": out_dir.name,
        "condition": condition,
        "executor": executor,
        "max_ticks": budget,
        "map": (out_dir / "map.txt").as_posix(),
        "mission": (out_dir / "mission.yaml").as_posix(),
        "task_visibility": settings["task_visibility"],
        "spawn": dict(settings["spawn"]),
    }
    if condition == "oracle":
        config["task_visibility"] = "posted"
        config["faults"] = {"drop_prob": 0.0}
        config["oracle"] = {**base["oracle"], "method": settings["oracle_method"]}
    if executor == "llm":
        config["llm"] = {**base["llm"], "model_path": settings["llm_model_path"]}
    return config


def measure_budget(settings: dict, out_dir: Path) -> int:
    """Budget_multiplier x the slowest oracle replay over a few spawns, rounded up.

    The oracle (C0) runs at seeds 0 to `budget_probe_seeds` - 1, so each run gets its own uniform
    spawn and one lucky spawn cannot set a budget too tight for the others. Probes use the heuristic
    planner: a budget needs a stable scale, not a proof, and this keeps generation quick. No
    competing condition sets the yardstick.
    """
    probe = episode_config(settings, out_dir, "oracle", settings["probe_ticks"])
    probe_path = Path(tempfile.mkdtemp()) / "probe.yaml"
    write_yaml(probe_path, probe)
    config = load_config(probe_path)
    makespans = []
    for seed in range(settings["budget_probe_seeds"]):
        seeded = dataclasses.replace(
            config,
            seed=seed,
            output_dir=probe_path.parent,
            oracle=dataclasses.replace(config.oracle, method="heuristic"),
        )
        _, makespan = run_config(seeded)
        if makespan >= settings["probe_ticks"] - 1:
            limit = settings["probe_ticks"]
            raise ValueError(f"The oracle did not finish {out_dir.name} (seed {seed}) in {limit}")
        makespans.append(makespan)
    raw = settings["budget_multiplier"] * max(makespans)
    return math.ceil(raw / BUDGET_ROUNDING) * BUDGET_ROUNDING


if __name__ == "__main__":
    main()
