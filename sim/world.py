"""World ground truth: grid, walls, rooms, and the object scene graph.

Agents never read this directly (hard rule 5). Perception turns it into beliefs, and primitives
check their preconditions against it when they apply.
"""

from dataclasses import dataclass, field

Cell = tuple[int, int]

BAG_TYPE = "trash_bag"
SHARDS_TYPE = "shards"


def chebyshev_distance(first: Cell, second: Cell) -> int:
    """Grid distance where diagonal neighbours count as adjacent."""
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def split_slot(name: str) -> tuple[str, int | None]:
    """Splits a slot reference like "cabinet_1[2]" into ("cabinet_1", 2).

    A plain object id returns (id, None). A malformed index returns the whole name, which then
    fails the belief check like any unknown object.
    """
    if not name.endswith("]") or "[" not in name:
        return name, None
    base, index = name[:-1].split("[", 1)
    if not index.isdigit():
        return name, None
    return base, int(index)


@dataclass
class WorldObject:
    """One object or piece of furniture.

    Furniture is static, blocks movement, and may span several cells (`footprint`). Each footprint
    cell is a slot; a movable object sitting on or in furniture has its `cell` set to its slot. A
    movable object may also lie on the floor (`parent_id` None) or be held by an agent (`held_by`).
    """

    obj_id: str
    obj_type: str
    cell: Cell
    parent_id: str | None
    parent_relation: str | None  # "on" | "in"
    visual_attrs: dict[str, str]  # perceivable, e.g. {"dirty": "yes"}
    hidden_attrs: dict[str, str]  # facts never perceived; only `scan` reveals them
    hands_required: int = 1
    is_furniture: bool = False
    is_surface: bool = False
    is_container: bool = False
    is_open: bool = True
    held_by: str | None = None
    footprint: tuple[Cell, ...] = ()  # furniture cells in slot order; empty means just `cell`
    capacity: int | None = None  # whole-object item limit (a trashcan)
    slot_capacity: int | None = None  # item limit per footprint cell (cupboards, pantries)
    washing_time: int = 0  # appliances only: ticks one wash cycle runs; 0 means not an appliance
    cycle: str = "idle"  # appliances only: "idle" | "running" | "done"

    def cells(self) -> tuple[Cell, ...]:
        """Every cell the object occupies, in slot order."""
        return self.footprint or (self.cell,)


def within_reach(agent_cell: Cell, obj: WorldObject) -> bool:
    """Whether an agent can touch an object: Chebyshev distance 1 to any of its cells."""
    return any(chebyshev_distance(agent_cell, cell) <= 1 for cell in obj.cells())


@dataclass
class World:
    """Ground-truth state. Cells are (x, y) with y increasing downward.

    Attributes:
        reservations: Workspace cells held by a lock: cell -> (holder, furniture id). The holder is
            an agent, or an appliance whose running cycle took over the lock.
    """

    width: int
    height: int
    walls: set[Cell]
    rooms: dict[Cell, str]  # every non-wall cell -> room name
    objects: dict[str, WorldObject] = field(default_factory=dict)
    tick: int = 0
    cycle_ticks_left: dict[str, int] = field(default_factory=dict)  # running appliance -> ticks
    broken: list[str] = field(default_factory=list)  # objects shattered since the loop last looked
    lost: set[str] = field(default_factory=set)  # every object shattered so far this episode
    drops: list[tuple[str, str]] = field(default_factory=list)  # (agent, object) since last looked
    # (appliance, agent who started it) since the loop last looked.
    cycles_started: list[tuple[str, str]] = field(default_factory=list)
    reservations: dict[Cell, tuple[str, str]] = field(default_factory=dict)

    def blocked_cells(self) -> set[Cell]:
        """Cells no agent can enter: walls plus every furniture cell."""
        furniture_cells = {
            cell for obj in self.objects.values() if obj.is_furniture for cell in obj.cells()
        }
        return self.walls | furniture_cells

    def contents_count(self, container_id: str) -> int:
        """Items inside a container. Each counts 1 against its capacity, bags excepted."""
        return sum(
            1
            for obj in self.objects.values()
            if obj.parent_id == container_id and obj.obj_type != BAG_TYPE
        )

    def slot_count(self, furniture_id: str, slot: Cell) -> int:
        """Items sitting in one slot of a piece of furniture."""
        return sum(
            1
            for obj in self.objects.values()
            if obj.parent_id == furniture_id and obj.cell == slot and obj.held_by is None
        )

    def slot_has_room(self, furniture_id: str, slot: Cell) -> bool:
        """Whether one more item fits in a slot."""
        furniture = self.objects[furniture_id]
        if furniture.capacity is not None:
            return self.contents_count(furniture_id) < furniture.capacity
        if furniture.slot_capacity is None:
            return True
        return self.slot_count(furniture_id, slot) < furniture.slot_capacity

    def reserved_by_other(self, cell: Cell, agent_id: str) -> str | None:
        """The holder of a reservation on `cell`, unless it is `agent_id` or there is none."""
        reservation = self.reservations.get(cell)
        if reservation is None or reservation[0] == agent_id:
            return None
        return reservation[0]

    def release_reservations(self, holder: str, furniture_id: str | None = None) -> None:
        """Frees a holder's reservations, on one piece of furniture or on everything."""
        for cell in sorted(self.reservations):
            held_by, held_on = self.reservations[cell]
            if held_by == holder and furniture_id in (None, held_on):
                del self.reservations[cell]

    def start_cycle(self, appliance_id: str, agent_id: str) -> None:
        """Starts a wash cycle. The agent's lock on the appliance passes to the running cycle."""
        self.objects[appliance_id].cycle = "running"
        self.cycle_ticks_left[appliance_id] = self.objects[appliance_id].washing_time
        self.cycles_started.append((appliance_id, agent_id))
        for cell, reservation in self.reservations.items():
            if reservation == (agent_id, appliance_id):
                self.reservations[cell] = (appliance_id, appliance_id)

    def advance_cycles(self) -> None:
        """Counts running cycles down. A finished cycle cleans its contents and frees its lock."""
        for appliance_id in sorted(self.cycle_ticks_left):
            self.cycle_ticks_left[appliance_id] -= 1
            if self.cycle_ticks_left[appliance_id] > 0:
                continue
            del self.cycle_ticks_left[appliance_id]
            self.objects[appliance_id].cycle = "done"
            self.release_reservations(appliance_id)
            for obj in self.objects.values():
                if obj.parent_id == appliance_id and "dirty" in obj.visual_attrs:
                    obj.visual_attrs["dirty"] = "no"

    def break_object(self, obj_id: str) -> None:
        """Shatters an object in place. The shards keep the object id and still count as an item."""
        obj = self.objects[obj_id]
        obj.obj_type = SHARDS_TYPE
        obj.visual_attrs = {"broken": "yes"}
        self.broken.append(obj_id)
        self.lost.add(obj_id)

    def hold(self, obj_id: str, agent_id: str, agent_cell: Cell) -> None:
        """Moves an object into an agent's gripper."""
        obj = self.objects[obj_id]
        obj.parent_id = None
        obj.parent_relation = None
        obj.held_by = agent_id
        obj.cell = agent_cell

    def attach(self, obj_id: str, parent_id: str, slot: Cell) -> None:
        """Places an object in one slot of a container or surface."""
        obj = self.objects[obj_id]
        parent = self.objects[parent_id]
        obj.parent_id = parent_id
        obj.parent_relation = "in" if parent.is_container else "on"
        obj.held_by = None
        obj.cell = slot

    def drop(self, obj_id: str, agent_cell: Cell) -> None:
        """Drop rule: the object falls to the floor on the agent's own tile."""
        obj = self.objects[obj_id]
        obj.parent_id = None
        obj.parent_relation = None
        obj.held_by = None
        obj.cell = agent_cell
