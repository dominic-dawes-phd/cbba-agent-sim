"""Perception-triggered task discovery.

After perception, each agent checks its own beliefs (never ground truth, hard rule 5) for states
that call for a task nobody it knows of has created yet: a dirty dish outside the dishwasher,
washed dishes inside a dishwasher, groceries left on the entry table, a clean dish on the drying
rack, a trashcan more than half full.
Task ids come from the triggering object, so two agents who spot the same dish create the same task
and the ledger can dedupe them. Shards are not a task: the agent that broke the dish cleans up by
reflex (`sim.primitives.start_clean_up`).
"""

from sim.beliefs import Belief
from sim.config import DISH_TYPES

TRASHCAN_TYPE = "trashcan"
DISHWASHER_TYPE = "dishwasher"
PANTRY_TYPE = "pantry"
DRYING_RACK_TYPE = "drying_rack"
CABINET_TYPE = "cabinet"
UNLOAD_KIND = "unpack_dishwasher"
BIN_TYPE = "bin"
BAG_SUFFIX = "_bag"  # a trashcan's liner is named "<can id>_bag"
# Triggers that follow up on another task's outcome, so that task does not cover them.
FOLLOW_UP_KINDS = ("store_dish", UNLOAD_KIND)

TaskEntry = dict[str, str | list[str]]


def find_triggered_tasks(
    beliefs: dict[str, Belief], covered: set[tuple[str | None, str]]
) -> list[TaskEntry]:
    """Mission-style task entries for every trigger the beliefs show.

    Args:
        beliefs: The agent's belief graph.
        covered: (task kind, object id) pairs from the goals of tasks the agent knows. Any known
            task about an object covers it, except for follow-up kinds (`FOLLOW_UP_KINDS`), which
            exist because of another task's outcome (a dish racked by an unpack): those need a task
            of their own kind, or a hand-written task (kind None).

    Returns:
        Entries for `scenario.task_from_entry`, in object id order.
    """
    dishwashers = _ids_of_type(beliefs, DISHWASHER_TYPE)
    pantries = _ids_of_type(beliefs, PANTRY_TYPE)
    bins = _ids_of_type(beliefs, BIN_TYPE)
    entries: list[TaskEntry] = []
    washed: dict[str, list[str]] = {}  # dishwasher -> clean dishes believed inside
    for obj_id in sorted(beliefs):
        belief = beliefs[obj_id]
        if belief.missing or any(relation == "held_by" for _, relation, _ in belief.relations):
            continue
        parents = [
            parent_id
            for _, relation, parent_id in belief.relations
            if relation in ("on", "in") and parent_id in beliefs
        ]
        parent_types = {beliefs[parent_id].obj_type for parent_id in parents}
        dirty = belief.visual_attrs.get("dirty")
        if belief.obj_type in DISH_TYPES and dirty == "no" and DISHWASHER_TYPE in parent_types:
            washed.setdefault(parents[0], []).append(obj_id)
        elif belief.obj_type in DISH_TYPES and dirty == "yes":
            if DISHWASHER_TYPE not in parent_types and dishwashers:
                entry = {"id": f"clean_{obj_id}", "kind": "clean_plate", "item": obj_id}
                entries.append({**entry, "appliance": dishwashers[0]})
        elif belief.obj_type in DISH_TYPES and DRYING_RACK_TYPE in parent_types:
            entry = {"id": f"store_{obj_id}", "kind": "store_dish", "item": obj_id}
            entries.append({**entry, "rack": parents[0]})
        elif parents and beliefs[parents[0]].visual_attrs.get("entry") == "yes" and pantries:
            entry = {"id": f"pack_{obj_id}", "kind": "pack_groceries", "source": parents[0]}
            entries.append({**entry, "items": [obj_id], "dest": pantries[0]})
        elif belief.obj_type == TRASHCAN_TYPE and _more_than_half(belief) and bins:
            entry = {"id": f"trash_{obj_id}", "kind": "take_out_trash", "can": obj_id}
            entries.append({**entry, "bag": f"{obj_id}{BAG_SUFFIX}", "bin": bins[0]})
    for appliance, dishes in sorted(washed.items()):
        # One unload per dishwasher, for the dishes no known unload covers yet.
        new = [obj_id for obj_id in dishes if (UNLOAD_KIND, obj_id) not in covered]
        if new and _ids_of_type(beliefs, CABINET_TYPE):
            entries.append(unload_entry(appliance, new, unload_dest(appliance, beliefs)))
    with_tasks = {obj_id for _, obj_id in covered}
    kept = []
    for entry in entries:
        key = _key_object(entry)
        if entry["kind"] in FOLLOW_UP_KINDS:
            if (entry["kind"], key) not in covered and (None, key) not in covered:
                kept.append(entry)
        elif key not in with_tasks:
            kept.append(entry)
    return kept


def unload_entry(appliance: str, dishes: list[str], dest: str) -> TaskEntry:
    """The task that puts washed dishes away. The id comes from the dishes, so finders agree."""
    return {
        "id": f"unload_{appliance}_{min(dishes)}",
        "kind": UNLOAD_KIND,
        "items": sorted(dishes),
        "appliance": appliance,
        "dest": dest,
    }


def unload_dest(appliance: str, beliefs: dict[str, Belief]) -> str:
    """A cupboard in the dishwasher's room, else the first cupboard (by id) the agent knows."""
    cabinets = _ids_of_type(beliefs, CABINET_TYPE)
    rooms = {
        obj_id: room
        for obj_id in [appliance, *cabinets]
        for _, relation, room in beliefs[obj_id].relations
        if relation == "in_room"
    }
    same_room = [cabinet for cabinet in cabinets if rooms.get(cabinet) == rooms.get(appliance)]
    return (same_room or cabinets)[0]


def _ids_of_type(beliefs: dict[str, Belief], obj_type: str) -> list[str]:
    return sorted(obj_id for obj_id, belief in beliefs.items() if belief.obj_type == obj_type)


def _more_than_half(belief: Belief) -> bool:
    fill = belief.visual_attrs.get("fill")
    if fill is None:
        return False
    count, capacity = fill.split("/")
    return 2 * int(count) > int(capacity)


def _key_object(entry: TaskEntry) -> str:
    """The object a triggered task's goals are about, used to skip tasks already known."""
    if entry["kind"] == "take_out_trash":
        return entry["bag"]
    if entry["kind"] in ("pack_groceries", UNLOAD_KIND):
        return entry["items"][0]
    return entry["item"]
