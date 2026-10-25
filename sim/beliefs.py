"""Per-agent belief graph: what an agent thinks is true, which can go stale."""

from dataclasses import dataclass, field

from sim.pathing import PathPlanner
from sim.perception import (
    Viewpoint,
    cell_visible,
    observed_attrs,
    observed_relations,
    visible_object_ids,
)
from sim.world import Cell, World


@dataclass
class Belief:
    """One agent's view of one object.

    Attributes:
        source: "seen" (own perception), "told" (teammate gossip), or "map" (static furniture
            known at episode start).
        scanned_attrs: Hidden attributes revealed by `scan`. Kept when the object is seen again.
    """

    obj_type: str
    relations: list[tuple[str, str, str]]
    visual_attrs: dict[str, str]
    believed_cell: Cell
    last_seen: int
    source: str
    told_by: str | None = None
    missing: bool = False
    scanned_attrs: dict[str, str] = field(default_factory=dict)


def initial_beliefs(world: World) -> dict[str, Belief]:
    """Static furniture is known from the map at tick 0. Movable objects must be perceived."""
    beliefs = {}
    for obj_id in sorted(world.objects):
        obj = world.objects[obj_id]
        if obj.is_furniture:
            beliefs[obj.obj_id] = Belief(
                obj.obj_type,
                [(obj.obj_id, "in_room", world.rooms[obj.cell])],
                observed_attrs(world, obj.obj_id),
                obj.cell,
                world.tick,
                "map",
            )
    return beliefs


def update_beliefs(
    beliefs: dict[str, Belief], world: World, viewpoint: Viewpoint
) -> list[tuple[str, str]]:
    """Applies one tick of perception to an agent's beliefs.

    Visible objects are overwritten with fresh relations and attributes. A believed object whose
    cell is in view but which is not itself visible is marked missing (negative evidence), except
    when it is believed to be inside a container that is now closed. Everything else goes stale.

    Returns:
        (object id, change) pairs, change being "added", "updated", or "missing", in id order.
    """
    visible = visible_object_ids(world, viewpoint)
    relations = observed_relations(world, visible)
    changes = []
    for obj_id in sorted(visible):
        obj = world.objects[obj_id]
        old = beliefs.get(obj_id)
        fresh = Belief(
            obj.obj_type,
            [relation for relation in relations if relation[0] == obj_id],
            observed_attrs(world, obj_id),
            obj.cell,
            world.tick,
            "seen",
        )
        if old is None:
            changes.append((obj_id, "added"))
        else:
            fresh.scanned_attrs = old.scanned_attrs
            content_changed = (
                old.relations != fresh.relations
                or old.visual_attrs != fresh.visual_attrs
                or old.believed_cell != fresh.believed_cell
                or old.missing
            )
            if content_changed:
                changes.append((obj_id, "updated"))
        beliefs[obj_id] = fresh
    for obj_id in sorted(beliefs):
        belief = beliefs[obj_id]
        if obj_id in visible or belief.missing:
            continue
        if not cell_visible(world.walls, viewpoint, belief.believed_cell):
            continue
        if _believed_in_closed_container(belief, world):
            continue
        belief.missing = True
        changes.append((obj_id, "missing"))
    return changes


def merge_told(beliefs: dict[str, Belief], told: dict[str, Belief], sender: str) -> list[str]:
    """Merges a teammate's beliefs. The newer `last_seen` wins.

    Returns:
        Ids of the beliefs that were replaced, in id order.
    """
    merged = []
    for obj_id in sorted(told):
        incoming = told[obj_id]
        own = beliefs.get(obj_id)
        if own is not None and own.last_seen >= incoming.last_seen:
            continue
        beliefs[obj_id] = Belief(
            incoming.obj_type,
            list(incoming.relations),
            dict(incoming.visual_attrs),
            incoming.believed_cell,
            incoming.last_seen,
            "told",
            told_by=sender,
            missing=incoming.missing,
            scanned_attrs=dict(incoming.scanned_attrs),
        )
        merged.append(obj_id)
    return merged


def serialize_beliefs(
    beliefs: dict[str, Belief], tick: int, agent_cell: Cell, planner: PathPlanner
) -> str:
    """Belief text for the prompt: seen now, remembered, and told, with staleness and step counts.

    Distances are BFS step counts from `planner`, never coordinates (hard rule 8).
    """
    seen_lines = []
    remembered_lines = []
    told_lines = []
    for obj_id in sorted(beliefs):
        belief = beliefs[obj_id]
        steps = planner.steps(agent_cell, belief.believed_cell)
        distance = "unreachable" if steps is None else f"{steps} steps"
        age = tick - belief.last_seen
        if belief.missing:
            remembered_lines.append(
                f"  {obj_id} - MISSING (not at last known spot, last seen {age} ticks ago)"
            )
            continue
        relation_text = ", ".join(f"{rel} {target}" for _, rel, target in belief.relations)
        attrs = {**belief.visual_attrs, **belief.scanned_attrs}
        attr_text = ", ".join(f"{key}={attrs[key]}" for key in sorted(attrs))
        description = f"{obj_id} {relation_text}" + (f" ({attr_text})" if attr_text else "")
        if belief.source == "seen" and belief.last_seen == tick:
            seen_lines.append(f"  {description} - {distance}")
        elif belief.source == "told":
            told_lines.append(f"  {description} (from {belief.told_by}, {age} ticks ago)")
        else:
            remembered_lines.append(f"  {description} - last seen {age} ticks ago - {distance}")
    sections = [
        "SEEN NOW:",
        *(seen_lines or ["  nothing"]),
        "REMEMBERED (not in view):",
        *(remembered_lines or ["  nothing"]),
        "TOLD BY TEAMMATES:",
        *(told_lines or ["  nothing"]),
    ]
    return "\n".join(sections)


def _believed_in_closed_container(belief: Belief, world: World) -> bool:
    for _, relation, parent_id in belief.relations:
        if relation == "in" and not world.objects[parent_id].is_open:
            return True
    return False
