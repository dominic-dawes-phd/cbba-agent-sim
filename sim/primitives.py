"""Action primitives as multi-tick skills.

Each tool call is checked twice. When the policy returns it, it is checked against the agent's
beliefs; naming an object never observed is a hallucination. When it applies, it is checked
against ground truth; a world that no longer matches the belief is a stale belief. The two are
logged separately because only the second should worsen as communication degrades.

Drops: with probability `drop_prob` a held object slips once per pick, place, or move. Drops are a
time penalty. A non-fragile object lands on the agent's tile and the same gripper picks it up after
a short pause. A breakable one shatters there and triggers a reflex: clean up, walk back to the task
area, and carry on with the same task.
"""

import random

from sim.agent import ActionResult, AgentState, Skill, ToolCall
from sim.channel import Network
from sim.config import (
    CLEAN_UP_TOOL,
    EXPLORING_TOOLS,
    PRIMITIVE_TICKS,
    REFLEX_TOOLS,
    RETURN_ITEM_TOOL,
    RETURN_TO_TASK_TOOL,
    SHARD_DISPOSE_TICKS,
    SHARD_PICKUP_TICKS,
)
from sim.motion import drop_held
from sim.pathing import PathPlanner
from sim.triggers import TRASHCAN_TYPE
from sim.world import (
    BAG_TYPE,
    Cell,
    World,
    WorldObject,
    chebyshev_distance,
    split_slot,
    within_reach,
)

# Tool name -> number of arguments.
TOOL_ARITY: dict[str, int] = {
    "move_to": 1,
    "pick": 1,
    "place": 2,
    "open": 1,
    "close": 1,
    "scan": 1,
    "handoff": 2,
    "report": 1,
    "wait": 1,
    "start": 1,
    "lock": 1,
    "unlock": 1,
    "tie_bag": 1,
    "claim": 1,
    "release": 1,
    "mark_done": 1,
    "give_up": 1,
    "talk_to_team": 0,
    "explore": 0,
    "search": 0,
}

# Tools that broadcast a task id to the team, and the message kind they send. `claim` and
# `release` are the simplest allocation tools: an announcement with no consensus behind it.
ANNOUNCE_KINDS: dict[str, str] = {"report": "report", "claim": "claim", "release": "release"}

# Ledger tools (auction conditions): the agent records a task done or gives it up in its own
# ledger copy. Unlike the announcements above they send nothing; ledger gossip carries the change.
LEDGER_TOOLS: dict[str, str] = {"mark_done": "marked done", "give_up": "gave up"}

# Tool name -> positions of arguments that must name an object in the agent's beliefs.
OBJECT_ARGS: dict[str, tuple[int, ...]] = {
    "move_to": (0,),
    "pick": (0,),
    "place": (0, 1),
    "open": (0,),
    "close": (0,),
    "scan": (0,),
    "handoff": (0,),
    "start": (0,),
    "lock": (0,),
    "unlock": (0,),
    "tie_bag": (0,),
}

# Tool name -> positions of arguments that may name one slot of furniture, like "cabinet_1[2]".
SLOT_ARGS: dict[str, tuple[int, ...]] = {"move_to": (0,), "place": (1,), "lock": (0,)}

# Tools that another agent's reserved workspace blocks, and the argument they work on.
RESERVABLE_ARG: dict[str, int] = {"pick": 0, "open": 0, "close": 0, "start": 0, "tie_bag": 0}


def check_against_beliefs(
    call: ToolCall, agent: AgentState, team_ids: list[str]
) -> ActionResult | None:
    """Call-time validation. Returns the failure, or None if the call may be scheduled."""
    if call.name not in TOOL_ARITY:
        return ActionResult(call, "invalid_call", f"unknown tool {call.name}")
    if len(call.args) != TOOL_ARITY[call.name]:
        return ActionResult(
            call, "invalid_call", f"{call.name} takes {TOOL_ARITY[call.name]} arguments"
        )
    for index in OBJECT_ARGS.get(call.name, ()):
        name = call.args[index]
        base, slot = split_slot(name) if index in SLOT_ARGS.get(call.name, ()) else (name, None)
        if base not in agent.beliefs:
            return ActionResult(call, "hallucination", f"no known object {base}")
        if slot is not None and slot >= len(agent.beliefs[base].footprint):
            return ActionResult(call, "invalid_call", f"{base} has no slot {slot}")
    if call.name == "handoff" and call.args[1] not in team_ids:
        return ActionResult(call, "invalid_call", f"no teammate {call.args[1]}")
    if call.name == "wait" and not call.args[0].isdigit():
        return ActionResult(call, "invalid_call", "wait takes a whole number of ticks")
    if call.name in EXPLORING_TOOLS and agent.held:
        return ActionResult(call, "invalid_call", f"put down what you hold before you {call.name}")
    return None


def start_skill(
    call: ToolCall, agent: AgentState, planner: PathPlanner, rng: random.Random, drop_prob: float
) -> Skill:
    """Turns a validated call into a skill. A move while holding may get a drop step.

    `explore` walks toward the nearest floor nobody is known to have searched, `search` toward the
    nearest floor this agent has not seen itself (targets set every tick by
    `sim.exploration.retarget_explorers`), until no such floor is left. Like `wait` both are
    interruptible: seeing a new object, finding a task, or a message ends them at a decision.
    """
    if call.name in EXPLORING_TOOLS:
        return Skill(call, 0, interruptible=True)
    if call.name == "move_to":
        base, slot = split_slot(call.args[0])
        belief = agent.beliefs[base]
        targets = belief.cells() if slot is None else (belief.footprint[slot],)
        drop_at_step = None
        if agent.held and drop_prob > 0 and rng.random() < drop_prob:
            steps = planner.steps_to_any(agent.cell, targets) or 0
            drop_at_step = rng.randint(1, max(1, steps))
        return Skill(call, 0, targets=targets, drop_at_step=drop_at_step)
    if call.name == "wait":
        return Skill(call, max(1, int(call.args[0])), interruptible=True)
    return Skill(call, PRIMITIVE_TICKS[call.name])


def advance_skill(
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    network: Network,
    rng: random.Random,
    drop_prob: float,
) -> ActionResult | None:
    """Runs one tick of the agent's current skill. Returns its result when it finishes.

    `move_to`, `explore`, and `search` are not handled here: every agent's movement runs together in
    `sim.motion.advance_moves`.
    """
    skill = agent.skill
    skill.ticks_left -= 1
    if skill.ticks_left > 0:
        return None
    call = skill.call
    for index in OBJECT_ARGS.get(call.name, ()):
        # Believed objects can stop existing (trash bagged by a teammate): a stale belief.
        if split_slot(call.args[index])[0] not in world.objects:
            return ActionResult(call, "stale_belief", f"{call.args[index]} no longer exists")
    if call.name in RESERVABLE_ARG:
        blocked = _check_reserved(call, agent, world)
        if blocked is not None:
            return blocked
    if call.name == "start":
        return _finish_start(call, agent, world, rng)
    if call.name == "lock":
        return _finish_lock(call, agent, world, rng)
    if call.name == "unlock":
        return _finish_unlock(call, agent, world)
    if call.name == "tie_bag":
        return _finish_tie_bag(call, agent, world, rng)
    if call.name == "pick":
        return _finish_pick(call, agent, world, rng, drop_prob)
    if call.name == "place":
        return _finish_place(call, agent, world, rng, drop_prob)
    if call.name in ("open", "close"):
        return _finish_open_close(call, agent, world, rng)
    if call.name == "scan":
        return _finish_scan(call, agent, world, rng)
    if call.name == "handoff":
        return _finish_handoff(call, agent, agents, world, rng)
    if call.name in ANNOUNCE_KINDS:
        positions = {agent_id: other.cell for agent_id, other in agents.items()}
        online = {agent_id: other.online for agent_id, other in agents.items()}
        network.broadcast(
            agent.agent_id,
            ANNOUNCE_KINDS[call.name],
            {"task_id": call.args[0]},
            world.tick,
            positions,
            online,
        )
        return ActionResult(call, "ok", f"announced {call.name} {call.args[0]}")
    if call.name == "talk_to_team":
        # C1's allocator reads this from the action history and asks for a dialogue round.
        return ActionResult(call, "ok", "asked the team to talk")
    if call.name in LEDGER_TOOLS:
        # The agent's allocator reads this from its action history and updates the ledger, which
        # gossip then spreads; no message is sent here.
        return ActionResult(call, "ok", f"{LEDGER_TOOLS[call.name]} {call.args[0]} in the ledger")
    return ActionResult(call, "ok", "waited")


def nearest_trashcan(cell: Cell, world: World, planner: PathPlanner) -> str | None:
    """The trashcan with the fewest static steps from a cell; ties break by id.

    A deliberate cheat: it reads ground truth, not the agent's beliefs, so a fragile drop can never
    turn into a search, however obscure the can's corner of a generated house. The choice reaches
    no belief, so the agent learns nothing from it.
    """
    options = []
    for can_id in sorted(world.objects):
        can = world.objects[can_id]
        if can.obj_type == TRASHCAN_TYPE:
            steps = planner.steps_to_any(cell, can.cells())
            if steps is not None:
                options.append((steps, can_id))
    return min(options)[1] if options else None


def start_clean_up(agent: AgentState, world: World, planner: PathPlanner) -> bool:
    """Starts the reflex after a shatter: pick up the shards and head for the nearest trashcan.

    It runs below the executor in every condition, with no LLM call and no decision.

    Returns:
        Whether the reflex started. With no reachable trashcan the shards stay on the floor.
    """
    obj_id = agent.shards
    agent.shards = None
    can_id = nearest_trashcan(agent.cell, world, planner)
    if can_id is None:
        return False
    agent.task_area = agent.cell
    world.hold(obj_id, agent.agent_id, agent.cell)
    agent.held.append(obj_id)
    agent.queue.clear()
    agent.skill = Skill(
        ToolCall(CLEAN_UP_TOOL, (obj_id, can_id)),
        0,
        started=world.tick,
        targets=world.objects[can_id].cells(),
    )
    agent.thinking_until = max(agent.thinking_until, world.tick + 1 + SHARD_PICKUP_TICKS)
    return True


def finish_clean_up(agent: AgentState, world: World) -> ActionResult:
    """Disposes of the shards in the trashcan the reflex reached, whatever its lid or fill.

    They count toward the can's fill gauge, so the can can still call for a `take_out_trash`.
    """
    call = agent.skill.call
    obj_id, can_id = call.args
    agent.held.remove(obj_id)
    can = world.objects[can_id]
    nearest = min((chebyshev_distance(agent.cell, cell), cell) for cell in can.cells())[1]
    world.attach(obj_id, can_id, nearest)
    agent.thinking_until = max(agent.thinking_until, world.tick + 1 + SHARD_DISPOSE_TICKS)
    return ActionResult(call, "ok", f"{obj_id} shattered; cleaned it up into {can_id}")


def return_to_task_area(agent: AgentState, world: World, call: ToolCall) -> None:
    """Sends the agent back along the static path to the tile where it dropped the object.

    The agent decides only once it is back, so the next step proceeds from its task area.
    """
    agent.skill = Skill(
        ToolCall(RETURN_TO_TASK_TOOL, call.args),
        0,
        started=world.tick,
        targets=(agent.task_area,),
    )
    agent.task_area = None


def finish_return(agent: AgentState) -> ActionResult:
    """The reflex is over: the agent is back at its task area and gets a decision point."""
    call = agent.skill.call
    obj_id, can_id = call.args
    text = f"{obj_id} shattered; cleaned it up into {can_id} and returned to the task area"
    return ActionResult(call, "ok", text)


def start_return_item(agent: AgentState, world: World, obj_id: str) -> bool:
    """Starts the reflex that carries a held dish back to where it was picked up.

    It runs when a teammate starts the dishwasher this dish was headed for (the task split), with
    no LLM call and no decision: whatever the agent had queued or was deliberating is dropped.

    Returns:
        Whether the reflex started: not when the agent is already in a reflex or the dish has no
        recorded source.
    """
    source = agent.picked_from.get(obj_id)
    if source is None or (agent.skill is not None and agent.skill.call.name in REFLEX_TOOLS):
        return False
    furniture_id = source[0]
    agent.queue.clear()
    agent.pick_places.clear()
    agent.skill = Skill(
        ToolCall(RETURN_ITEM_TOOL, (obj_id, furniture_id)),
        0,
        started=world.tick,
        targets=world.objects[furniture_id].cells(),
    )
    agent.thinking_until = min(agent.thinking_until, world.tick)
    return True


def finish_return_item(agent: AgentState, world: World) -> ActionResult:
    """Puts the dish back in the slot it was picked up from; the agent then decides."""
    call = agent.skill.call
    obj_id, furniture_id = call.args
    slot_cell = agent.picked_from.pop(obj_id)[1]
    agent.held.remove(obj_id)
    world.attach(obj_id, furniture_id, slot_cell)
    agent.thinking_until = max(agent.thinking_until, world.tick + 1 + PRIMITIVE_TICKS["place"])
    return ActionResult(call, "ok", f"its task split; put {obj_id} back on {furniture_id}")


def _finish_pick(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random, drop_prob: float
) -> ActionResult:
    obj = world.objects[call.args[0]]
    if obj.is_furniture:
        return ActionResult(call, "invalid_call", f"{obj.obj_id} is furniture")
    if obj.held_by == agent.agent_id:
        return ActionResult(call, "invalid_call", f"already holding {obj.obj_id}")
    if obj.held_by is not None:
        return ActionResult(call, "stale_belief", f"{obj.obj_id} is held by {obj.held_by}")
    if obj.obj_id in agent.pick_places:
        # Hard rule 4's re-check: the pick was aimed where the agent believed the item was.
        expected = agent.pick_places.pop(obj.obj_id)
        actual = (obj.parent_relation, obj.parent_id) if obj.parent_id is not None else None
        if actual != expected:
            return ActionResult(call, "stale_belief", moved_text(obj.obj_id, expected))
    if not within_reach(agent.cell, obj):
        return ActionResult(call, "stale_belief", f"pick failed: {obj.obj_id} is not here")
    if obj.parent_relation == "in" and not world.objects[obj.parent_id].is_open:
        return ActionResult(call, "stale_belief", f"{obj.parent_id} is closed")
    if agent.hands_free(world) < obj.hands_required:
        return ActionResult(
            call, "invalid_call", f"{obj.obj_id} needs {obj.hands_required} free hands"
        )
    if rng.random() >= agent.card.success_prob("pick"):
        return ActionResult(call, "skill_failure", f"pick failed: missed {obj.obj_id}")
    if obj.parent_id is not None:
        agent.picked_from[obj.obj_id] = (obj.parent_id, obj.cell)
    world.hold(obj.obj_id, agent.agent_id, agent.cell)
    agent.held.append(obj.obj_id)
    if drop_prob > 0 and rng.random() < drop_prob:
        shattered = drop_held(call, agent, world, obj.obj_id, "right after pickup")
        if shattered is not None:
            return shattered
    return ActionResult(call, "ok", f"picked up {obj.obj_id}")


def moved_text(obj_id: str, expected: tuple[str, str] | None) -> str:
    """The failure text for a pick whose item is no longer where the agent believed it was."""
    if expected is None:
        return f"{obj_id} has moved"
    relation, parent = expected
    return f"{obj_id} is no longer {relation} {parent}"


def _finish_place(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random, drop_prob: float
) -> ActionResult:
    """Places into the closest reachable slot with room, or into the one slot named."""
    obj_id, target_name = call.args
    base, slot_index = split_slot(target_name)
    target = world.objects[base]
    if obj_id not in agent.held:
        return ActionResult(call, "invalid_call", f"not holding {obj_id}")
    if not target.is_surface and not target.is_container:
        return ActionResult(call, "invalid_call", f"{base} is not a surface or container")
    if target.is_container and not target.is_open:
        return ActionResult(call, "stale_belief", f"{base} is closed")
    slots = list(target.cells()) if slot_index is None else [target.cells()[slot_index]]
    # Closest first; ties keep slot order.
    reachable = [
        slot
        for distance, _, slot in sorted(
            (chebyshev_distance(agent.cell, slot), index, slot) for index, slot in enumerate(slots)
        )
        if distance <= 1
    ]
    if not reachable:
        return ActionResult(call, "stale_belief", f"place failed: {target_name} is not here")
    chosen = None
    for slot in reachable:
        holder = world.reserved_by_other(slot, agent.agent_id)
        if holder is not None:
            return ActionResult(
                call, "stale_belief", f"{_slot_name(target, slot)} is reserved by {holder}"
            )
        if world.slot_has_room(base, slot):
            chosen = slot
            break
    if chosen is None:
        full = ", ".join(_slot_name(target, slot) for slot in reachable)
        return ActionResult(call, "stale_belief", f"{full} full")
    if drop_prob > 0 and rng.random() < drop_prob:
        shattered = drop_held(call, agent, world, obj_id, "while placing")
        if shattered is not None:
            return shattered
    if rng.random() >= agent.card.success_prob("place"):
        return ActionResult(call, "skill_failure", f"place failed: fumbled, still holding {obj_id}")
    agent.held.remove(obj_id)
    agent.picked_from.pop(obj_id, None)
    world.attach(obj_id, base, chosen)
    relation = world.objects[obj_id].parent_relation
    return ActionResult(call, "ok", f"placed {obj_id} {relation} {_slot_name(target, chosen)}")


def _finish_open_close(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    container = world.objects[call.args[0]]
    opening = call.name == "open"
    if not container.is_container:
        return ActionResult(call, "invalid_call", f"{container.obj_id} cannot be opened")
    if not within_reach(agent.cell, container):
        return ActionResult(call, "stale_belief", f"{container.obj_id} is not here")
    if container.is_open == opening:
        state = "open" if opening else "closed"
        return ActionResult(call, "stale_belief", f"{container.obj_id} is already {state}")
    if opening and container.cycle == "running":
        return ActionResult(call, "stale_belief", f"{container.obj_id} is mid-cycle")
    if rng.random() >= agent.card.success_prob(call.name):
        return ActionResult(call, "skill_failure", f"{call.name} {container.obj_id} failed")
    container.is_open = opening
    if opening and container.cycle == "done":
        container.cycle = "idle"
    verb = "opened" if opening else "closed"
    return ActionResult(call, "ok", f"{verb} {container.obj_id}")


def _finish_scan(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    obj = world.objects[call.args[0]]
    if not agent.card.can_scan:
        return ActionResult(call, "invalid_call", "this agent has no scanner")
    if not within_reach(agent.cell, obj):
        return ActionResult(call, "stale_belief", f"{obj.obj_id} is not here")
    if rng.random() >= agent.card.success_prob("scan"):
        return ActionResult(call, "skill_failure", f"scan of {obj.obj_id} failed")
    agent.beliefs[obj.obj_id].scanned_attrs = dict(obj.hidden_attrs)
    found = ", ".join(f"{key}={obj.hidden_attrs[key]}" for key in sorted(obj.hidden_attrs))
    return ActionResult(call, "ok", f"scanned {obj.obj_id}: {found or 'nothing notable'}")


def _finish_handoff(
    call: ToolCall,
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    rng: random.Random,
) -> ActionResult:
    obj_id, receiver_id = call.args
    receiver = agents[receiver_id]
    if obj_id not in agent.held:
        return ActionResult(call, "invalid_call", f"not holding {obj_id}")
    if not receiver.online or chebyshev_distance(agent.cell, receiver.cell) > 1:
        return ActionResult(call, "stale_belief", f"{receiver_id} is not here")
    if receiver.hands_free(world) < world.objects[obj_id].hands_required:
        return ActionResult(call, "stale_belief", f"{receiver_id} has no free hands")
    if rng.random() >= agent.card.success_prob("handoff"):
        return ActionResult(call, "skill_failure", f"handoff of {obj_id} failed")
    agent.held.remove(obj_id)
    receiver.held.append(obj_id)
    world.hold(obj_id, receiver_id, receiver.cell)
    return ActionResult(call, "ok", f"handed {obj_id} to {receiver_id}")


def _check_reserved(call: ToolCall, agent: AgentState, world: World) -> ActionResult | None:
    """Fails the call if its target sits in a workspace another holder reserved."""
    target = world.objects[call.args[RESERVABLE_ARG[call.name]]]
    for cell in target.cells():
        holder = world.reserved_by_other(cell, agent.agent_id)
        if holder is not None:
            return ActionResult(call, "stale_belief", f"{target.obj_id} is reserved by {holder}")
    return None


def _finish_start(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    appliance = world.objects[call.args[0]]
    if appliance.washing_time == 0:
        return ActionResult(call, "invalid_call", f"{appliance.obj_id} cannot be started")
    if not within_reach(agent.cell, appliance):
        return ActionResult(call, "stale_belief", f"{appliance.obj_id} is not here")
    if appliance.is_open or appliance.cycle == "running":
        return ActionResult(call, "stale_belief", f"{appliance.obj_id} is open or already running")
    if rng.random() >= agent.card.success_prob("start"):
        return ActionResult(call, "skill_failure", f"start {appliance.obj_id} failed")
    world.start_cycle(appliance.obj_id, agent.agent_id)
    return ActionResult(call, "ok", f"started {appliance.obj_id} ({appliance.washing_time} ticks)")


def _finish_lock(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    """Reserves the workspace: the target's cells within reach plus the floor around them."""
    base, slot_index = split_slot(call.args[0])
    target = world.objects[base]
    if not target.is_furniture:
        return ActionResult(call, "invalid_call", f"{base} is not furniture")
    slots = target.cells() if slot_index is None else (target.cells()[slot_index],)
    reached = [slot for slot in slots if chebyshev_distance(agent.cell, slot) <= 1]
    if not reached:
        return ActionResult(call, "stale_belief", f"{call.args[0]} is not here")
    blocked = world.blocked_cells()
    floor = {
        (slot[0] + dx, slot[1] + dy)
        for slot in reached
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (slot[0] + dx, slot[1] + dy) in world.rooms
        and (slot[0] + dx, slot[1] + dy) not in blocked
    }
    workspace = sorted(set(reached) | floor)
    for cell in workspace:
        holder = world.reserved_by_other(cell, agent.agent_id)
        if holder is not None:
            return ActionResult(call, "stale_belief", f"{call.args[0]} is reserved by {holder}")
    if rng.random() >= agent.card.success_prob("lock"):
        return ActionResult(call, "skill_failure", f"lock {call.args[0]} failed")
    for cell in workspace:
        world.reservations[cell] = (agent.agent_id, base)
    return ActionResult(call, "ok", f"reserved the workspace at {call.args[0]}")


def _finish_unlock(call: ToolCall, agent: AgentState, world: World) -> ActionResult:
    target_id = call.args[0]
    if (agent.agent_id, target_id) not in world.reservations.values():
        return ActionResult(call, "invalid_call", f"you hold no reservation on {target_id}")
    world.release_reservations(agent.agent_id, target_id)
    return ActionResult(call, "ok", f"released the workspace at {target_id}")


def _finish_tie_bag(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    can = world.objects[call.args[0]]
    if not within_reach(agent.cell, can):
        return ActionResult(call, "stale_belief", f"{can.obj_id} is not here")
    if not can.is_open:
        return ActionResult(call, "stale_belief", f"{can.obj_id} is closed")
    bag_ids = sorted(
        obj.obj_id
        for obj in world.objects.values()
        if obj.parent_id == can.obj_id and obj.obj_type == BAG_TYPE
    )
    if not bag_ids:
        return ActionResult(call, "stale_belief", f"{can.obj_id} has no bag")
    bag = world.objects[bag_ids[0]]
    if bag.visual_attrs.get("tied") == "yes":
        return ActionResult(call, "stale_belief", f"{bag.obj_id} is already tied")
    if rng.random() >= agent.card.success_prob("tie_bag"):
        return ActionResult(call, "skill_failure", f"tie of {bag.obj_id} failed")
    # The trash is now inside the tied bag. The bag is one carriable object.
    for obj_id in [obj.obj_id for obj in world.objects.values() if obj.parent_id == can.obj_id]:
        if obj_id != bag.obj_id:
            del world.objects[obj_id]
    bag.visual_attrs["tied"] = "yes"
    return ActionResult(call, "ok", f"tied {bag.obj_id}")


def _slot_name(furniture: WorldObject, slot: Cell) -> str:
    """Slot name for text: "cabinet_1[2]" for multi-cell furniture, else the plain id."""
    if len(furniture.cells()) == 1:
        return furniture.obj_id
    return f"{furniture.obj_id}[{furniture.cells().index(slot)}]"
