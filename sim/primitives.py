"""Action primitives as multi-tick skills.

Each tool call is checked twice. When the policy returns it, it is checked against the agent's
beliefs; naming an object never observed is a hallucination. When it applies, it is checked
against ground truth; a world that no longer matches the belief is a stale belief. The two are
logged separately because only the second should worsen as communication degrades.
"""

import math
import random

from sim.agent import ActionResult, AgentState, Skill, ToolCall
from sim.channel import Network
from sim.config import PRIMITIVE_TICKS
from sim.pathing import PathPlanner, travel_ticks
from sim.world import Cell, World, chebyshev_distance

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
}

# Tool name -> positions of arguments that must name an object in the agent's beliefs.
OBJECT_ARGS: dict[str, tuple[int, ...]] = {
    "move_to": (0,),
    "pick": (0,),
    "place": (0, 1),
    "open": (0,),
    "close": (0,),
    "scan": (0,),
    "handoff": (0,),
}


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
        if call.args[index] not in agent.beliefs:
            return ActionResult(call, "hallucination", f"no known object {call.args[index]}")
    if call.name == "handoff" and call.args[1] not in team_ids:
        return ActionResult(call, "invalid_call", f"no teammate {call.args[1]}")
    if call.name == "wait" and not call.args[0].isdigit():
        return ActionResult(call, "invalid_call", "wait takes a whole number of ticks")
    return None


def start_skill(call: ToolCall, agent: AgentState, planner: PathPlanner) -> Skill:
    """Turns a validated call into a skill with its duration in ticks."""
    if call.name == "move_to":
        target_cell = agent.beliefs[call.args[0]].believed_cell
        steps = planner.steps(agent.cell, target_cell)
        ticks = travel_ticks(0 if steps is None else steps, agent.card.speed)
        return Skill(call, ticks, target_cell=target_cell)
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
) -> ActionResult | None:
    """Runs one tick of the agent's current skill. Returns its result when it finishes."""
    skill = agent.skill
    if skill.call.name == "move_to":
        for _ in range(agent.card.speed):
            next_cell = planner.next_cell(agent.cell, skill.target_cell)
            if next_cell is None:
                break
            agent.heading = _bearing(agent.cell, next_cell)
            agent.cell = next_cell
            for obj_id in agent.held:
                world.objects[obj_id].cell = next_cell
    skill.ticks_left -= 1
    if skill.ticks_left > 0:
        return None
    call = skill.call
    if call.name == "move_to":
        return _finish_move(call, agent, planner, skill)
    if call.name == "pick":
        return _finish_pick(call, agent, world, rng)
    if call.name == "place":
        return _finish_place(call, agent, world, rng)
    if call.name in ("open", "close"):
        return _finish_open_close(call, agent, world, rng)
    if call.name == "scan":
        return _finish_scan(call, agent, world, rng)
    if call.name == "handoff":
        return _finish_handoff(call, agent, agents, world, rng)
    if call.name == "report":
        positions = {agent_id: other.cell for agent_id, other in agents.items()}
        online = {agent_id: other.online for agent_id, other in agents.items()}
        network.broadcast(
            agent.agent_id, "report", {"task_id": call.args[0]}, world.tick, positions, online
        )
        return ActionResult(call, "ok", f"reported {call.args[0]} done")
    return ActionResult(call, "ok", "waited")


def _finish_move(
    call: ToolCall, agent: AgentState, planner: PathPlanner, skill: Skill
) -> ActionResult:
    if planner.steps(agent.cell, skill.target_cell) != 0:
        return ActionResult(call, "skill_failure", f"no path to {call.args[0]}")
    if skill.target_cell != agent.cell:
        agent.heading = _bearing(agent.cell, skill.target_cell)
    return ActionResult(call, "ok", f"arrived next to {call.args[0]}")


def _finish_pick(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    obj = world.objects[call.args[0]]
    if obj.is_furniture:
        return ActionResult(call, "invalid_call", f"{obj.obj_id} is furniture")
    if obj.held_by == agent.agent_id:
        return ActionResult(call, "invalid_call", f"already holding {obj.obj_id}")
    if obj.held_by is not None:
        return ActionResult(call, "stale_belief", f"{obj.obj_id} is held by {obj.held_by}")
    if chebyshev_distance(agent.cell, obj.cell) > 1:
        return ActionResult(call, "stale_belief", f"pick failed: {obj.obj_id} is not here")
    if obj.parent_relation == "in" and not world.objects[obj.parent_id].is_open:
        return ActionResult(call, "stale_belief", f"{obj.parent_id} is closed")
    if agent.hands_free(world) < obj.hands_required:
        return ActionResult(
            call, "invalid_call", f"{obj.obj_id} needs {obj.hands_required} free hands"
        )
    if rng.random() >= agent.card.success_prob("pick"):
        return ActionResult(call, "skill_failure", f"pick failed: {obj.obj_id} slipped")
    world.hold(obj.obj_id, agent.agent_id, agent.cell)
    agent.held.append(obj.obj_id)
    return ActionResult(call, "ok", f"picked up {obj.obj_id}")


def _finish_place(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    obj_id, target_id = call.args
    target = world.objects[target_id]
    if obj_id not in agent.held:
        return ActionResult(call, "invalid_call", f"not holding {obj_id}")
    if not target.is_surface and not target.is_container:
        return ActionResult(call, "invalid_call", f"{target_id} is not a surface or container")
    if chebyshev_distance(agent.cell, target.cell) > 1:
        return ActionResult(call, "stale_belief", f"place failed: {target_id} is not here")
    if target.is_container and not target.is_open:
        return ActionResult(call, "stale_belief", f"{target_id} is closed")
    agent.held.remove(obj_id)
    if rng.random() >= agent.card.success_prob("place"):
        landed = world.drop(obj_id, agent.cell)
        return ActionResult(call, "skill_failure", f"place failed: {obj_id} dropped, now {landed}")
    world.attach(obj_id, target_id)
    relation = world.objects[obj_id].parent_relation
    return ActionResult(call, "ok", f"placed {obj_id} {relation} {target_id}")


def _finish_open_close(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    container = world.objects[call.args[0]]
    opening = call.name == "open"
    if not container.is_container:
        return ActionResult(call, "invalid_call", f"{container.obj_id} cannot be opened")
    if chebyshev_distance(agent.cell, container.cell) > 1:
        return ActionResult(call, "stale_belief", f"{container.obj_id} is not here")
    if container.is_open == opening:
        state = "open" if opening else "closed"
        return ActionResult(call, "stale_belief", f"{container.obj_id} is already {state}")
    if rng.random() >= agent.card.success_prob(call.name):
        return ActionResult(call, "skill_failure", f"{call.name} {container.obj_id} failed")
    container.is_open = opening
    verb = "opened" if opening else "closed"
    return ActionResult(call, "ok", f"{verb} {container.obj_id}")


def _finish_scan(
    call: ToolCall, agent: AgentState, world: World, rng: random.Random
) -> ActionResult:
    obj = world.objects[call.args[0]]
    if not agent.card.can_scan:
        return ActionResult(call, "invalid_call", "this agent has no scanner")
    if chebyshev_distance(agent.cell, obj.cell) > 1:
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


def _bearing(start: Cell, end: Cell) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360
