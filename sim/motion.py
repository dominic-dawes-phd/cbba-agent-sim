"""Simultaneous movement with a seeded 4-way-stop rule.

Every tick, all agents in a moving skill (`MOVING_TOOLS`) move together, in rounds: a speed-2 agent
takes part in two rounds, a speed-1 agent in one. Each round:

1. Every mover that is not yet next to its target proposes one next cell: the next cell on its
   static shortest path (a cached BFS distance field over walls and furniture). If that cell is
   blocked by a body or another holder's reservation, it replans with a fresh BFS that avoids
   them and takes the first cell of the detour. This happens before every cell, so it reacts to
   obstacles where they are now (local replanning); it does not predict others' moves.
2. Cells occupied at the start of the round are off-limits, so nobody follows into a cell that is
   being vacated and two agents can never swap.
3. If several movers propose the same cell, a seeded RNG picks one winner; the others wait this
   round and count a lost conflict. There is no id-order priority.

Collisions are impossible by construction (the runtime shield). Distances for bids and prompts are
the static ones and never see bodies or reservations (hard rule 8).
"""

import math
import random

from sim.agent import ActionResult, AgentState, ToolCall
from sim.config import (
    BACKOFF_AFTER_STUCK_TICKS,
    BACKOFF_SIT_OUT_PROB,
    BLOCKED_PATIENCE_TICKS,
    BREAKABLE_TYPES,
    DROP_RECOVERY_TICKS,
    EXPLORING_TOOLS,
    MOVING_TOOLS,
    REFLEX_TOOLS,
)
from sim.pathing import PathPlanner
from sim.world import Cell, World, chebyshev_distance

# Directions an agent steps in, and an idle agent yields in (4-connected).
STEP_OFFSETS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def advance_moves(
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    rng: random.Random,
    finished: frozenset[str] = frozenset(),
) -> dict[str, ActionResult]:
    """Runs one tick of movement for every agent in a moving skill (`MOVING_TOOLS`).

    Args:
        agents: Every agent, by id.
        world: Ground truth; held objects move with their agent.
        planner: The static distance function (hard rule 8), plus obstacle-aware detours.
        rng: The seeded movement stream that breaks same-cell ties.
        finished: Agents whose skill already ended this tick (an explorer with no frontier left);
            they do not move.

    Returns:
        Results of the moves that ended this tick (arrived, dropped, no path, or blocked), by agent.
    """
    movers = [
        agent_id
        for agent_id in sorted(agents)
        if agents[agent_id].online
        and world.tick >= agents[agent_id].thinking_until
        and agents[agent_id].skill is not None
        and agents[agent_id].skill.call.name in MOVING_TOOLS
        and agent_id not in finished
    ]
    results: dict[str, ActionResult] = {}
    for agent_id in movers:
        agent = agents[agent_id]
        if planner.steps_to_any(agent.cell, agent.skill.targets) is None:
            results[agent_id] = ActionResult(
                agent.skill.call, "skill_failure", f"no path to {agent.skill.call.args[0]}"
            )
    for agent_id in movers:
        skill = agents[agent_id].skill
        if skill.best_steps is None:
            skill.best_steps = planner.steps_to_any(agents[agent_id].cell, skill.targets)
    arrived: set[str] = set()
    for round_index in range(max((agents[agent_id].card.speed for agent_id in movers), default=0)):
        occupied = {agent.cell for agent in agents.values()}
        proposals: dict[Cell, list[str]] = {}
        for agent_id in movers:
            agent = agents[agent_id]
            if agent_id in results or agent_id in arrived or round_index >= agent.card.speed:
                continue
            if world.tick < agent.thinking_until:  # just dropped something this tick
                continue
            # Randomized backoff: two movers detouring in step can undo each other's moves forever
            # without either being blocked at the end of a tick. A coin flip breaks the symmetry.
            if (
                agent.skill.stuck_ticks >= BACKOFF_AFTER_STUCK_TICKS
                and rng.random() < BACKOFF_SIT_OUT_PROB
            ):
                continue
            targets = agent.skill.targets
            # An agent's own cell is never in its avoid set, so the static check is exact here.
            if planner.steps_to_any(agent.cell, targets) == 0:
                arrived.add(agent_id)
                continue
            # Follow the cached static path; search around obstacles only when the next cell is
            # blocked (local replanning, like a robot's local planner). A detour is kept until it
            # gets closer: a detour that first moves away would otherwise step straight back onto
            # the static path and oscillate forever.
            avoid = _avoid_for(agent, occupied, world)
            next_cell = planner.next_cell(agent.cell, targets)
            if next_cell is None or next_cell in avoid or agent.skill.detouring:
                agent.skill.detouring = True
                next_cell = planner.next_cell(agent.cell, targets, avoid)
            if next_cell is not None:
                proposals.setdefault(next_cell, []).append(agent_id)
        for cell in sorted(proposals):
            contenders = proposals[cell]
            winner = contenders[0] if len(contenders) == 1 else rng.choice(contenders)
            for loser in contenders:
                if loser != winner:
                    agents[loser].move_conflicts += 1
            agent = agents[winner]
            agent.heading = bearing(agent.cell, cell)
            agent.cell = cell
            for obj_id in agent.held:
                world.objects[obj_id].cell = cell
            agent.skill.steps_taken += 1
            if agent.skill.steps_taken == agent.skill.drop_at_step:
                shattered = drop_held(agent.skill.call, agent, world, agent.held[0], "while moving")
                if shattered is not None:
                    results[winner] = shattered
    for agent_id in movers:
        if agent_id not in results:
            result = _settle(agents[agent_id], agents, world, planner, rng)
            if result is not None:
                results[agent_id] = result
    return results


def _avoid_for(agent: AgentState, occupied: set[Cell], world: World) -> frozenset[Cell]:
    """Cells this agent may not step into: other bodies and other holders' reservations."""
    reserved = {
        cell for cell in world.reservations if world.reserved_by_other(cell, agent.agent_id)
    }
    return frozenset((occupied | reserved) - {agent.cell})


def _settle(
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    rng: random.Random,
) -> ActionResult | None:
    """End-of-tick check for one mover: arrived, closer than before, or blocked too long.

    A tick counts as stuck unless the agent got closer to its target, so two agents shuffling
    around each other in a doorway (a livelock) still run out of patience. While stuck, an idle
    body in the way is nudged aside, and of two movers blocking each other head-on, a seeded coin
    flip picks one to step aside. An explorer (`explore`, `search`) that reaches its frontier cell
    faces it and goes on:
    perception marks the cell searched next tick, and the explorer is retargeted.
    """
    skill = agent.skill
    call = skill.call
    if planner.steps_to_any(agent.cell, skill.targets) == 0:
        nearest = min((chebyshev_distance(agent.cell, cell), cell) for cell in skill.targets)[1]
        if nearest != agent.cell:
            agent.heading = bearing(agent.cell, nearest)
        if call.name in EXPLORING_TOOLS:
            return None
        return ActionResult(call, "ok", f"arrived next to {call.args[0]}")
    steps = planner.steps_to_any(agent.cell, skill.targets)
    if steps < skill.best_steps:
        skill.best_steps = steps
        skill.stuck_ticks = 0
        skill.detouring = False
        return None
    skill.stuck_ticks += 1
    agent.blocked_ticks += 1
    _nudge_idle_blocker(agent, agents, world, planner, skill.targets)
    blocker = _first_blocker(agent, agents, world, planner)
    if blocker in agents and _head_on(agent, agents[blocker], planner):
        _step_aside(rng.choice((agent, agents[blocker])), agents, world, planner)
    if skill.stuck_ticks < BLOCKED_PATIENCE_TICKS:
        return None
    if call.name in REFLEX_TOOLS:  # a reflex has no one to report a blocked path to
        skill.stuck_ticks = 0
        return None
    reason = f" by {blocker}" if blocker else ""
    destination = call.args[0] if call.args else "the unsearched area"
    return ActionResult(call, "stale_belief", f"path to {destination} blocked{reason}")


def _first_blocker(
    agent: AgentState, agents: dict[str, AgentState], world: World, planner: PathPlanner
) -> str | None:
    """The first body or reservation holder on the agent's static path, if any."""
    bodies = {other.cell: other_id for other_id, other in agents.items() if other is not agent}
    cell = planner.next_cell(agent.cell, agent.skill.targets)
    while cell is not None:
        blocker = bodies.get(cell) or world.reserved_by_other(cell, agent.agent_id)
        if blocker is not None:
            return blocker
        cell = planner.next_cell(cell, agent.skill.targets)
    return None


def _static_path(start: Cell, targets: tuple[Cell, ...], planner: PathPlanner) -> set[Cell]:
    """Cells on the static shortest path from `start` to `targets`."""
    path = set()
    cell = planner.next_cell(start, targets)
    while cell is not None:
        path.add(cell)
        cell = planner.next_cell(cell, targets)
    return path


def _head_on(agent: AgentState, other: AgentState, planner: PathPlanner) -> bool:
    """Whether two movers each stand on the other's static path, so neither can pass."""
    other_moving = (
        other.online and other.skill is not None and other.skill.call.name in MOVING_TOOLS
    )
    return (
        other_moving
        and other.skill.stuck_ticks > 0
        and agent.cell in _static_path(other.cell, other.skill.targets, planner)
    )


def _step_aside(
    agent: AgentState, agents: dict[str, AgentState], world: World, planner: PathPlanner
) -> None:
    """Moves an agent to a free neighbouring cell off every other mover's static path."""
    blocked = world.blocked_cells()
    occupied = {other.cell for other in agents.values()}
    wanted = set()
    for other in agents.values():
        moving = other.skill is not None and other.skill.call.name in MOVING_TOOLS
        if other is not agent and moving:
            wanted |= _static_path(other.cell, other.skill.targets, planner)
    for dx, dy in STEP_OFFSETS:
        spot = (agent.cell[0] + dx, agent.cell[1] + dy)
        free = (
            spot in world.rooms
            and spot not in blocked
            and spot not in occupied
            and spot not in wanted
            and world.reserved_by_other(spot, agent.agent_id) is None
        )
        if free:
            agent.heading = bearing(agent.cell, spot)
            agent.cell = spot
            for obj_id in agent.held:
                world.objects[obj_id].cell = spot
            agent.yields += 1
            return


def _nudge_idle_blocker(
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    targets: tuple[Cell, ...],
) -> None:
    """Moves one idle teammate out of the way of a blocked mover (local yielding).

    An idle body (online, nothing queued, or only waiting) standing on the mover's static path or
    on a cell from which the mover could reach its target steps to a free neighbouring cell off
    those cells. This is body-level collision avoidance shared by every condition, not
    coordination: busy agents never yield, so contested workspaces still block.
    """
    path = _static_path(agent.cell, targets, planner)
    blocked = world.blocked_cells()
    access = {
        (target[0] + dx, target[1] + dy)
        for target in targets
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (target[0] + dx, target[1] + dy) in world.rooms
        and (target[0] + dx, target[1] + dy) not in blocked
    }
    wanted = path | access
    occupied = {other.cell for other in agents.values()}
    for other_id in sorted(agents):
        other = agents[other_id]
        idle = other.skill is None and not other.queue
        waiting = other.skill is not None and other.skill.call.name == "wait"
        if other is agent or not other.online or other.cell not in wanted or not (idle or waiting):
            continue
        for dx, dy in STEP_OFFSETS:
            spot = (other.cell[0] + dx, other.cell[1] + dy)
            free = (
                spot in world.rooms
                and spot not in blocked
                and spot not in occupied
                and spot not in wanted
                and world.reserved_by_other(spot, other_id) is None
            )
            if free:
                other.heading = bearing(other.cell, spot)
                other.cell = spot
                for obj_id in other.held:
                    world.objects[obj_id].cell = spot
                other.yields += 1
                return


def drop_held(
    call: ToolCall, agent: AgentState, world: World, obj_id: str, when: str
) -> ActionResult | None:
    """A held object slips.

    A non-fragile object lands on the agent's own tile and the same gripper picks it up again after
    a pause. It stays in that gripper's custody throughout (no teammate can grab it in those ticks)
    and the skill carries on, so nothing fails and no decision is needed. A breakable one shatters
    on the agent's tile, which fails the skill and sets up the `clean_up` reflex.

    Returns:
        The failure if the object shattered, else None.
    """
    world.drops.append((agent.agent_id, obj_id))
    if world.objects[obj_id].obj_type not in BREAKABLE_TYPES:
        agent.thinking_until = max(agent.thinking_until, world.tick + 1 + DROP_RECOVERY_TICKS)
        return None
    agent.held.remove(obj_id)
    world.drop(obj_id, agent.cell)
    world.break_object(obj_id)
    agent.shards = obj_id
    return ActionResult(call, "skill_failure", f"{obj_id} dropped {when} and shattered")


def bearing(start: Cell, end: Cell) -> float:
    """Heading in degrees from one cell to another (0 east, 90 south)."""
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360
