"""Event-driven episode loop on the simulated clock (hard rules 1-4).

Order within a tick: churn, message delivery, perception (which also marks the coverage map),
task reveal (discovered mode), task discovery, coverage sharing, decisions, skill starts, explorer
retargeting, movement (all movers together, `sim.motion`), other skill steps, appliance cycles,
breakage, ground-truth task progress, poses, snapshot. Perception runs before decisions
so a prompt frozen at tick t shows what is in view at t. A decision freezes the agent's observation
at tick t, and its tool calls start at t + L, where L comes from the latency model. Teammates keep
acting during those L ticks.

Task visibility: in "posted" mode every mission task is logged `posted` at tick 0 and known to the
team. In "discovered" mode mission tasks are logged `exists` (ground truth, for metrics) and no
agent knows them until it sees one of a task's reveal objects. Either way the mission is the
ground-truth task set; perception triggers (`sim.triggers`) only add follow-ups, never a copy of a
mission task.
"""

import copy
import dataclasses
import random
import time
from concurrent.futures import ThreadPoolExecutor

from llm.latency import latency_ticks
from sim.agent import (
    ActionResult,
    AgentState,
    DecisionRequest,
    DecisionResponse,
    Policy,
    TickingPolicy,
    TickView,
    ToolCall,
    build_prompt,
)
from sim.beliefs import (
    Belief,
    initial_beliefs,
    omniscient_beliefs,
    sync_hands,
    update_beliefs,
)
from sim.channel import Message, Network
from sim.config import (
    BREAKABLE_TYPES,
    CLEAN_UP_TOOL,
    CODE_MESSAGE_KINDS,
    DIALOGUE_CONDITION,
    EXPLORING_TOOLS,
    MOVING_TOOLS,
    REFLEX_TOOLS,
    RETURN_ITEM_TOOL,
    RETURN_TO_TASK_TOOL,
    SPLITTABLE_KIND,
    ChurnConfig,
    EpisodeConfig,
    LatencyConfig,
)
from sim.exploration import (
    COVERAGE_KIND,
    coverage_payload,
    is_searched,
    mark_searched,
    merge_coverage,
    nearest_frontier,
    retarget_explorers,
    team_coverage,
)
from sim.logger import EpisodeLogger, JsonValue
from sim.motion import advance_moves
from sim.pathing import PathPlanner
from sim.perception import VisibleCells
from sim.primitives import (
    advance_skill,
    check_against_beliefs,
    finish_clean_up,
    finish_return,
    finish_return_item,
    moved_text,
    return_to_task_area,
    start_clean_up,
    start_return_item,
    start_skill,
)
from sim.scenario import Scenario, TaskSpec, goal_holds, task_from_entry
from sim.triggers import find_triggered_tasks
from sim.world import Cell, World

# Announcing tools -> the task transition their success logs.
ANNOUNCE_TRANSITIONS: dict[str, str] = {
    "report": "reported",
    "claim": "claimed",
    "release": "released",
}


def run_episode(
    config: EpisodeConfig, scenario: Scenario, policies: dict[str, Policy], logger: EpisodeLogger
) -> int:
    """Runs one episode until every task is done or `max_ticks` is reached.

    Returns:
        The last tick simulated.
    """
    world = scenario.world
    planner = PathPlanner(world)
    sight = VisibleCells(world)
    blocked = world.blocked_cells()
    floor = frozenset(cell for cell in world.rooms if cell not in blocked)
    furniture = frozenset(blocked - world.walls)
    ttl = config.exploration.search_ttl
    share_every = config.exploration.share_every
    silenced = ("task_found",) if config.condition == DIALOGUE_CONDITION else ()
    network = Network(config.channel, world.walls, config.seed, logger, silenced)
    skill_rng = random.Random(f"{config.seed}:skills")
    motion_rng = random.Random(f"{config.seed}:motion")
    starting_beliefs = omniscient_beliefs if config.condition == "oracle" else initial_beliefs
    agents = {
        spec.agent_id: AgentState(
            spec.agent_id, spec.card, spec.cell, spec.heading, starting_beliefs(world)
        )
        for spec in scenario.agents
    }
    team_ids = sorted(agents)
    write_belief_graphs(agents, "initial", 0, logger)
    crash_ticks = schedule_crashes(config.churn, team_ids, random.Random(f"{config.seed}:churn"))
    done_ticks: dict[str, int] = {}
    voided: set[str] = set()
    mission_ids = {task.task_id for task in scenario.tasks}
    posted = config.task_visibility == "posted"
    reveal_objects = {} if posted else reveal_objects_of(scenario.tasks, world)
    revealed: set[str] = set()
    for task in scenario.tasks:
        transition = "posted" if posted else "exists"
        logger.write("events", 0, {"agent": None, "task": task.task_id, "transition": transition})

    code_inbox: dict[str, list[Message]] = {agent_id: [] for agent_id in team_ids}
    for tick in range(config.max_ticks):
        world.tick = tick
        for agent_id, crash_tick in sorted(crash_ticks.items()):
            if tick == crash_tick:
                crash(agents[agent_id], world, logger)
            elif tick == crash_tick + config.churn.rejoin_delay:
                rejoin(agents[agent_id], logger, tick)
        online = {agent_id: agents[agent_id].online for agent_id in team_ids}
        for message in network.deliver(tick, online):
            if message.kind == COVERAGE_KIND:
                # Perception-layer data: it updates the map and never prompts a decision.
                merge_coverage(agents[message.receiver].searched, message.payload)
                continue
            if message.kind in CODE_MESSAGE_KINDS:
                code_inbox[message.receiver].append(message)
                continue
            agents[message.receiver].unread.append(message)
            if message.kind == "task_found":
                agents[message.receiver].known_tasks.add(message.payload["task_id"])
        for agent_id in team_ids:
            if agents[agent_id].online:
                perceive(agents[agent_id], world, sight, ttl, logger)
                reveal_tasks(
                    agents[agent_id], agents, reveal_objects, revealed, network, tick, logger
                )
                discover_tasks(
                    agents[agent_id], agents, scenario, mission_ids, network, tick, logger
                )
        if share_every is not None and tick % share_every == 0:
            share_coverage(agents, network, tick)
        for agent_id in team_ids:
            if agents[agent_id].online and isinstance(policies[agent_id], TickingPolicy):
                coordinate(
                    agents,
                    agent_id,
                    policies[agent_id],
                    code_inbox,
                    world,
                    planner,
                    network,
                    config.latency,
                    logger,
                )
        requests = {
            agent_id: prepare_decision(agents[agent_id], world, planner, floor, furniture, config)
            for agent_id in team_ids
            if agents[agent_id].at_decision_point(tick)
        }
        answers = decide_all(policies, requests, config.executor == "llm")
        for agent_id, request in requests.items():
            response, wall_seconds = answers[agent_id]
            apply_decision(
                agents[agent_id], request, response, wall_seconds, team_ids, config, logger
            )
        for agent_id in team_ids:
            start_next_skill(agents[agent_id], planner, skill_rng, config.faults.drop_prob, tick)
        explore_results = retarget_explorers(agents, floor, furniture, tick, ttl)
        move_results = advance_moves(agents, world, planner, motion_rng, frozenset(explore_results))
        move_results.update(explore_results)
        for agent_id in team_ids:
            step_agent(
                agents[agent_id],
                agents,
                world,
                planner,
                network,
                skill_rng,
                config.faults.drop_prob,
                move_results.get(agent_id),
                logger,
            )
        for child in handle_cycle_starts(
            world, scenario, agents, set(done_ticks) | voided, revealed, network, logger
        ):
            mission_ids.add(child.task_id)  # its dishes are covered: no trigger may copy them
            if not posted:
                reveal_objects[child.task_id] = tuple(goal.obj_id for goal in child.goals)
        world.advance_cycles()
        log_drops(world, agents, logger)
        handle_breaks(world, scenario, done_ticks, voided, logger)
        record_task_progress(world, scenario.tasks, done_ticks, voided, logger)
        all_done = len(done_ticks) + len(voided) == len(scenario.tasks)
        last_tick = all_done or tick == config.max_ticks - 1
        write_poses(agents, tick, logger)
        if tick % config.snapshot_every == 0 or last_tick:
            write_snapshot(world, agents, floor, ttl, logger)
        if last_tick:
            write_belief_graphs(agents, "final", tick, logger)
            return tick
    return config.max_ticks - 1


def schedule_crashes(churn: ChurnConfig, team_ids: list[str], rng: random.Random) -> dict[str, int]:
    """Picks which k agents crash and when. Each rejoins `rejoin_delay` ticks later."""
    chosen = sorted(rng.sample(team_ids, churn.num_agents))
    return {agent_id: rng.randint(churn.earliest_tick, churn.latest_tick) for agent_id in chosen}


def crash(agent: AgentState, world: World, logger: EpisodeLogger) -> None:
    """Takes an agent offline. Held objects fall to its tile. Beliefs are kept.

    Breakable objects shatter, with nobody left to clean up: the shards stay on the floor as debris,
    with no task. Other objects can be picked up by anyone. Its reservations are released. The body
    stays on its cell as an obstacle until it rejoins.
    """
    for obj_id in agent.held:
        world.drop(obj_id, agent.cell)
        world.drops.append((agent.agent_id, obj_id))
        if world.objects[obj_id].obj_type in BREAKABLE_TYPES:
            world.break_object(obj_id)
    agent.held.clear()
    agent.shards = None
    agent.task_area = None
    agent.pick_places.clear()
    agent.picked_from.clear()
    world.release_reservations(agent.agent_id)
    agent.queue.clear()
    agent.unread.clear()
    agent.skill = None
    agent.online = False
    agent.decision_due = False
    agent.thinking_until = 0
    agent.last_result = None
    logger.write(
        "events", world.tick, {"agent": agent.agent_id, "task": None, "transition": "crashed"}
    )


def rejoin(agent: AgentState, logger: EpisodeLogger, tick: int) -> None:
    """Brings a crashed agent back with its memory retained. Rejoining is a decision point."""
    agent.online = True
    agent.decision_due = True
    logger.write("events", tick, {"agent": agent.agent_id, "task": None, "transition": "rejoined"})


def coordinate(
    agents: dict[str, AgentState],
    agent_id: str,
    policy: TickingPolicy,
    code_inbox: dict[str, list[Message]],
    world: World,
    planner: PathPlanner,
    network: Network,
    latency: LatencyConfig,
    logger: EpisodeLogger,
) -> None:
    """Runs one agent's coordination code for this tick (CBBA, ledger gossip), with no LLM call.

    Its messages since the last tick are handed over, its broadcasts go out over the lossy
    channel, its changed ledger rows are logged, and a changed assignment makes a decision due.
    """
    agent = agents[agent_id]
    tick = world.tick
    observation = None
    if policy.wants_observation(tick):
        observation = build_prompt(agent, world, planner, tick, None, None)
    view = TickView(
        agent_id,
        tick,
        agent.cell,
        agent.beliefs,
        tuple(sorted(agent.known_tasks)),
        tuple(code_inbox[agent_id]),
        tuple(agent.history),
        agent.card,
        planner,
        {other_id: other.cell for other_id, other in agents.items()},
        {other_id: other.online for other_id, other in agents.items()},
        observation,
    )
    code_inbox[agent_id] = []
    report = policy.on_tick(view)
    positions = {other_id: other.cell for other_id, other in agents.items()}
    online = {other_id: other.online for other_id, other in agents.items()}
    for kind, payload in report.broadcasts:
        network.broadcast(agent_id, kind, payload, tick, positions, online)
    if report.ledger_rows:
        logger.write("ledgers", tick, {"agent": agent_id, "rows": report.ledger_rows})
    for row in report.dialogue:
        logger.write("dialogues", tick, {"agent": agent_id, **row})
    for call in report.llm_calls:
        delay = latency_ticks(call.prompt_tokens, call.output_tokens, latency)
        logger.write(
            "llm_calls",
            tick,
            {
                "agent": agent_id,
                "tick_applied": tick + delay,
                "task": None,
                "tool_calls": [],
                "rejected": False,
                "purpose": call.purpose,
                "latency_ticks": delay,
                "prompt": call.prompt,
                "raw_response": call.raw_text,
                "prompt_tokens": call.prompt_tokens,
                "output_tokens": call.output_tokens,
                "cached": call.cached,
                "wall_seconds": call.wall_seconds,
            },
        )
    if report.decision_due:
        agent.decision_due = True


def prepare_decision(
    agent: AgentState,
    world: World,
    planner: PathPlanner,
    floor: frozenset[Cell],
    furniture: frozenset[Cell],
    config: EpisodeConfig,
) -> DecisionRequest:
    """Freezes an agent's observation at this tick into a decision request (hard rule 4)."""
    tick = world.tick
    ttl = config.exploration.search_ttl
    frontier = nearest_frontier(agent.cell, floor, furniture, agent.searched, tick, ttl)
    frontier_steps = None if frontier is None else planner.steps(agent.cell, frontier)
    own_frontier = nearest_frontier(agent.cell, floor, furniture, agent.own_searched, tick, ttl)
    own_frontier_steps = None if own_frontier is None else planner.steps(agent.cell, own_frontier)
    object_steps = {
        obj_id: planner.steps_to_any(agent.cell, belief.cells())
        for obj_id, belief in agent.beliefs.items()
    }
    return DecisionRequest(
        agent.agent_id,
        tick,
        build_prompt(agent, world, planner, tick, frontier_steps, own_frontier_steps),
        list(agent.unread),
        agent.last_result,
        copy.deepcopy(agent.beliefs),
        object_steps,
        tuple(sorted(agent.known_tasks)),
        tuple(
            sorted(
                {
                    furniture_id
                    for holder, furniture_id in world.reservations.values()
                    if holder == agent.agent_id
                }
            )
        ),
        frontier_steps,
        own_frontier_steps,
        agent.card,
        tuple(agent.history),
    )


def decide_all(
    policies: dict[str, Policy], requests: dict[str, DecisionRequest], parallel: bool
) -> dict[str, tuple[DecisionResponse, float]]:
    """Every deciding agent's response and its wall-clock seconds (a cost only, hard rule 1).

    With `parallel`, the requests go out together, so a model server batches them (one model, many
    agents, hard rule 6). Each policy holds only its own agent's state and the results are applied
    in agent order afterwards, so thread timing cannot change an outcome.
    """
    if not parallel or len(requests) < 2:
        return {
            agent_id: timed_decide(policies[agent_id], request)
            for agent_id, request in requests.items()
        }
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        futures = {
            agent_id: pool.submit(timed_decide, policies[agent_id], request)
            for agent_id, request in requests.items()
        }
        return {agent_id: future.result() for agent_id, future in futures.items()}


def timed_decide(policy: Policy, request: DecisionRequest) -> tuple[DecisionResponse, float]:
    """A policy's response and the wall-clock seconds it took."""
    began = time.perf_counter()
    response = policy.decide(request)
    return response, time.perf_counter() - began


def apply_decision(
    agent: AgentState,
    request: DecisionRequest,
    response: DecisionResponse,
    wall_seconds: float,
    team_ids: list[str],
    config: EpisodeConfig,
    logger: EpisodeLogger,
) -> None:
    """Schedules a response's calls L ticks after the request and logs every model call.

    L is the sum of each model call's latency (hard rule 3). A reply the policy could not use, or
    any call that fails the belief check, rejects the whole response; the failure is shown at the
    next decision, which happens once the L ticks have passed. An interruptible skill (wait,
    explore) ends here; an interrupted explore is logged so exploration time is measurable.
    """
    tick = request.tick
    calls = response.llm_calls
    if calls:
        delays = [
            latency_ticks(call.prompt_tokens, call.output_tokens, config.latency) for call in calls
        ]
    else:
        delays = [latency_ticks(response.prompt_tokens, response.output_tokens, config.latency)]
    delay = sum(delays)
    rejection = None
    if response.error is not None:
        rejection = ActionResult(ToolCall("reply", ()), "invalid_call", response.error)
    for call in response.tool_calls:
        if rejection is not None:
            break
        rejection = check_against_beliefs(call, agent, team_ids)
    if agent.skill is not None and agent.skill.interruptible:
        if agent.skill.call.name in EXPLORING_TOOLS:
            interrupted = ActionResult(agent.skill.call, "ok", "interrupted")
            log_action(agent, interrupted, agent.skill.started, tick, logger)
        agent.skill = None
    agent.unread = []
    agent.thinking_until = tick + delay
    agent.queue = [] if rejection is not None else list(response.tool_calls)
    agent.pick_places = {
        call.args[0]: agent.beliefs[call.args[0]].place()
        for call in agent.queue
        if call.name == "pick"
    }
    agent.current_task = response.task_id
    agent.last_result = rejection
    agent.decision_due = rejection is not None
    shared = {
        "agent": agent.agent_id,
        "tick_applied": tick + delay,
        "task": response.task_id,
    }
    acted = {
        "tool_calls": [[call.name, *call.args] for call in response.tool_calls],
        "rejected": rejection is not None,
    }
    if not calls:
        logger.write(
            "llm_calls",
            tick,
            {
                **shared,
                **acted,
                "purpose": "act",
                "latency_ticks": delay,
                "prompt": request.prompt,
                "raw_response": response.raw_text,
                "prompt_tokens": response.prompt_tokens,
                "output_tokens": response.output_tokens,
                "cached": False,
                "wall_seconds": wall_seconds,
            },
        )
    for call, call_delay in zip(calls, delays if calls else [], strict=True):
        logger.write(
            "llm_calls",
            tick,
            {
                **shared,
                **(acted if call.purpose == "act" else {"tool_calls": [], "rejected": False}),
                "purpose": call.purpose,
                "latency_ticks": call_delay,
                "prompt": call.prompt,
                "raw_response": call.raw_text,
                "prompt_tokens": call.prompt_tokens,
                "output_tokens": call.output_tokens,
                "cached": call.cached,
                "wall_seconds": call.wall_seconds,
            },
        )
    if rejection is not None:
        log_action(agent, rejection, tick, tick, logger)


def start_next_skill(
    agent: AgentState, planner: PathPlanner, rng: random.Random, drop_prob: float, tick: int
) -> None:
    """Starts the next queued call if the agent is idle, online, and done thinking."""
    if not agent.online or tick < agent.thinking_until or agent.skill is not None:
        return
    if agent.queue:
        agent.skill = start_skill(agent.queue.pop(0), agent, planner, rng, drop_prob)
        agent.skill.started = tick


def step_agent(
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    network: Network,
    rng: random.Random,
    drop_prob: float,
    move_result: ActionResult | None,
    logger: EpisodeLogger,
) -> None:
    """Advances the agent's current skill by one tick and handles its result.

    A moving skill already ran in the movement phase; `move_result` is its outcome, if it ended. A
    shatter starts a reflex in place of a decision (clean up, then return to the task area), so the
    agent decides only once it is back.
    """
    if not agent.online or world.tick < agent.thinking_until or agent.skill is None:
        return
    if agent.skill.call.name in MOVING_TOOLS:
        result = move_result
    else:
        result = advance_skill(agent, agents, world, planner, network, rng, drop_prob)
    if result is None:
        return
    if agent.skill.call.name == CLEAN_UP_TOOL:
        result = finish_clean_up(agent, world)
    elif agent.skill.call.name == RETURN_TO_TASK_TOOL:
        result = finish_return(agent)
    elif agent.skill.call.name == RETURN_ITEM_TOOL:
        result = finish_return_item(agent, world)
    started = agent.skill.started
    agent.skill = None
    agent.last_result = result
    log_action(agent, result, started, world.tick, logger)
    if agent.shards is not None and start_clean_up(agent, world, planner):
        return
    if result.call.name == CLEAN_UP_TOOL:
        return_to_task_area(agent, world, result.call)
        return
    if result.ok and result.call.name in ANNOUNCE_TRANSITIONS:
        logger.write(
            "events",
            world.tick,
            {
                "agent": agent.agent_id,
                "task": result.call.args[0],
                "transition": ANNOUNCE_TRANSITIONS[result.call.name],
            },
        )
    if not result.ok:
        agent.queue.clear()
        agent.decision_due = True
    elif not agent.queue or agent.unread:
        agent.decision_due = True


def perceive(
    agent: AgentState,
    world: World,
    sight: VisibleCells,
    ttl: int | None,
    logger: EpisodeLogger,
) -> None:
    """Runs deterministic perception for one agent: coverage map, then beliefs (changes logged).

    Seeing an object for the first time, or again after believing it missing, while exploring
    makes a decision due, which ends the explore or search so the agent can act on what it found.
    """
    viewpoint = agent.viewpoint()
    mark_searched(agent, sight.of(viewpoint), world.tick, ttl)
    missing = {obj_id for obj_id, belief in agent.beliefs.items() if belief.missing}
    changes = update_beliefs(agent.beliefs, world, viewpoint)
    changes += sync_hands(agent.beliefs, world, agent.agent_id)
    for obj_id, change in changes:
        logger.write(
            "beliefs",
            world.tick,
            {
                "agent": agent.agent_id,
                "obj": obj_id,
                "change": change,
                "belief": dataclasses.asdict(agent.beliefs[obj_id]),
            },
        )
    found = any(
        change == "added" or (obj_id in missing and not agent.beliefs[obj_id].missing)
        for obj_id, change in changes
    )
    exploring = agent.skill is not None and agent.skill.call.name in EXPLORING_TOOLS
    if exploring and found:
        agent.decision_due = True
    cancel_stale_picks(agent, [obj_id for obj_id, _ in changes], world.tick, logger)


def cancel_stale_picks(
    agent: AgentState, changed: list[str], tick: int, logger: EpisodeLogger
) -> None:
    """Cancels a queued pick once perception shows its item is not where the agent believed.

    The walk toward it stops and the queue is dropped, so the agent decides again now instead of
    arriving to find the item gone (or picking it out of the dishwasher a teammate just loaded).
    """
    for obj_id in changed:
        pick = ToolCall("pick", (obj_id,))
        if obj_id not in agent.pick_places or pick not in agent.queue:
            continue
        belief = agent.beliefs[obj_id]
        seen_at = None if belief.missing else belief.place()
        if seen_at == agent.pick_places[obj_id]:
            continue
        expected = agent.pick_places.pop(obj_id)
        result = ActionResult(pick, "stale_belief", moved_text(obj_id, expected))
        if agent.skill is not None and agent.skill.call.name not in REFLEX_TOOLS:
            agent.skill = None
        agent.queue.clear()
        agent.last_result = result
        agent.decision_due = True
        log_action(agent, result, tick, tick, logger)
        return


def reveal_objects_of(tasks: list[TaskSpec], world: World) -> dict[str, tuple[str, ...]]:
    """Per mission task, the objects whose sight reveals it.

    That is the task's `reveal_on`, else its movable goal objects (furniture is known from the
    map, so the default never reveals a task by furniture alone).
    """
    return {
        task.task_id: task.reveal_on
        or tuple(
            goal.obj_id
            for goal in task.goals
            if goal.obj_id in world.objects and not world.objects[goal.obj_id].is_furniture
        )
        for task in tasks
    }


def reveal_tasks(
    agent: AgentState,
    agents: dict[str, AgentState],
    reveal_objects: dict[str, tuple[str, ...]],
    revealed: set[str],
    network: Network,
    tick: int,
    logger: EpisodeLogger,
) -> None:
    """Discovered mode: makes a mission task known to an agent that sees one of its reveal objects.

    Mission ids are unique, so two finders reveal the same task. The first finder in the episode
    logs `discovered`. The finder announces it (`task_found`, over the lossy channel) and, if it can
    be interrupted, decides at once. A busy agent keeps working and only records the task.
    """
    found = [
        task_id
        for task_id in sorted(reveal_objects)
        if task_id not in agent.known_tasks
        and any(_seen_now(agent.beliefs.get(obj_id), tick) for obj_id in reveal_objects[task_id])
    ]
    if not found:
        return
    positions = {agent_id: other.cell for agent_id, other in agents.items()}
    online = {agent_id: other.online for agent_id, other in agents.items()}
    for task_id in found:
        agent.known_tasks.add(task_id)
        if task_id not in revealed:
            revealed.add(task_id)
            logger.write(
                "events",
                tick,
                {"agent": agent.agent_id, "task": task_id, "transition": "discovered"},
            )
        network.broadcast(
            agent.agent_id, "task_found", {"task_id": task_id}, tick, positions, online
        )
    if agent.can_be_interrupted():
        agent.decision_due = True


def _seen_now(belief: Belief | None, tick: int) -> bool:
    """Whether a belief comes from the agent's own sight this tick (map knowledge is not sight)."""
    return (
        belief is not None
        and belief.source == "seen"
        and belief.last_seen == tick
        and not belief.missing
    )


def share_coverage(agents: dict[str, AgentState], network: Network, tick: int) -> None:
    """Broadcasts each online agent's new coverage entries. Only its own sightings are sent."""
    positions = {agent_id: agent.cell for agent_id, agent in agents.items()}
    online = {agent_id: agent.online for agent_id, agent in agents.items()}
    for agent_id in sorted(agents):
        agent = agents[agent_id]
        if agent.online and agent.unsent_coverage:
            payload = coverage_payload(agent.unsent_coverage)
            network.broadcast(agent_id, COVERAGE_KIND, payload, tick, positions, online)
            agent.unsent_coverage = {}


def discover_tasks(
    agent: AgentState,
    agents: dict[str, AgentState],
    scenario: Scenario,
    mission_ids: set[str],
    network: Network,
    tick: int,
    logger: EpisodeLogger,
) -> None:
    """Turns triggers in the agent's beliefs into tasks and announces them on the channel.

    A task new to the episode is added to `scenario.tasks` and logged as "discovered" by this
    agent. Teammates learn of it only through the `task_found` message (which the channel may
    drop) or by perceiving the trigger themselves. Allocating it is the coordination layer's job.
    """
    tasks_by_id = {task.task_id: task for task in scenario.tasks}
    known = [tasks_by_id[task_id] for task_id in sorted(mission_ids | agent.known_tasks)]
    covered = {(task.kind, goal.obj_id) for task in known for goal in task.goals}
    # An agent that has not seen a dish break may still believe in it. A task for a dish that no
    # longer exists could never finish, so triggers skip it (part of the perfect-dedupe premise).
    covered |= {(None, obj_id) for obj_id in scenario.world.lost}
    positions = {agent_id: other.cell for agent_id, other in agents.items()}
    online = {agent_id: other.online for agent_id, other in agents.items()}
    for entry in find_triggered_tasks(agent.beliefs, covered):
        task = task_from_entry(entry, scenario.templates)
        if task.task_id not in tasks_by_id:
            scenario.tasks.append(task)
            tasks_by_id[task.task_id] = task
            logger.write(
                "events",
                tick,
                {"agent": agent.agent_id, "task": task.task_id, "transition": "discovered"},
            )
        agent.known_tasks.add(task.task_id)
        network.broadcast(
            agent.agent_id, "task_found", {"task_id": task.task_id}, tick, positions, online
        )
        if agent.can_be_interrupted():
            agent.decision_due = True


def log_drops(world: World, agents: dict[str, AgentState], logger: EpisodeLogger) -> None:
    """Logs this tick's drops, with the task each agent was on."""
    for agent_id, obj_id in world.drops:
        logger.write(
            "events",
            world.tick,
            {
                "agent": agent_id,
                "task": agents[agent_id].current_task,
                "transition": "dropped",
                "obj": obj_id,
            },
        )
    world.drops.clear()


def handle_breaks(
    world: World,
    scenario: Scenario,
    done_ticks: dict[str, int],
    voided: set[str],
    logger: EpisodeLogger,
) -> None:
    """Takes each shattered object out of the unfinished tasks that needed it.

    Only that item is lost: its goals go, and the task stays with its agent for the rest of its
    items. A task with no movable item left has nothing to do, so it is voided, which counts as
    finished for precedence and for the end of the episode and is logged "voided", never "done".
    """
    for obj_id in world.broken:
        logger.write(
            "events",
            world.tick,
            {"agent": None, "task": None, "transition": "broken", "obj": obj_id},
        )
        for index, task in enumerate(scenario.tasks):
            needs_object = any(goal.obj_id == obj_id for goal in task.goals)
            if not needs_object or task.task_id in done_ticks or task.task_id in voided:
                continue
            remaining = tuple(goal for goal in task.goals if goal.obj_id != obj_id)
            scenario.tasks[index] = dataclasses.replace(task, goals=remaining)
            if not any(_is_movable_item(goal.obj_id, world) for goal in remaining):
                voided.add(task.task_id)
                logger.write(
                    "events",
                    world.tick,
                    {"agent": None, "task": task.task_id, "transition": "voided"},
                )
    world.broken.clear()


def handle_cycle_starts(
    world: World,
    scenario: Scenario,
    agents: dict[str, AgentState],
    finished: set[str],
    revealed: set[str],
    network: Network,
    logger: EpisodeLogger,
) -> list[TaskSpec]:
    """Splits each wash caught part-loaded when someone starts its dishwasher.

    A simulator rule, the same in every condition. The dishes inside finish with this cycle, so
    they stay in the task, which is then done when the cycle ends; its steps are emptied, since
    nothing is left to do. The dishes not yet loaded become a new task with the full recipe, an
    ordinary task for the team's allocator. Every agent holding one of them, and the agent that
    started the machine, finds the new task and announces it. A holder carries its dish back to
    where it got it by reflex (`return_item`) before it decides anything.

    Returns:
        The new tasks.
    """
    children = []
    positions = {agent_id: agent.cell for agent_id, agent in agents.items()}
    online = {agent_id: agent.online for agent_id, agent in agents.items()}
    for appliance_id, starter in world.cycles_started:
        for index, task in enumerate(list(scenario.tasks)):
            if task.kind != SPLITTABLE_KIND or task.task_id in finished:
                continue
            items = [goal.obj_id for goal in task.goals if goal.relation == "washed"]
            loaded = [item for item in items if world.objects[item].parent_id == appliance_id]
            rest = [
                item
                for item in items
                if item not in loaded and world.objects[item].visual_attrs.get("dirty") != "no"
            ]
            if not loaded or not rest:
                continue
            child = split_wash(task, rest, scenario)
            kept = tuple(goal for goal in task.goals if goal.obj_id not in rest)
            scenario.tasks[index] = dataclasses.replace(task, goals=kept, steps=())
            scenario.tasks.append(child)
            children.append(child)
            row = {"agent": starter, "task": task.task_id, "transition": "split"}
            logger.write("events", world.tick, {**row, "child": child.task_id, "items": rest})
            logger.write(
                "events", world.tick, {"agent": None, "task": child.task_id, "transition": "exists"}
            )
            holders = sorted(
                agent_id for agent_id, agent in agents.items() if set(agent.held) & set(rest)
            )
            for agent_id in sorted({starter, *holders}):
                if child.task_id not in revealed:
                    revealed.add(child.task_id)
                    row = {"agent": agent_id, "task": child.task_id, "transition": "discovered"}
                    logger.write("events", world.tick, row)
                agents[agent_id].known_tasks.add(child.task_id)
                payload = {"task_id": child.task_id}
                network.broadcast(agent_id, "task_found", payload, world.tick, positions, online)
            for agent_id in holders:
                for item in sorted(set(agents[agent_id].held) & set(rest)):
                    start_return_item(agents[agent_id], world, item)
            for agent_id in sorted(set(agents) - set(holders)):
                stop_picks_of(agents[agent_id], rest, task.task_id, world.tick, logger)
    world.cycles_started.clear()
    return children


def stop_picks_of(
    agent: AgentState, items: list[str], task_id: str, tick: int, logger: EpisodeLogger
) -> None:
    """Stops a pick, running or queued, of a dish that just moved to a split's new task.

    Otherwise the agent would finish picking it for a task that no longer holds it.
    """
    running = agent.skill is not None and agent.skill.call in [
        ToolCall("pick", (item,)) for item in items
    ]
    queued = [call for call in agent.queue if call.name == "pick" and call.args[0] in items]
    if not running and not queued:
        return
    call = agent.skill.call if running else queued[0]
    if running:
        agent.skill = None
    agent.queue.clear()
    agent.decision_due = True
    agent.last_result = ActionResult(
        call, "stale_belief", f"{task_id} split; not picking {call.args[0]}"
    )
    log_action(agent, agent.last_result, tick, tick, logger)


def split_wash(task: TaskSpec, items: list[str], scenario: Scenario) -> TaskSpec:
    """A new wash task for the given dishes, with the parent's machine and precedence."""
    appliance = next(step.args[0] for step in task.steps if step.name == "start")
    earlier = sum(1 for other in scenario.tasks if other.task_id.startswith(f"{task.task_id}_r"))
    entry = {
        "id": f"{task.task_id}_r{earlier + 1}",
        "kind": task.kind,
        "items": items,
        "appliance": appliance,
        "after": list(task.after),
    }
    return task_from_entry(entry, scenario.templates)


def _is_movable_item(obj_id: str, world: World) -> bool:
    """Whether a goal is about an item rather than furniture (or an item already deleted)."""
    return obj_id in world.objects and not world.objects[obj_id].is_furniture


def record_task_progress(
    world: World,
    tasks: list[TaskSpec],
    done_ticks: dict[str, int],
    voided: set[str],
    logger: EpisodeLogger,
) -> None:
    """Marks tasks done in ground truth. Done is sticky and requires predecessors to be finished."""
    for task in tasks:
        if task.task_id in done_ticks or task.task_id in voided:
            continue
        predecessors_done = all(
            task_id in done_ticks or task_id in voided for task_id in task.after
        )
        if predecessors_done and all(goal_holds(world, goal) for goal in task.goals):
            done_ticks[task.task_id] = world.tick
            logger.write(
                "events", world.tick, {"agent": None, "task": task.task_id, "transition": "done"}
            )


def log_action(
    agent: AgentState, result: ActionResult, started: int, tick: int, logger: EpisodeLogger
) -> None:
    """Logs one tool-call outcome, keeping hallucinations and stale beliefs distinct.

    `tick` is when it finished, `start_tick` when it began, and `task` what it was for. The
    outcome also joins the agent's action history, which LLM prompts show.
    """
    agent.history.append((tick, result))
    logger.write(
        "actions",
        tick,
        {
            "agent": agent.agent_id,
            "call": [result.call.name, *result.call.args],
            "outcome": result.outcome,
            "text": result.text,
            "start_tick": started,
            "task": agent.current_task,
        },
    )


def write_snapshot(
    world: World,
    agents: dict[str, AgentState],
    floor: frozenset[Cell],
    ttl: int | None,
    logger: EpisodeLogger,
) -> None:
    """Writes a full state snapshot (a keyframe; `poses` has the agents every tick).

    `coverage` is the share of floor some agent counts as searched; each agent's `searched` is the
    number of floor cells its own map counts.
    """
    agent_rows: dict[str, JsonValue] = {
        agent_id: {
            "cell": list(agent.cell),
            "heading": agent.heading,
            "held": list(agent.held),
            "online": agent.online,
            "blocked_ticks": agent.blocked_ticks,
            "yields": agent.yields,
            "move_conflicts": agent.move_conflicts,
            "searched": sum(
                1
                for cell in agent.searched
                if cell in floor and is_searched(agent.searched, cell, world.tick, ttl)
            ),
        }
        for agent_id, agent in sorted(agents.items())
    }
    object_rows: dict[str, JsonValue] = {
        obj_id: {
            "cell": list(obj.cell),
            "parent": obj.parent_id,
            "relation": obj.parent_relation,
            "held_by": obj.held_by,
            "open": obj.is_open,
            "type": obj.obj_type,
            "cycle": obj.cycle,
        }
        for obj_id, obj in sorted(world.objects.items())
        if not obj.is_furniture or obj.is_container
    }
    reservations: list[JsonValue] = [
        [list(cell), holder, furniture_id]
        for cell, (holder, furniture_id) in sorted(world.reservations.items())
    ]
    logger.write(
        "state",
        world.tick,
        {
            "agents": agent_rows,
            "objects": object_rows,
            "reservations": reservations,
            "coverage": team_coverage(agents, floor, world.tick, ttl),
        },
    )


def write_poses(agents: dict[str, AgentState], tick: int, logger: EpisodeLogger) -> None:
    """Writes every agent's cell, heading, held objects, and online flag for this tick."""
    poses: dict[str, JsonValue] = {
        agent_id: {
            "cell": list(agent.cell),
            "heading": agent.heading,
            "held": list(agent.held),
            "online": agent.online,
        }
        for agent_id, agent in sorted(agents.items())
    }
    logger.write("poses", tick, {"agents": poses})


def write_belief_graphs(
    agents: dict[str, AgentState], change: str, tick: int, logger: EpisodeLogger
) -> None:
    """Writes every agent's full belief graph: "initial" at tick 0, "final" at the end.

    Between the two, the stream holds every change with its new belief, so the initial graph plus
    the changes rebuild any agent's beliefs at any tick.
    """
    for agent_id, agent in sorted(agents.items()):
        beliefs = {
            obj_id: dataclasses.asdict(agent.beliefs[obj_id]) for obj_id in sorted(agent.beliefs)
        }
        logger.write("beliefs", tick, {"agent": agent_id, "change": change, "beliefs": beliefs})
