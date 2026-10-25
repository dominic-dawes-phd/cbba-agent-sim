"""Event-driven episode loop on the simulated clock (hard rules 1-4).

Order within a tick: churn, message delivery, perception, decisions, skill steps, ground-truth task
progress, snapshot. Perception runs before decisions so a prompt frozen at tick t shows what is in
view at t. A decision freezes the agent's observation at tick t, and its tool calls start
at t + L, where L comes from the latency model. Teammates keep acting during those L ticks.
"""

import dataclasses
import random
import time

from llm.latency import latency_ticks
from sim.agent import ActionResult, AgentState, DecisionRequest, Policy, build_prompt
from sim.beliefs import initial_beliefs, update_beliefs
from sim.channel import Network
from sim.config import ChurnConfig, EpisodeConfig
from sim.logger import EpisodeLogger, JsonValue
from sim.pathing import PathPlanner
from sim.primitives import advance_skill, check_against_beliefs, start_skill
from sim.scenario import Scenario, TaskSpec, goal_holds
from sim.world import World


def run_episode(
    config: EpisodeConfig, scenario: Scenario, policies: dict[str, Policy], logger: EpisodeLogger
) -> int:
    """Runs one episode until every task is done or `max_ticks` is reached.

    Returns:
        The last tick simulated.
    """
    world = scenario.world
    planner = PathPlanner(world)
    network = Network(config.channel, world.walls, config.seed, logger)
    skill_rng = random.Random(f"{config.seed}:skills")
    agents = {
        spec.agent_id: AgentState(
            spec.agent_id, spec.card, spec.cell, spec.heading, initial_beliefs(world)
        )
        for spec in scenario.agents
    }
    team_ids = sorted(agents)
    crash_ticks = schedule_crashes(config.churn, team_ids, random.Random(f"{config.seed}:churn"))
    done_ticks: dict[str, int] = {}

    for tick in range(config.max_ticks):
        world.tick = tick
        for agent_id, crash_tick in sorted(crash_ticks.items()):
            if tick == crash_tick:
                crash(agents[agent_id], world, logger)
            elif tick == crash_tick + config.churn.rejoin_delay:
                rejoin(agents[agent_id], logger, tick)
        online = {agent_id: agents[agent_id].online for agent_id in team_ids}
        for message in network.deliver(tick, online):
            agents[message.receiver].unread.append(message)
        for agent_id in team_ids:
            if agents[agent_id].online:
                perceive(agents[agent_id], world, logger)
        for agent_id in team_ids:
            if agents[agent_id].at_decision_point(tick):
                run_decision(
                    agents[agent_id], policies[agent_id], world, planner, team_ids, config, logger
                )
        for agent_id in team_ids:
            step_agent(agents[agent_id], agents, world, planner, network, skill_rng, logger)
        record_task_progress(world, scenario.tasks, done_ticks, logger)
        all_done = len(done_ticks) == len(scenario.tasks)
        if tick % config.snapshot_every == 0 or all_done:
            write_snapshot(world, agents, logger)
        if all_done:
            return tick
    return config.max_ticks - 1


def schedule_crashes(churn: ChurnConfig, team_ids: list[str], rng: random.Random) -> dict[str, int]:
    """Picks which k agents crash and when. Each rejoins `rejoin_delay` ticks later."""
    chosen = sorted(rng.sample(team_ids, churn.num_agents))
    return {agent_id: rng.randint(churn.earliest_tick, churn.latest_tick) for agent_id in chosen}


def crash(agent: AgentState, world: World, logger: EpisodeLogger) -> None:
    """Takes an agent offline. Held objects fall by the drop rule. Beliefs are kept."""
    for obj_id in agent.held:
        world.drop(obj_id, agent.cell)
    agent.held.clear()
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


def run_decision(
    agent: AgentState,
    policy: Policy,
    world: World,
    planner: PathPlanner,
    team_ids: list[str],
    config: EpisodeConfig,
    logger: EpisodeLogger,
) -> None:
    """Queries the policy on a frozen observation and schedules its calls L ticks later.

    If any returned call fails the belief check, the whole response is rejected and the failure
    is shown at the next decision, which happens once the L ticks have passed.
    """
    tick = world.tick
    prompt = build_prompt(agent, world, planner, tick)
    request = DecisionRequest(agent.agent_id, tick, prompt, list(agent.unread), agent.last_result)
    wall_start = time.perf_counter()
    response = policy.decide(request)
    wall_seconds = time.perf_counter() - wall_start
    delay = latency_ticks(response.prompt_tokens, response.output_tokens, config.latency)
    rejection = None
    for call in response.tool_calls:
        rejection = check_against_beliefs(call, agent, team_ids)
        if rejection is not None:
            break
    if agent.skill is not None and agent.skill.interruptible:
        agent.skill = None
    agent.unread = []
    agent.thinking_until = tick + delay
    agent.queue = [] if rejection is not None else list(response.tool_calls)
    agent.last_result = rejection
    agent.decision_due = rejection is not None
    logger.write(
        "llm_calls",
        tick,
        {
            "agent": agent.agent_id,
            "tick_applied": tick + delay,
            "latency_ticks": delay,
            "prompt": prompt,
            "raw_response": response.raw_text,
            "tool_calls": [[call.name, *call.args] for call in response.tool_calls],
            "prompt_tokens": response.prompt_tokens,
            "output_tokens": response.output_tokens,
            "rejected": rejection is not None,
            "wall_seconds": wall_seconds,
        },
    )
    if rejection is not None:
        log_action(agent, rejection, tick, logger)


def step_agent(
    agent: AgentState,
    agents: dict[str, AgentState],
    world: World,
    planner: PathPlanner,
    network: Network,
    rng: random.Random,
    logger: EpisodeLogger,
) -> None:
    """Advances the agent's current skill by one tick, starting the next queued call if idle."""
    if not agent.online or world.tick < agent.thinking_until:
        return
    if agent.skill is None:
        if not agent.queue:
            return
        agent.skill = start_skill(agent.queue.pop(0), agent, planner)
    result = advance_skill(agent, agents, world, planner, network, rng)
    if result is None:
        return
    agent.skill = None
    agent.last_result = result
    log_action(agent, result, world.tick, logger)
    if result.ok and result.call.name == "report":
        logger.write(
            "events",
            world.tick,
            {"agent": agent.agent_id, "task": result.call.args[0], "transition": "reported"},
        )
    if not result.ok:
        agent.queue.clear()
        agent.decision_due = True
    elif not agent.queue or agent.unread:
        agent.decision_due = True


def perceive(agent: AgentState, world: World, logger: EpisodeLogger) -> None:
    """Runs deterministic perception for one agent and logs belief changes."""
    for obj_id, change in update_beliefs(agent.beliefs, world, agent.viewpoint()):
        logger.write(
            "beliefs", world.tick, {"agent": agent.agent_id, "obj": obj_id, "change": change}
        )


def record_task_progress(
    world: World, tasks: list[TaskSpec], done_ticks: dict[str, int], logger: EpisodeLogger
) -> None:
    """Marks tasks done in ground truth. Done is sticky and requires predecessors to be done."""
    for task in tasks:
        if task.task_id in done_ticks:
            continue
        predecessors_done = all(task_id in done_ticks for task_id in task.after)
        if predecessors_done and all(goal_holds(world, goal) for goal in task.goals):
            done_ticks[task.task_id] = world.tick
            logger.write(
                "events", world.tick, {"agent": None, "task": task.task_id, "transition": "done"}
            )


def log_action(agent: AgentState, result: ActionResult, tick: int, logger: EpisodeLogger) -> None:
    """Logs one tool-call outcome, keeping hallucinations and stale beliefs distinct."""
    logger.write(
        "actions",
        tick,
        {
            "agent": agent.agent_id,
            "call": [result.call.name, *result.call.args],
            "outcome": result.outcome,
            "text": result.text,
        },
    )


def write_snapshot(world: World, agents: dict[str, AgentState], logger: EpisodeLogger) -> None:
    """Writes a state snapshot and every agent's full belief graph."""
    agent_rows: dict[str, JsonValue] = {
        agent_id: {
            "cell": list(agent.cell),
            "heading": agent.heading,
            "held": list(agent.held),
            "online": agent.online,
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
        }
        for obj_id, obj in sorted(world.objects.items())
        if not obj.is_furniture or obj.is_container
    }
    logger.write("state", world.tick, {"agents": agent_rows, "objects": object_rows})
    for agent_id, agent in sorted(agents.items()):
        beliefs = {
            obj_id: dataclasses.asdict(agent.beliefs[obj_id]) for obj_id in sorted(agent.beliefs)
        }
        logger.write(
            "beliefs", world.tick, {"agent": agent_id, "change": "snapshot", "beliefs": beliefs}
        )
