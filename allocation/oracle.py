"""Oracle (C0): a central plan made with ground truth, then replayed in the sim.

The plan decides who does which task and in what order, minimizing the finish time of the whole
mission. It is solved over a planning model: task durations and travel come from primitive ticks and
static BFS steps (hard rule 8), plus precedence and exclusive furniture. The model has no bodies or
doorway queues, so the plan is replayed by the shared coded executor, and the replayed makespan is
the denominator for every other condition. This is the "best known plan": any proof of optimality
(`allocation/oracle_cpsat.py`) covers the model, not the sim.

Three numbers per (mission, seed): the proven lower bound in the model, the planned makespan, and
the replayed makespan. Their differences separate solver quality from execution overhead.

Search here is a seeded insertion heuristic with local search (relocate and swap), limited by counts
rather than a clock, so the same seed always gives the same plan.
"""

import dataclasses
import itertools
import math
import random
from dataclasses import dataclass, field

from allocation.scripted import TARGET_ARG, ScriptedPolicy
from sim.agent import DecisionRequest, DecisionResponse, ToolCall
from sim.beliefs import omniscient_beliefs
from sim.config import (
    ORACLE_MAX_MACHINE_ORDERS,
    PRIMITIVE_TICKS,
    WASH_KINDS,
    OracleConfig,
    ScriptedConfig,
)
from sim.pathing import PathPlanner, travel_ticks
from sim.scenario import Scenario, TaskSpec, task_from_entry
from sim.triggers import UNLOAD_KIND, find_triggered_tasks, unload_dest, unload_entry
from sim.world import Cell, World, split_slot

# Steps that can make a goal about their first argument true.
GOAL_MAKING_STEPS = ("place", "open", "close", "start")

Plan = dict[str, list[str]]
# Lower is better: the makespan, then the sum of task completions to break makespan ties.
Score = tuple[float, float]
NO_SCHEDULE: Score = (math.inf, math.inf)


@dataclass(frozen=True)
class LockSpan:
    """Furniture a task holds exclusively, in ticks after the task starts."""

    furniture: str
    start: int
    end: int


@dataclass(frozen=True)
class TaskProfile:
    """One task as one agent would do it, in ticks after it arrives at the first target.

    Attributes:
        busy: Until the agent is free again.
        completion: Until the task's goals hold: later than `busy` when a wash runs on, earlier
            when the goal is met before the agent finishes tidying up.
        locks: Furniture held exclusively, including a dishwasher through each loading trip and
            through its wash. One task can hold the same furniture more than once.
    """

    busy: int
    completion: int
    locks: tuple[LockSpan, ...]


@dataclass(frozen=True)
class PlanProblem:
    """The planning model. An agent with no profile for a task cannot do it (too few hands).

    Attributes:
        first_travel: Ticks for an agent to go from its start cell to a task's first target.
        transition: Ticks for an agent to go from the end of one task to the next task's first
            target.
        after: Each task's predecessors, which must complete before it starts.
        gates: (predecessor, furniture) pairs per task: the predecessor must complete before the
            task's first hold on that furniture (a wash may fetch dishes while the previous
            batch is unloaded, but not load them).
    """

    agents: tuple[str, ...]
    tasks: tuple[str, ...]
    profiles: dict[tuple[str, str], TaskProfile]
    first_travel: dict[tuple[str, str], int]
    transition: dict[tuple[str, str, str], int]
    after: dict[str, tuple[str, ...]]
    gates: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Schedule:
    """A plan's timing in the model."""

    starts: dict[str, int]
    completions: dict[str, int]
    makespan: int


@dataclass(frozen=True)
class OraclePlan:
    """A solved plan, logged to oracle_plan.json.

    Attributes:
        assignment: Each agent's tasks, in order.
        planned: Planned makespan in the model.
        bound: Proven lower bound on any plan's makespan in the model.
        method: "heuristic" or "cpsat".
        status: The solver's verdict: "HEURISTIC", "FEASIBLE", or "OPTIMAL".
        release: The tick each agent should set off for each task. The assignment alone does not
            fix the order on shared furniture, so the replay needs these to keep the plan's order.
        gates: The model's gates (see `PlanProblem`), which the replay keeps too.
    """

    assignment: Plan
    planned: int
    bound: int
    method: str
    status: str
    release: dict[str, int]
    gates: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)


class OraclePolicy(ScriptedPolicy):
    """Replays one agent's solved task list with the shared coded executor.

    A solved plan involves no deliberation, so its decisions cost no tokens and no ticks. Agents
    already know where everything is (`omniscient_beliefs`), so they never explore; one that loses
    track of an object (a teammate moved it out of sight) may search for it. An agent does
    not begin a task before its planned release tick, which keeps the plan's order on shared
    furniture. An agent that finishes parks at its dock, as in every condition.

    Replays drift from plan, so the release tick alone cannot keep two washes apart on one
    dishwasher. A gated wash (see `PlanProblem.gates`) does not touch its machine until the unload
    before it is known to be done, and a busy machine is waited for rather than avoided: a second
    load mixed into a running wash would split it (`sim.loop.handle_cycle_starts`).
    """

    reroutes_busy_washes = False

    def __init__(
        self,
        tasks: list[TaskSpec],
        config: ScriptedConfig,
        task_pool: list[TaskSpec],
        release: dict[str, int],
        gates: dict[str, tuple[tuple[str, str], ...]],
    ) -> None:
        super().__init__(tasks, config, task_pool)
        self._release = release
        self._gates = gates

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Returns the coded executor's calls at no deliberation cost, held until the release."""
        knows_everything = dataclasses.replace(request, frontier_steps=None)
        response = super().decide(knows_everything)
        release = self._release.get(response.task_id, 0)
        if self._step_index == 0 and response.task_id is not None and request.tick < release:
            hold = ToolCall("wait", (str(release - request.tick),))
            response = dataclasses.replace(response, tool_calls=[hold], raw_text="hold for release")
        for pred, furniture in self._gates.get(response.task_id, ()):
            touches = any(
                split_slot(call.args[TARGET_ARG[call.name]])[0] == furniture
                for call in response.tool_calls
                if call.name in TARGET_ARG and call.name != "pick"
            )
            if touches and pred not in self._known_done:
                hold = ToolCall("wait", (str(self._wait_ticks),))
                response = dataclasses.replace(
                    response, tool_calls=[hold], raw_text=f"after {pred}"
                )
        return dataclasses.replace(response, prompt_tokens=0, output_tokens=0)


def solve_oracle(config: OracleConfig, scenario: Scenario, seed: int) -> OraclePlan:
    """Plans the scenario's tasks for its spawned agents. Adds tasks visible from the start.

    Raises:
        ImportError: `cpsat` was requested but ortools is not installed (the [solver] extra).
    """
    add_visible_followups(scenario)
    solved = []
    for unloads, gates in unload_variants(scenario):
        variant = dataclasses.replace(scenario, tasks=[*scenario.tasks, *unloads])
        plan = _solve(build_problem(variant, gates), config, seed)
        solved.append((plan, unloads, gates))
    best, unloads, gates = min(solved, key=_planned_makespan)
    scenario.tasks.extend(unloads)
    proven = all(plan.status == "OPTIMAL" for plan, _, _ in solved)
    status = best.status if best.status != "OPTIMAL" or proven else "FEASIBLE"
    bound = min(plan.bound for plan, _, _ in solved)
    return dataclasses.replace(best, bound=bound, status=status, gates=gates)


def _planned_makespan(solved: tuple[OraclePlan, list[TaskSpec], dict]) -> int:
    return solved[0].planned


def _solve(problem: PlanProblem, config: OracleConfig, seed: int) -> OraclePlan:
    """The heuristic's plan, improved by CP-SAT when the config asks for it."""
    rng = random.Random(f"{seed}:oracle")
    plan = heuristic_plan(problem, rng, config.restarts, config.max_passes)
    timing = schedule(problem, plan)
    bound = simple_lower_bound(problem)
    if config.method == "heuristic":
        release = release_ticks(problem, plan, timing.starts)
        return OraclePlan(plan, timing.makespan, bound, "heuristic", "HEURISTIC", release)
    # Imported here because ortools is an optional extra: the sim and the heuristic never need it.
    from allocation.oracle_cpsat import solve_cpsat

    return solve_cpsat(problem, plan, timing.makespan, bound, config, seed)


def unload_variants(
    scenario: Scenario,
) -> list[tuple[list[TaskSpec], dict[str, tuple[tuple[str, str], ...]]]]:
    """The unloads and gates for each order of the washes that share a dishwasher.

    Washed dishes stay in the machine until someone unloads them, and a wash run on top of them
    would show them to its agent, who must then announce an unload (`sim.triggers`). C0 plans the
    clean way: between consecutive washes on a machine, the earlier batch is unloaded (by any
    agent) before the next is loaded. Which batch precedes which is a choice, so every order is
    tried (up to `ORACLE_MAX_MACHINE_ORDERS` combinations). An unload's id is the trigger's, so the
    replay's own sighting of the dishes finds the task C0 already planned.

    Returns:
        (unload tasks, gates) per order combination; one empty variant when no machine is shared.
    """
    machines: dict[str, list[TaskSpec]] = {}
    for task in scenario.tasks:
        if task.kind in WASH_KINDS:
            machine = next(step.args[0] for step in task.steps if step.name == "start")
            machines.setdefault(machine, []).append(task)
    shared = {machine: washes for machine, washes in sorted(machines.items()) if len(washes) > 1}
    beliefs = omniscient_beliefs(scenario.world)
    orders = [list(itertools.permutations(washes)) for washes in shared.values()]
    variants = []
    for combination in itertools.islice(itertools.product(*orders), ORACLE_MAX_MACHINE_ORDERS):
        unloads = []
        gates: dict[str, tuple[tuple[str, str], ...]] = {}
        for machine, order in zip(shared, combination, strict=True):
            dest = unload_dest(machine, beliefs)
            for washed, following in zip(order, order[1:], strict=False):
                dishes = [goal.obj_id for goal in washed.goals if goal.relation == "washed"]
                entry = {**unload_entry(machine, dishes, dest), "after": [washed.task_id]}
                unload = task_from_entry(entry, scenario.templates)
                unloads.append(unload)
                gates[following.task_id] = ((unload.task_id, machine),)
        variants.append((unloads, gates))
    return variants


def release_ticks(problem: PlanProblem, plan: Plan, starts: dict[str, int]) -> dict[str, int]:
    """The tick each agent should set off for each task to arrive at the task's planned start."""
    release: dict[str, int] = {}
    for agent, tasks in plan.items():
        previous = None
        for task in tasks:
            if previous is None:
                travel = problem.first_travel[agent, task]
            else:
                travel = problem.transition[agent, previous, task]
            release[task] = max(0, starts[task] - travel)
            previous = task
    return release


def retime(problem: PlanProblem, plan: Plan, order: list[str]) -> Schedule:
    """Earliest starts that process tasks in a given order, for each agent and piece of furniture.

    Unlike `schedule`, which chooses its own order on shared furniture, this keeps a solver's order
    and only removes idle time.
    """
    owner = {task: agent for agent, tasks in plan.items() for task in tasks}
    free: dict[str, int] = {}
    last: dict[str, str] = {}
    furniture_free: dict[str, int] = {}
    starts: dict[str, int] = {}
    completions: dict[str, int] = {}
    for task in order:
        agent = owner[task]
        profile = problem.profiles[agent, task]
        if agent in last:
            arrival = free[agent] + problem.transition[agent, last[agent], task]
        else:
            arrival = problem.first_travel[agent, task]
        start = max([arrival, *(completions[pred] for pred in problem.after[task])])
        for pred, furniture in problem.gates.get(task, ()):
            start = max(start, completions[pred] - first_hold(profile, furniture))
        for span in profile.locks:
            start = max(start, furniture_free.get(span.furniture, 0) - span.start)
        starts[task] = start
        completions[task] = start + profile.completion
        free[agent] = start + profile.busy
        last[agent] = task
        for span in profile.locks:
            furniture_free[span.furniture] = max(
                furniture_free.get(span.furniture, 0), start + span.end
            )
    return Schedule(starts, completions, max(completions.values(), default=0))


def add_visible_followups(scenario: Scenario) -> None:
    """Adds the tasks triggers would create from full knowledge at the start (a stray dirty dish).

    Other conditions discover these. Tasks that arise later (shards, racked dishes) do not exist
    for the oracle, which runs without drops.
    """
    known = {(task.kind, goal.obj_id) for task in scenario.tasks for goal in task.goals}
    for entry in find_triggered_tasks(omniscient_beliefs(scenario.world), known):
        scenario.tasks.append(task_from_entry(entry, scenario.templates))


def build_problem(
    scenario: Scenario, gates: dict[str, tuple[tuple[str, str], ...]] | None = None
) -> PlanProblem:
    """The planning model for a scenario's tasks and spawned agents, with optional gates."""
    world = scenario.world
    planner = PathPlanner(world)
    task_ids = tuple(task.task_id for task in scenario.tasks)
    agent_ids = tuple(spec.agent_id for spec in scenario.agents)
    walks = {task.task_id: _walk(task, world, planner) for task in scenario.tasks}
    first_targets = {task.task_id: _first_targets(task, world) for task in scenario.tasks}
    profiles: dict[tuple[str, str], TaskProfile] = {}
    first_travel: dict[tuple[str, str], int] = {}
    transition: dict[tuple[str, str, str], int] = {}
    for spec in scenario.agents:
        for task in scenario.tasks:
            if spec.card.hands < _hands_needed(task, world):
                continue
            profiles[spec.agent_id, task.task_id] = _profile(
                task, walks[task.task_id][0], world, spec.card.speed
            )
            steps = planner.steps_to_any(spec.cell, first_targets[task.task_id])
            first_travel[spec.agent_id, task.task_id] = travel_ticks(steps, spec.card.speed)
        for first, second in itertools.permutations(task_ids, 2):
            steps = planner.steps_to_any(walks[first][1], first_targets[second])
            transition[spec.agent_id, first, second] = travel_ticks(steps, spec.card.speed)
    after = {
        task.task_id: tuple(pred for pred in task.after if pred in task_ids)
        for task in scenario.tasks
    }
    return PlanProblem(agent_ids, task_ids, profiles, first_travel, transition, after, gates or {})


def _target_cells(name: str, world: World) -> tuple[Cell, ...]:
    """Cells of a step's target: an object or slot, or any furniture of a named type."""
    base = split_slot(name)[0]
    if base in world.objects:
        return world.objects[base].cells()
    return tuple(
        cell for obj in world.objects.values() if obj.obj_type == base for cell in obj.cells()
    )


def _step_targets(task: TaskSpec, step: ToolCall, world: World) -> tuple[Cell, ...]:
    """Cells a step's target will occupy when the task runs.

    An unload picks dishes out of its dishwasher, wherever those dishes lie before their wash.
    """
    if task.kind == UNLOAD_KIND and step.name == "pick":
        return world.objects[task.steps[0].args[0]].cells()
    return _target_cells(step.args[TARGET_ARG[step.name]], world)


def _first_targets(task: TaskSpec, world: World) -> tuple[Cell, ...]:
    """Where a task begins: its first step that has a target.

    Raises:
        ValueError: If no step has a target, so the model cannot place the task.
    """
    for step in task.steps:
        if step.name in TARGET_ARG:
            return _step_targets(task, step, world)
    raise ValueError(f"Task {task.task_id} has no step with a target to plan around")


def _walk(task: TaskSpec, world: World, planner: PathPlanner) -> tuple[list[int], Cell]:
    """Static steps of each move within a task, and the cell the task ends on.

    The walk starts from a fixed free cell next to the first target, so a task's duration does not
    depend on the direction it was approached from.

    Returns:
        (steps before each targeted step after the first, end cell).
    """
    blocked = world.blocked_cells()
    first = _first_targets(task, world)
    cell = min(
        (target[0] + dx, target[1] + dy)
        for target in first
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (target[0] + dx, target[1] + dy) in world.rooms
        and (target[0] + dx, target[1] + dy) not in blocked
        and planner.steps_to_any((target[0] + dx, target[1] + dy), first) is not None
    )
    moves = []
    targeted = [step for step in task.steps if step.name in TARGET_ARG]
    for step in targeted[1:]:
        targets = _step_targets(task, step, world)
        moves.append(planner.steps_to_any(cell, targets))
        cell = _arrival_cell(cell, targets, planner)
    return moves, cell


def _arrival_cell(start: Cell, targets: tuple[Cell, ...], planner: PathPlanner) -> Cell:
    """Where the static shortest path from `start` to next to `targets` ends."""
    cell = start
    next_cell = planner.next_cell(cell, targets)
    while next_cell is not None:
        cell = next_cell
        next_cell = planner.next_cell(cell, targets)
    return cell


def _hands_needed(task: TaskSpec, world: World) -> int:
    """The most hands any object the task picks up needs."""
    return max(
        (
            world.objects[step.args[0]].hands_required
            for step in task.steps
            if step.name == "pick" and step.args[0] in world.objects
        ),
        default=1,
    )


def _profile(task: TaskSpec, moves: list[int], world: World, speed: int) -> TaskProfile:
    """Walks a task's steps at an agent's speed, timing its work and its exclusive furniture.

    The first targeted step's travel belongs to the transition, not the task. A later targeted step
    costs at least one tick of travel because the executor always issues `move_to` before it.

    A task completes when its last goal-making step ends (or its wash does), which can be before
    the agent is free: packing is done when the item lands in the pantry, not after the walk back
    to unlock the table. A wash holds its dishwasher from the first loading trip until its cycle
    ends: an appliance cannot be opened mid-cycle, and another task's cycle started mid-load would
    split this task (`sim.loop.handle_cycle_starts`).
    """
    goal_objects = {goal.obj_id for goal in task.goals}
    clock = 0
    goals_met = 0
    open_locks: dict[str, int] = {}
    loading: dict[str, int] = {}
    spans: list[LockSpan] = []
    remaining_moves = iter(moves)
    seen_target = False
    for step in task.steps:
        if step.name in TARGET_ARG:
            if seen_target:
                clock += travel_ticks(next(remaining_moves), speed)
            seen_target = True
        base = split_slot(step.args[0])[0] if step.args else ""
        began = clock
        clock += PRIMITIVE_TICKS[step.name]
        if step.name == "lock":
            open_locks[base] = began
        if step.name == "unlock":
            spans.append(LockSpan(base, open_locks.pop(base), clock))
        is_appliance = base in world.objects and world.objects[base].washing_time > 0
        if step.name == "open" and is_appliance and base not in open_locks:
            loading.setdefault(base, began)
        if step.name in GOAL_MAKING_STEPS and base in goal_objects:
            goals_met = clock
        if step.name == "start":
            cycle_end = clock + world.objects[base].washing_time
            locked = open_locks.pop(base, began)
            spans.append(LockSpan(base, loading.pop(base, locked), cycle_end))
            goals_met = max(goals_met, cycle_end)
    busy = clock + PRIMITIVE_TICKS["report"]
    spans += [LockSpan(base, began, busy) for base, began in sorted(open_locks.items())]
    return TaskProfile(busy, goals_met or busy, tuple(spans))


def schedule(problem: PlanProblem, plan: Plan) -> Schedule | None:
    """Times a plan: each task starts as early as its agent, predecessors, and furniture allow.

    The agent whose next task can start soonest goes first. Predecessors absent from the plan (a
    partial plan during construction) are ignored.

    Returns:
        The schedule, or None if the plan's orders and precedence form a cycle.
    """
    planned = {task for tasks in plan.values() for task in tasks}
    next_index = dict.fromkeys(problem.agents, 0)
    free = dict.fromkeys(problem.agents, 0)
    last: dict[str, str] = {}
    starts: dict[str, int] = {}
    completions: dict[str, int] = {}
    furniture_busy: dict[str, list[tuple[int, int]]] = {}
    while len(starts) < len(planned):
        best: tuple[int, str, str] | None = None
        for agent in problem.agents:
            if next_index[agent] >= len(plan[agent]):
                continue
            task = plan[agent][next_index[agent]]
            predecessors = [pred for pred in problem.after[task] if pred in planned]
            gates = [gate for gate in problem.gates.get(task, ()) if gate[0] in planned]
            if any(pred not in completions for pred in [*predecessors, *(g[0] for g in gates)]):
                continue
            if agent in last:
                arrival = free[agent] + problem.transition[agent, last[agent], task]
            else:
                arrival = problem.first_travel[agent, task]
            profile = problem.profiles[agent, task]
            ready = max(
                [
                    arrival,
                    *(completions[pred] for pred in predecessors),
                    *(
                        completions[pred] - first_hold(profile, furniture)
                        for pred, furniture in gates
                    ),
                ]
            )
            start = _first_fit(ready, profile.locks, furniture_busy)
            if best is None or start < best[0]:
                best = (start, agent, task)
        if best is None:
            return None
        start, agent, task = best
        profile = problem.profiles[agent, task]
        starts[task] = start
        completions[task] = start + profile.completion
        free[agent] = start + profile.busy
        last[agent] = task
        next_index[agent] += 1
        for span in profile.locks:
            furniture_busy.setdefault(span.furniture, []).append(
                (start + span.start, start + span.end)
            )
    return Schedule(starts, completions, max(completions.values(), default=0))


def _first_fit(
    ready: int, locks: tuple[LockSpan, ...], furniture_busy: dict[str, list[tuple[int, int]]]
) -> int:
    """The earliest start at or after `ready` at which every lock span finds its furniture free."""
    start = ready
    moved = True
    while moved:
        moved = False
        for span in locks:
            for began, ended in furniture_busy.get(span.furniture, ()):
                if start + span.start < ended and began < start + span.end:
                    start = ended - span.start
                    moved = True
    return start


def score_plan(problem: PlanProblem, plan: Plan) -> Score:
    """A plan's score: makespan, then total completion time."""
    timing = schedule(problem, plan)
    if timing is None:
        return NO_SCHEDULE
    return float(timing.makespan), float(sum(timing.completions.values()))


def heuristic_plan(
    problem: PlanProblem, rng: random.Random, restarts: int, max_passes: int
) -> Plan:
    """The best plan over seeded starts of cheapest insertion followed by local search."""
    best_plan: Plan = {}
    best_score = NO_SCHEDULE
    for _ in range(restarts):
        order = list(problem.tasks)
        rng.shuffle(order)
        plan = _insert_all(problem, _predecessors_first(problem, order))
        plan, score = _improve(problem, plan, max_passes)
        if score < best_score:
            best_plan, best_score = plan, score
    return best_plan


def _predecessors_first(problem: PlanProblem, order: list[str]) -> list[str]:
    """The order, changed only so that every task comes after its predecessors and gates.

    Inserting a gated wash before its unload exists could fix the two washes' order on one agent
    the wrong way round, leaving the unload nowhere to go.
    """
    placed: list[str] = []
    remaining = list(order)
    while remaining:
        for task in remaining:
            needs = [*problem.after[task], *(pred for pred, _ in problem.gates.get(task, ()))]
            if all(pred in placed for pred in needs):
                placed.append(task)
                remaining.remove(task)
                break
        else:
            raise ValueError(f"Precedence cycle among {remaining}")
    return placed


def _insert_all(problem: PlanProblem, order: list[str]) -> Plan:
    """Adds tasks one at a time, each at the position that keeps the score lowest."""
    plan: Plan = {agent: [] for agent in problem.agents}
    for task in order:
        plan, _ = _best_insertion(problem, plan, task)
    return plan


def _best_insertion(problem: PlanProblem, plan: Plan, task: str) -> tuple[Plan, Score]:
    """The best way to add one task to a plan.

    Raises:
        ValueError: If no agent can do the task or no position gives a valid schedule.
    """
    best_plan: Plan | None = None
    best_score = NO_SCHEDULE
    for agent in problem.agents:
        if (agent, task) not in problem.profiles:
            continue
        for position in range(len(plan[agent]) + 1):
            tasks = plan[agent]
            trial = {**plan, agent: [*tasks[:position], task, *tasks[position:]]}
            score = score_plan(problem, trial)
            if score < best_score:
                best_plan, best_score = trial, score
    if best_plan is None:
        raise ValueError(f"No agent can do task {task}")
    return best_plan, best_score


def _improve(problem: PlanProblem, plan: Plan, max_passes: int) -> tuple[Plan, Score]:
    """Local search: moves single tasks and swaps pairs while the score improves."""
    score = score_plan(problem, plan)
    for _ in range(max_passes):
        improved = False
        for task in problem.tasks:
            without = {agent: [t for t in tasks if t != task] for agent, tasks in plan.items()}
            trial, trial_score = _best_insertion(problem, without, task)
            if trial_score < score:
                plan, score = trial, trial_score
                improved = True
        for first, second in itertools.combinations(problem.tasks, 2):
            trial = _swapped(problem, plan, first, second)
            if trial is None:
                continue
            trial_score = score_plan(problem, trial)
            if trial_score < score:
                plan, score = trial, trial_score
                improved = True
        if not improved:
            break
    return plan, score


def _swapped(problem: PlanProblem, plan: Plan, first: str, second: str) -> Plan | None:
    """The plan with two tasks' places exchanged, or None if an agent cannot do its new task."""
    swap = {first: second, second: first}
    trial = {agent: [swap.get(task, task) for task in tasks] for agent, tasks in plan.items()}
    for agent, tasks in trial.items():
        if any((agent, task) not in problem.profiles for task in tasks):
            return None
    return trial


def simple_lower_bound(problem: PlanProblem) -> int:
    """A lower bound any plan's makespan must meet in the model, without a solver.

    The largest of: the least total work shared evenly among the agents, the longest precedence
    chain at each task's fastest completion, and each piece of furniture's exclusive time.
    """
    fastest_completion: dict[str, int] = {}
    least_work = 0
    for task in problem.tasks:
        options = [agent for agent in problem.agents if (agent, task) in problem.profiles]
        fastest_completion[task] = min(
            least_arrival(problem, agent, task) + problem.profiles[agent, task].completion
            for agent in options
        )
        least_work += min(
            least_arrival(problem, agent, task) + problem.profiles[agent, task].busy
            for agent in options
        )
    chain: dict[str, int] = {}
    for task in _precedence_order(problem):
        before = max((chain[pred] for pred in problem.after[task]), default=0)
        chain[task] = before + _least_completion_after_arrival(problem, task)
        chain[task] = max(chain[task], fastest_completion[task])
    longest = max(chain.values(), default=0)
    furniture_floor = 0
    earliest = earliest_starts(problem)
    for furniture in furniture_names(problem):
        hold = 0
        for task in problem.tasks:
            holds = [
                held_ticks(profile, furniture)
                for (_, other), profile in problem.profiles.items()
                if other == task
            ]
            hold += min(holds)
        floor = furniture_head(problem, furniture, earliest) + hold
        furniture_floor = max(furniture_floor, floor - furniture_overhang(problem, furniture))
    return max(longest, math.ceil(least_work / len(problem.agents)), furniture_floor)


def furniture_names(problem: PlanProblem) -> list[str]:
    """Every piece of furniture any task can hold exclusively."""
    return sorted(
        {span.furniture for profile in problem.profiles.values() for span in profile.locks}
    )


def first_hold(profile: TaskProfile, furniture: str) -> int:
    """Ticks after a task starts at which it first holds a piece of furniture (0 if never)."""
    return min((span.start for span in profile.locks if span.furniture == furniture), default=0)


def held_ticks(profile: TaskProfile, furniture: str) -> int:
    """Total ticks a task, done this way, holds one piece of furniture."""
    return sum(span.end - span.start for span in profile.locks if span.furniture == furniture)


def earliest_starts(problem: PlanProblem) -> dict[str, int]:
    """The soonest each task could begin, from travel and precedence alone."""
    earliest: dict[str, int] = {}
    for task in _precedence_order(problem):
        agents = [agent for agent in problem.agents if (agent, task) in problem.profiles]
        arrive = min(problem.first_travel[agent, task] for agent in agents)
        after = [
            earliest[pred] + _least_completion_after_arrival(problem, pred)
            for pred in problem.after[task]
        ]
        earliest[task] = max([arrive, *after])
    return earliest


def furniture_head(problem: PlanProblem, furniture: str, earliest: dict[str, int]) -> int:
    """The soonest any task could begin holding a piece of furniture."""
    return min(
        earliest[task] + span.start
        for (_, task), profile in problem.profiles.items()
        for span in profile.locks
        if span.furniture == furniture
    )


def furniture_overhang(problem: PlanProblem, furniture: str) -> int:
    """How far past its task's completion a hold can run (unlocking after the goal is met).

    Exclusive holds fit between `furniture_head` and the last hold's end, and that end can pass the
    makespan by at most this much, so head + total hold - overhang is a valid bound on the makespan.
    """
    return max(
        max(span.end - profile.completion, 0)
        for profile in problem.profiles.values()
        for span in profile.locks
        if span.furniture == furniture
    )


def least_arrival(problem: PlanProblem, agent: str, task: str) -> int:
    """The fewest ticks an agent can spend reaching a task, from its start or any other task."""
    arrivals = [problem.first_travel[agent, task]]
    arrivals += [problem.transition[agent, other, task] for other in problem.tasks if other != task]
    return min(arrivals)


def _least_completion_after_arrival(problem: PlanProblem, task: str) -> int:
    """A task's fastest completion, counted from the moment an agent has arrived."""
    return min(
        problem.profiles[agent, task].completion
        for agent in problem.agents
        if (agent, task) in problem.profiles
    )


def _precedence_order(problem: PlanProblem) -> list[str]:
    """Tasks with every task after its predecessors."""
    ordered: list[str] = []
    placed: set[str] = set()
    remaining = list(problem.tasks)
    while remaining:
        ready = [task for task in remaining if set(problem.after[task]) <= placed]
        if not ready:
            raise ValueError(f"Precedence cycle among {remaining}")
        ordered += ready
        placed |= set(ready)
        remaining = [task for task in remaining if task not in placed]
    return ordered
