"""CP-SAT for the oracle's planning model: a better plan and a proven lower bound on its makespan.

One circuit per agent orders that agent's tasks: a depot node starts it, a task the agent does not
take is a self-loop, and each used arc forces the later task to start after the earlier one plus the
travel between them. Optional intervals keep furniture exclusive. The heuristic's plan is the hint
and sets the horizon, so the solver can only improve on it.

The bound holds for the model, not the sim: the model has no bodies or doorway queues.

Limits use deterministic time and interleaved search, which OR-Tools documents as deterministic
independently of the number of workers, so results do not depend on machine speed (the spirit of
hard rule 1). This module needs ortools (the [solver] extra); nothing else imports it at load time.
"""

import math

from ortools.sat.python import cp_model

from allocation.oracle import (
    OraclePlan,
    Plan,
    PlanProblem,
    earliest_starts,
    first_hold,
    furniture_head,
    furniture_names,
    furniture_overhang,
    held_ticks,
    least_arrival,
    release_ticks,
    retime,
    schedule,
)
from sim.config import OracleConfig

DEPOT = 0


def solve_cpsat(
    problem: PlanProblem,
    hint: Plan,
    upper: int,
    lower: int,
    config: OracleConfig,
    seed: int,
) -> OraclePlan:
    """Solves the planning model, starting from a known plan.

    Args:
        problem: The planning model.
        hint: A valid plan; its makespan must be `upper`.
        upper: Makespan of the hint, the search's upper limit.
        lower: A lower bound already known.
        config: Solver limits.
        seed: The episode seed.

    Returns:
        The best plan found, with the solver's proven bound. If the solver finds nothing within
        its limit, the hint and `lower`, with the solver's status.
    """
    model = cp_model.CpModel()
    makespan = model.new_int_var(lower, upper, "makespan")
    starts = {task: model.new_int_var(0, upper, f"start_{task}") for task in problem.tasks}
    completions = {task: model.new_int_var(0, upper, f"done_{task}") for task in problem.tasks}
    present = {
        (agent, task): model.new_bool_var(f"{agent}_does_{task}")
        for agent in problem.agents
        for task in problem.tasks
    }
    for task in problem.tasks:
        model.add_exactly_one(present[agent, task] for agent in problem.agents)
        model.add(makespan >= completions[task])
        for pred in problem.after[task]:
            model.add(starts[task] >= completions[pred])

    furniture_intervals: dict[str, list[cp_model.IntervalVar]] = {}
    arcs_by_agent = {}
    for agent in problem.agents:
        arcs_by_agent[agent] = _add_agent(model, problem, agent, starts, present)
        agent_intervals = []
        load = []
        for task in problem.tasks:
            profile = problem.profiles.get((agent, task))
            if profile is None:
                model.add(present[agent, task] == 0)
                continue
            doing = present[agent, task]
            model.add(completions[task] >= starts[task] + profile.completion).only_enforce_if(doing)
            for pred, furniture in problem.gates.get(task, ()):
                hold = starts[task] + first_hold(profile, furniture)
                model.add(hold >= completions[pred]).only_enforce_if(doing)
            agent_intervals.append(
                model.new_optional_fixed_size_interval_var(
                    starts[task], profile.busy, doing, f"busy_{agent}_{task}"
                )
            )
            load.append(doing * (profile.busy + least_arrival(problem, agent, task)))
            for span in profile.locks:
                furniture_intervals.setdefault(span.furniture, []).append(
                    model.new_optional_fixed_size_interval_var(
                        starts[task] + span.start,
                        span.end - span.start,
                        doing,
                        f"lock_{agent}_{task}_{span.furniture}",
                    )
                )
        model.add_no_overlap(agent_intervals)
        # Redundant, but it gives the solver a much better bound on a min-max objective.
        model.add(sum(load) <= makespan)
    for intervals in furniture_intervals.values():
        model.add_no_overlap(intervals)
    # Redundant, but optional intervals alone say little about a machine several agents could use:
    # all its exclusive time has to fit between the first possible hold and the mission's end.
    earliest = earliest_starts(problem)
    for furniture in furniture_names(problem):
        total_hold = sum(
            present[agent, task] * held_ticks(profile, furniture)
            for (agent, task), profile in problem.profiles.items()
        )
        head = furniture_head(problem, furniture, earliest)
        model.add(head + total_hold - furniture_overhang(problem, furniture) <= makespan)

    _add_hints(model, problem, hint, upper, makespan, starts, completions, present, arcs_by_agent)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = config.deterministic_time
    solver.parameters.num_workers = config.workers
    solver.parameters.interleave_search = True
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        release = release_ticks(problem, hint, schedule(problem, hint).starts)
        return OraclePlan(hint, upper, lower, "cpsat", status_name, release)
    plan: Plan = {}
    for agent in problem.agents:
        taken = [
            (solver.value(starts[task]), task)
            for task in problem.tasks
            if solver.boolean_value(present[agent, task])
        ]
        plan[agent] = [task for _, task in sorted(taken)]
    solved_starts = {task: solver.value(starts[task]) for task in problem.tasks}
    solved_value = solver.value(makespan)
    # Remove the idle time the objective leaves on non-critical tasks. Keeping the solver's order
    # by task start can in rare cases reorder lock spans; then its own timing stands.
    order = sorted(problem.tasks, key=solved_starts.__getitem__)
    retimed = retime(problem, plan, order)
    if retimed.makespan <= solved_value:
        solved_starts, solved_value = retimed.starts, retimed.makespan
    bound = max(lower, math.ceil(solver.best_objective_bound))
    release = release_ticks(problem, plan, solved_starts)
    return OraclePlan(plan, solved_value, bound, "cpsat", status_name, release)


def _add_agent(
    model: cp_model.CpModel,
    problem: PlanProblem,
    agent: str,
    starts: dict[str, cp_model.IntVar],
    present: dict[tuple[str, str], cp_model.IntVar],
) -> dict[tuple[int, int], cp_model.IntVar]:
    """One agent's route: a circuit through the depot and the tasks it takes, timed along arcs.

    Returns:
        The arc literals by (from node, to node), for hints.
    """
    nodes = {task: index + 1 for index, task in enumerate(problem.tasks)}
    arcs = {(DEPOT, DEPOT): model.new_bool_var(f"{agent}_idle")}
    for task, node in nodes.items():
        profile = problem.profiles.get((agent, task))
        if profile is None:
            continue
        leaves = model.new_bool_var(f"{agent}_starts_{task}")
        arcs[DEPOT, node] = leaves
        arcs[node, DEPOT] = model.new_bool_var(f"{agent}_ends_{task}")
        model.add(starts[task] >= problem.first_travel[agent, task]).only_enforce_if(leaves)
        for other, other_node in nodes.items():
            if other == task or (agent, other) not in problem.profiles:
                continue
            follows = model.new_bool_var(f"{agent}_{task}_then_{other}")
            arcs[node, other_node] = follows
            gap = profile.busy + problem.transition[agent, task, other]
            model.add(starts[other] >= starts[task] + gap).only_enforce_if(follows)
    circuit = [(tail, head, literal) for (tail, head), literal in arcs.items()]
    circuit += [(node, node, ~present[agent, task]) for task, node in nodes.items()]
    model.add_circuit(circuit)
    return arcs


def _add_hints(
    model: cp_model.CpModel,
    problem: PlanProblem,
    hint: Plan,
    upper: int,
    makespan: cp_model.IntVar,
    starts: dict[str, cp_model.IntVar],
    completions: dict[str, cp_model.IntVar],
    present: dict[tuple[str, str], cp_model.IntVar],
    arcs_by_agent: dict[str, dict[tuple[int, int], cp_model.IntVar]],
) -> None:
    """Starts the search from a known plan."""
    timing = schedule(problem, hint)
    nodes = {task: index + 1 for index, task in enumerate(problem.tasks)}
    model.add_hint(makespan, upper)
    for task in problem.tasks:
        model.add_hint(starts[task], timing.starts[task])
        model.add_hint(completions[task], timing.completions[task])
    for agent, tasks in hint.items():
        route = [DEPOT, *(nodes[task] for task in tasks), DEPOT]
        used = set(zip(route, route[1:], strict=False))
        for arc, literal in arcs_by_agent[agent].items():
            model.add_hint(literal, int(arc in used))
        for task in problem.tasks:
            model.add_hint(present[agent, task], int(task in tasks))
