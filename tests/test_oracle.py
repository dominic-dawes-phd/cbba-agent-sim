import dataclasses
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from allocation.oracle import (
    LockSpan,
    PlanProblem,
    TaskProfile,
    build_problem,
    heuristic_plan,
    release_ticks,
    retime,
    schedule,
    simple_lower_bound,
    solve_oracle,
)
from experiments.analyze import attach_oracle_ratios, run_metrics
from experiments.generate_mission import _nearest
from experiments.run_single import run_config
from sim.config import FaultConfig, OracleConfig, load_config
from sim.logger import read_stream
from sim.pathing import PathPlanner
from sim.scenario import load_scenario

HOME = Path("configs/m2_home.yaml")
HEURISTIC = OracleConfig("heuristic", 2, 4, 10.0, 2)
CPSAT = OracleConfig("cpsat", 2, 4, 10.0, 2)


def tiny_problem(locks=False):
    """Two agents, four tasks; travel is the absolute difference between task numbers."""
    agents = ("a", "b")
    tasks = ("t1", "t2", "t3", "t4")
    profiles = {}
    for agent, task in itertools.product(agents, tasks):
        number = int(task[1])
        spans = (LockSpan("shared", 0, number + 2),) if locks and number >= 3 else ()
        profiles[agent, task] = TaskProfile(number + 2, number + 2, spans)
    first_travel = {
        (agent, task): int(task[1]) * (1 if agent == "a" else 2)
        for agent, task in itertools.product(agents, tasks)
    }
    transition = {
        (agent, first, second): abs(int(first[1]) - int(second[1]))
        for agent in agents
        for first, second in itertools.permutations(tasks, 2)
    }
    after = {"t1": (), "t2": ("t1",), "t3": (), "t4": ()}
    return PlanProblem(agents, tasks, profiles, first_travel, transition, after)


def brute_force_makespan(problem):
    """The true optimum: every assignment, every order, timed by the schedule simulator."""
    best = None
    for owners in itertools.product(problem.agents, repeat=len(problem.tasks)):
        mine = {agent: [] for agent in problem.agents}
        for task, owner in zip(problem.tasks, owners, strict=True):
            mine[owner].append(task)
        orders = [list(itertools.permutations(mine[agent])) for agent in problem.agents]
        for lists in itertools.product(*orders):
            plan = {agent: list(tasks) for agent, tasks in zip(problem.agents, lists, strict=True)}
            timing = schedule(problem, plan)
            if timing is not None and (best is None or timing.makespan < best):
                best = timing.makespan
    return best


def check_plan_is_valid(problem, solved):
    """Rebuilds a solved plan's timing from its release ticks and checks every model constraint."""
    starts, completions = {}, {}
    busy_spans = {}
    for agent, tasks in solved.assignment.items():
        free = 0
        previous = None
        for task in tasks:
            assert (agent, task) in problem.profiles
            if previous is None:
                travel = problem.first_travel[agent, task]
            else:
                travel = problem.transition[agent, previous, task]
            profile = problem.profiles[agent, task]
            starts[task] = max(solved.release[task] + travel, travel)
            assert starts[task] >= free + travel
            completions[task] = starts[task] + profile.completion
            free = starts[task] + profile.busy
            previous = task
            for span in profile.locks:
                busy_spans.setdefault(span.furniture, []).append(
                    (starts[task] + span.start, starts[task] + span.end)
                )
    assert sorted(starts) == sorted(problem.tasks)
    for task in problem.tasks:
        for pred in problem.after[task]:
            assert starts[task] >= completions[pred]
    for spans in busy_spans.values():
        ordered = sorted(spans)
        for (_, first_end), (second_start, _) in zip(ordered, ordered[1:], strict=False):
            assert second_start >= first_end
    assert max(completions.values()) == solved.planned


def test_schedule_respects_precedence_and_exclusive_furniture():
    problem = tiny_problem(locks=True)
    timing = schedule(problem, {"a": ["t1", "t2"], "b": ["t3", "t4"]})
    assert timing.starts["t2"] >= timing.completions["t1"]
    first, second = sorted((timing.starts["t3"], timing.starts["t4"]))
    assert second >= first + 5  # t3 holds the shared furniture for 5 ticks; t4 waits its turn


def test_precedence_cycles_have_no_schedule():
    problem = tiny_problem()
    assert schedule(problem, {"a": ["t2", "t1"], "b": ["t3", "t4"]}) is None


def test_heuristic_is_seeded_and_never_beats_the_true_optimum():
    problem = tiny_problem()
    plan = heuristic_plan(problem, random.Random("0:oracle"), 3, 5)
    assert plan == heuristic_plan(problem, random.Random("0:oracle"), 3, 5)
    assert schedule(problem, plan).makespan >= brute_force_makespan(problem)
    assert simple_lower_bound(problem) <= brute_force_makespan(problem)


def test_retime_keeps_a_given_order_on_shared_furniture():
    problem = tiny_problem(locks=True)
    plan = {"a": ["t1", "t2"], "b": ["t4", "t3"]}
    timing = retime(problem, plan, ["t1", "t4", "t2", "t3"])
    assert timing.starts["t3"] >= timing.starts["t4"] + 6  # t4 holds the furniture first


def test_release_ticks_arrive_at_the_planned_start():
    problem = tiny_problem()
    plan = {"a": ["t1", "t2"], "b": ["t3", "t4"]}
    timing = schedule(problem, plan)
    release = release_ticks(problem, plan, timing.starts)
    assert release["t1"] == timing.starts["t1"] - problem.first_travel["a", "t1"]
    assert release["t2"] == timing.starts["t2"] - problem.transition["a", "t1", "t2"]


def test_cpsat_matches_brute_force_and_proves_its_bound():
    pytest.importorskip("ortools")
    from allocation.oracle_cpsat import solve_cpsat

    for locks in (False, True):
        problem = tiny_problem(locks)
        hint = heuristic_plan(problem, random.Random("0:oracle"), 1, 1)
        upper = schedule(problem, hint).makespan
        solved = solve_cpsat(problem, hint, upper, simple_lower_bound(problem), CPSAT, 0)
        assert solved.status == "OPTIMAL"
        assert solved.planned == solved.bound
        check_plan_is_valid(problem, solved)
        enumerated = brute_force_makespan(problem)
        if locks:
            # Enumeration times furniture by first fit, so it can only overestimate the optimum.
            assert solved.planned <= enumerated
        else:
            assert solved.planned == enumerated


def test_cpsat_is_deterministic_on_a_real_mission():
    pytest.importorskip("ortools")
    config = load_config(HOME)
    plans = [
        solve_oracle(CPSAT, load_scenario(config.map_path, config.mission_path), 0)
        for _ in range(2)
    ]
    assert plans[0] == plans[1]
    assert plans[0].bound <= plans[0].planned


def test_planning_model_gives_every_task_a_doer_and_washes_outlast_the_agent():
    config = load_config(HOME)
    problem = build_problem(load_scenario(config.map_path, config.mission_path))
    for task in problem.tasks:
        assert any((agent, task) in problem.profiles for agent in problem.agents)
    wash = problem.profiles["arm_a", "t_clean1"]
    assert wash.completion > wash.busy  # the goal needs the 60-tick cycle to end
    assert any(span.furniture == "dishwasher_1" for span in wash.locks)


def test_a_task_completes_when_its_goal_is_met_not_when_the_agent_is_free():
    # Found by a replay that beat its "optimal" plan: packing is done when the item lands in the
    # pantry, before the walk back to unlock the table.
    config = load_config(HOME)
    problem = build_problem(load_scenario(config.map_path, config.mission_path))
    pack = problem.profiles["arm_a", "t_groceries"]
    assert pack.completion < pack.busy


def test_a_wash_holds_its_dishwasher_from_the_first_loading_trip_to_the_end_of_its_cycle():
    # A dishwasher cannot be opened mid-cycle, and a cycle started mid-load splits the task.
    generated = Path("configs/generated/house_48x32_s0")
    scenario = load_scenario(generated / "map.txt", generated / "mission.yaml")
    problem = build_problem(scenario)
    task = next(task for task in scenario.tasks if task.kind == "clean_dishes")
    dishes = [goal.obj_id for goal in task.goals if goal.relation == "washed"]
    machine = next(step.args[0] for step in task.steps if step.name == "start")
    wash = problem.profiles["arm_a", task.task_id]
    holds = [span for span in wash.locks if span.furniture == machine]
    assert len(dishes) > 1 and len(holds) == 1
    washing = scenario.world.objects[machine].washing_time
    assert holds[0].end - holds[0].start > washing  # the loading trips, then the cycle
    assert holds[0].end == wash.completion


def test_oracle_replays_its_plan_and_logs_it(tmp_path):
    config = dataclasses.replace(
        load_config(HOME),
        condition="oracle",
        output_dir=tmp_path,
        oracle=HEURISTIC,
        faults=FaultConfig(0.0),
    )
    run_dir, final_tick = run_config(config)
    assert final_tick < config.max_ticks - 1
    logged = json.loads((run_dir / "oracle_plan.json").read_text())
    assert logged["method"] == "heuristic" and logged["bound"] <= logged["planned"]
    calls = read_stream(run_dir, "llm_calls")
    assert {call["latency_ticks"] for call in calls} == {0}
    metrics = run_metrics(run_dir)
    assert metrics["finished"] and metrics["oracle_bound"] == logged["bound"]
    assert metrics["solver_gap"] >= 0.0


def test_makespan_ratio_pairs_runs_with_the_oracle_at_the_same_mission_and_seed():
    runs = [
        {"condition": "oracle", "mission": "m", "seed": 0, "makespan": 100},
        {"condition": "greedy", "mission": "m", "seed": 0, "makespan": 150},
        {"condition": "greedy", "mission": "m", "seed": 1, "makespan": 150},
        {"condition": "greedy", "mission": "m", "seed": 0, "makespan": None},
    ]
    attach_oracle_ratios(runs)
    assert runs[0]["makespan_vs_oracle"] == 1.0 and runs[1]["makespan_vs_oracle"] == 1.5
    assert "makespan_vs_oracle" not in runs[2] and "makespan_vs_oracle" not in runs[3]


def test_nearest_measures_from_the_floor_around_furniture():
    # Furniture cells are not walkable. Measuring from one gave no distance and silently returned
    # the first option, which sent every wash in every generated house to the first dishwasher.
    world = load_scenario(
        Path("configs/generated/house_48x32_s0/map.txt"),
        Path("configs/generated/house_48x32_s0/mission.yaml"),
    ).world
    planner = PathPlanner(world)
    furniture = {
        obj_id: ("dishwasher", obj.cells(), "room")
        for obj_id, obj in world.objects.items()
        if obj.obj_type == "dishwasher"
    }
    house = SimpleNamespace(furniture=furniture)
    options = ["dishwasher_1", "dishwasher_2"]
    # A dishwasher is nearest to itself, and the second option must be reachable as an answer.
    assert _nearest(planner, world, "dishwasher_2", options, house) == "dishwasher_2"
    assert _nearest(planner, world, "dishwasher_1", options, house) == "dishwasher_1"


def test_a_gate_holds_a_task_until_its_predecessor_is_done_but_only_from_its_first_hold():
    problem = tiny_problem(locks=True)
    gated = dataclasses.replace(problem, gates={"t4": (("t1", "shared"),)})
    timing = schedule(gated, {"a": ["t1", "t2"], "b": ["t4", "t3"]})
    hold = gated.profiles["b", "t4"].locks[0].start
    assert timing.starts["t4"] + hold >= timing.completions["t1"]


def test_c0_plans_an_unload_between_washes_that_share_a_dishwasher(tmp_path):
    config = dataclasses.replace(
        load_config(HOME), condition="oracle", output_dir=tmp_path, oracle=HEURISTIC,
        faults=FaultConfig(0.0),
    )  # fmt: skip
    run_dir, final_tick = run_config(config)
    logged = json.loads((run_dir / "oracle_plan.json").read_text())
    planned = {task for tasks in logged["assignment"].values() for task in tasks}
    unloads = {task for task in planned if task.startswith("unload_")}
    assert unloads and len(logged["gates"]) == len(unloads)
    seen = {row["task"] for row in read_stream(run_dir, "events") if row["task"]}
    assert {task for task in seen if task.startswith("unload_")} == unloads  # nothing unplanned
    assert final_tick < config.max_ticks - 1
