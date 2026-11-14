"""Headroom pilot: python -m experiments.headroom configs/pilot_headroom.yaml.

Answers "is this mission hard enough for allocation to matter?" before any allocator is compared on
it. Static assignments of the mission tasks to agents are run with the scripted executor (all of
them when there are few enough, otherwise a seeded random sample), and the spread of their
makespans (best, median, worst) is compared with the greedy baseline and with the mission's own
scripted assignment. If the median assignment is barely slower than the best, no allocator can show
a gain here and the mission needs more tasks, agents, or space.

The best static assignment is an upper bound on what a static oracle could do with this executor;
the real oracle (C0) can also reorder tasks, so it may do better still.
"""

import argparse
import dataclasses
import itertools
import random
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from allocation.greedy import GreedyPolicy
from allocation.scripted import ScriptedPolicy
from experiments.run_single import load_episode_scenario
from sim.agent import Policy
from sim.config import EpisodeConfig, FaultConfig, load_config
from sim.logger import EpisodeLogger
from sim.loop import run_episode
from sim.scenario import load_scenario


def run_assignment(config: EpisodeConfig, assignment: dict[str, list[str]] | None) -> int:
    """Runs one episode; `assignment` None means the greedy baseline. Returns the last tick."""
    scenario = load_episode_scenario(config)
    tasks_by_id = {task.task_id: task for task in scenario.tasks}
    policies: dict[str, Policy] = {}
    for spec in scenario.agents:
        if assignment is None:
            policies[spec.agent_id] = GreedyPolicy(
                config.scripted, scenario.tasks, config.task_visibility == "posted"
            )
        else:
            policies[spec.agent_id] = ScriptedPolicy(
                [tasks_by_id[task_id] for task_id in assignment[spec.agent_id]],
                config.scripted,
                scenario.tasks,
            )
    with EpisodeLogger(Path(tempfile.mkdtemp(prefix="headroom_"))) as logger:
        return run_episode(config, scenario, policies, logger)


def static_assignments(
    agent_ids: list[str], task_ids: list[str], pilot: dict
) -> list[dict[str, list[str]]]:
    """Every assignment (each agent keeps mission order), or a seeded sample when too many."""
    total = len(agent_ids) ** len(task_ids)
    if total <= pilot["max_enumerated"]:
        owner_lists = list(itertools.product(agent_ids, repeat=len(task_ids)))
    else:
        rng = random.Random(f"{pilot['seed']}:headroom")
        owner_lists = [[rng.choice(agent_ids) for _ in task_ids] for _ in range(pilot["samples"])]
    assignments = []
    for owners in owner_lists:
        assignment = {agent_id: [] for agent_id in agent_ids}
        for task_id, owner in zip(task_ids, owners, strict=True):
            assignment[owner].append(task_id)
        assignments.append(assignment)
    return assignments


def report(episode: Path, pilot: dict) -> None:
    """Runs and prints the headroom numbers for one episode config."""
    config = dataclasses.replace(
        load_config(episode), faults=FaultConfig(float(pilot["drop_prob"]))
    )
    scenario = load_scenario(config.map_path, config.mission_path)
    agent_ids = [spec.agent_id for spec in scenario.agents]
    task_ids = [task.task_id for task in scenario.tasks]
    assignments = static_assignments(agent_ids, task_ids, pilot)
    runs = [*assignments, scenario.scripted_assignment, None]
    with ProcessPoolExecutor(max_workers=pilot["workers"]) as pool:
        ticks = list(pool.map(run_assignment, [config] * len(runs), runs))
    unfinished = config.max_ticks - 1
    static = [tick for tick in ticks[: len(assignments)] if tick < unfinished]
    hand, greedy = ticks[-2], ticks[-1]
    best, median, worst = min(static), statistics.median(static), max(static)
    kind = "all" if len(assignments) == len(agent_ids) ** len(task_ids) else "sampled"
    print(f"{episode}: {len(agent_ids)} agents, {len(task_ids)} tasks, {kind} {len(assignments)}")
    unfinished_count = len(assignments) - len(static)
    print(f"  static: best {best}, median {median}, worst {worst}; unfinished {unfinished_count}")
    print(f"  mission's scripted assignment: {hand}")
    print(f"  greedy: {greedy} ({(greedy - best) / best:+.0%} vs best)")
    headroom = (median - best) / best
    verdict = "enough" if headroom >= float(pilot["min_headroom"]) else "too little"
    print(f"  headroom (median vs best): {headroom:.0%}, {verdict} for allocation to matter")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path, help="pilot config YAML")
    args = parser.parse_args()
    with args.pilot.open(encoding="utf-8") as handle:
        pilot = yaml.safe_load(handle)
    for episode in pilot["episodes"]:
        report(Path(episode), pilot)


if __name__ == "__main__":
    main()
