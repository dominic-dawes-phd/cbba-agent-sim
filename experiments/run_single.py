"""Runs one episode from a config file: python -m experiments.run_single configs/<file>.yaml.

Add `--run-name <experiment>` for a run to keep. It goes to
`runs/<experiment>/<env>/<name>/<YYYYMMDD-HHMMSS>/seed_<n>/` and never overwrites an earlier launch.
Without it, the run goes to `runs/<name>/seed_<n>/`, replacing the last one.
"""

import argparse
import dataclasses
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from allocation.cbba import CbbaAllocator
from allocation.dialogue import DialogueAllocator, DialogueRounds
from allocation.greedy import GreedyAllocator, GreedyPolicy
from allocation.llm_policy import LLMPolicy
from allocation.oracle import OraclePlan, OraclePolicy, solve_oracle
from allocation.scripted import ScriptedPolicy
from experiments.analyze import write_metrics
from experiments.console import (
    configure_console,
    print_banner,
    print_episode_header,
    print_episode_summary,
    run_checks,
)
from llm.client import LLMClient, OpenAIChatClient, model_file_hash
from sim.agent import Policy
from sim.config import ORACLE_PLAN_FILE, RUN_STAMP_FORMAT, EpisodeConfig, load_config
from sim.logger import EpisodeLogger
from sim.loop import run_episode
from sim.scenario import Scenario, draw_spawns, load_scenario

# Conditions that run with the LLM executor (executor: llm). The rest are coded only.
# C2 (numeric_cbba) and C3 (llm_cbba) exist only with it.
LLM_CONDITIONS = ("greedy", "numeric_cbba", "llm_cbba", "llm_only")
# Conditions that run with the coded executor (executor: coded).
CODED_CONDITIONS = ("scripted", "greedy", "oracle")


def load_episode_scenario(config: EpisodeConfig) -> Scenario:
    """Loads the config's map and mission and applies its spawn mode.

    Uniform spawns use their own seeded stream (`<seed>:spawn`), so every condition run at the same
    seed starts from the same cells.
    """
    scenario = load_scenario(config.map_path, config.mission_path)
    if config.spawn.mode == "uniform":
        scenario.agents = draw_spawns(
            scenario.world,
            scenario.agents,
            random.Random(f"{config.seed}:spawn"),
            config.spawn.min_separation,
        )
    return scenario


def build_policies(
    config: EpisodeConfig,
    scenario: Scenario,
    plan: OraclePlan | None = None,
    client: LLMClient | None = None,
) -> dict[str, Policy]:
    """One policy per agent for the config's condition. Every agent runs the same condition.

    Args:
        config: The episode config.
        scenario: The loaded scenario with its spawned agents.
        plan: The solved plan when the condition is "oracle".
        client: The shared model client when the executor is "llm".

    Raises:
        ValueError: An unknown condition, or one that has no LLM-executor version.
    """
    if config.executor == "llm" and config.condition not in LLM_CONDITIONS:
        raise ValueError(f"Condition {config.condition!r} has no LLM executor")
    tasks_by_id = {task.task_id: task for task in scenario.tasks}
    policies: dict[str, Policy] = {}
    rounds = DialogueRounds(config.dialogue, config.channel.comm_range)  # C1's shared rounds
    for slot, spec in enumerate(scenario.agents):
        posted = config.task_visibility == "posted"
        if config.executor == "llm" and config.condition == "greedy":
            allocator = GreedyAllocator(scenario.tasks, posted)
            policies[spec.agent_id] = LLMPolicy(slot, allocator, client, config.llm)
        elif config.condition == "llm_only":
            allocator = DialogueAllocator(
                spec.agent_id,
                slot,
                rounds,
                scenario.tasks,
                client,
                config.llm,
                config.latency,
                config.dialogue,
                posted,
            )
            policies[spec.agent_id] = LLMPolicy(slot, allocator, client, config.llm)
        elif config.condition in ("numeric_cbba", "llm_cbba"):
            allocator = CbbaAllocator(
                spec.agent_id,
                scenario.tasks,
                config.cbba,
                posted,
                config.condition == "llm_cbba",
            )
            policies[spec.agent_id] = LLMPolicy(slot, allocator, client, config.llm)
        elif config.condition == "oracle":
            assigned = plan.assignment[spec.agent_id]
            policies[spec.agent_id] = OraclePolicy(
                [tasks_by_id[task_id] for task_id in assigned],
                config.scripted,
                scenario.tasks,
                plan.release,
                plan.gates,
            )
        elif config.condition == "scripted":
            assigned = scenario.scripted_assignment.get(spec.agent_id, [])
            # A fixed assignment is agreed in advance, so teammates never adopt each other's tasks.
            reserved = frozenset(
                task_id
                for agent_id, task_ids in scenario.scripted_assignment.items()
                if agent_id != spec.agent_id
                for task_id in task_ids
            )
            policies[spec.agent_id] = ScriptedPolicy(
                [tasks_by_id[task_id] for task_id in assigned],
                config.scripted,
                scenario.tasks,
                reserved,
            )
        elif config.condition == "greedy":
            policies[spec.agent_id] = GreedyPolicy(
                config.scripted, scenario.tasks, config.task_visibility == "posted"
            )
        else:
            raise ValueError(f"Unknown condition {config.condition!r}")
    return policies


def run_dir_of(
    config: EpisodeConfig, experiment: str | None = None, stamp: str | None = None
) -> Path:
    """Where an episode's logs go: its own stamped folder for a named experiment.

    Args:
        config: The episode config.
        experiment: The experiment name, or None for a scratch run that replaces the last one.
        stamp: The launch's stamp. Required with an experiment.
    """
    if experiment is None:
        return config.output_dir / config.name / f"seed_{config.seed}"
    return config.output_dir / experiment / config.env / config.name / stamp / f"seed_{config.seed}"


def run_config(
    config: EpisodeConfig,
    experiment: str | None = None,
    stamp: str | None = None,
    client: LLMClient | None = None,
    scenario: Scenario | None = None,
) -> tuple[Path, int]:
    """Runs one episode and writes its logs.

    Args:
        config: The episode config.
        experiment: The experiment name, or None for a scratch run.
        stamp: The launch's stamp. Required with an experiment.
        client: A model client to use instead of the configured server (tests use a fake).
        scenario: The loaded scenario to run, when the caller already holds it (a test double that
            reads the same world); loaded from the config otherwise.

    Returns:
        The run directory and the last tick simulated.
    """
    if scenario is None:
        scenario = load_episode_scenario(config)
    plan = None
    if config.condition == "oracle":
        plan = solve_oracle(config.oracle, scenario, config.seed)
    model_hash = None
    if config.executor == "llm" and client is None:
        client = OpenAIChatClient(config.llm)
        if config.llm.model_path is not None:
            model_hash = model_file_hash(config.llm.model_path)
    policies = build_policies(config, scenario, plan, client)
    run_dir = run_dir_of(config, experiment, stamp)
    with EpisodeLogger(run_dir) as logger:
        logger.write_run(config, experiment, stamp, model_hash)
        if plan is not None:
            with (run_dir / ORACLE_PLAN_FILE).open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(dataclasses.asdict(plan), handle, indent=2, sort_keys=True)
                handle.write("\n")
        final_tick = run_episode(config, scenario, policies, logger)
    return run_dir, final_tick


def main() -> None:
    """Command-line entry point: banner, status checks, the episode, and its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="episode config YAML")
    parser.add_argument("--run-name", help="the experiment this run belongs to; keeps its logs")
    args = parser.parse_args()
    configure_console()
    config = load_config(args.config)
    print_banner(f"single episode: {args.config}")
    if not run_checks([config], config.output_dir):
        print("A status check failed; fix it and rerun.")
        sys.exit(2)
    stamp = datetime.now().strftime(RUN_STAMP_FORMAT) if args.run_name else None
    scenario = load_episode_scenario(config)
    print_episode_header(config, scenario, run_dir_of(config, args.run_name, stamp))
    began = time.perf_counter()
    run_dir, final_tick = run_config(config, args.run_name, stamp, scenario=scenario)
    metrics = write_metrics([run_dir])[0]
    print_episode_summary(config, metrics, final_tick, time.perf_counter() - began, run_dir)


if __name__ == "__main__":
    main()
