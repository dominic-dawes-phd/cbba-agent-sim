"""Runs one episode from a config file: python -m experiments.run_single configs/<file>.yaml."""

import argparse
from pathlib import Path

from allocation.scripted import ScriptedPolicy
from sim.agent import Policy
from sim.config import EpisodeConfig, load_config
from sim.logger import EpisodeLogger
from sim.loop import run_episode
from sim.scenario import load_scenario


def run_config(config: EpisodeConfig) -> tuple[Path, int]:
    """Runs one episode with scripted policies and writes its logs.

    Returns:
        The run directory and the last tick simulated.
    """
    scenario = load_scenario(config.map_path, config.mission_path)
    tasks_by_id = {task.task_id: task for task in scenario.tasks}
    policies: dict[str, Policy] = {}
    for spec in scenario.agents:
        assigned = scenario.scripted_assignment.get(spec.agent_id, [])
        policies[spec.agent_id] = ScriptedPolicy(
            [tasks_by_id[task_id] for task_id in assigned], config.scripted
        )
    run_dir = config.output_dir / config.name / f"seed_{config.seed}"
    with EpisodeLogger(run_dir) as logger:
        logger.write_run(config)
        final_tick = run_episode(config, scenario, policies, logger)
    return run_dir, final_tick


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="episode config YAML")
    args = parser.parse_args()
    run_dir, final_tick = run_config(load_config(args.config))
    print(f"Episode ended at tick {final_tick}. Logs in {run_dir}")


if __name__ == "__main__":
    main()
