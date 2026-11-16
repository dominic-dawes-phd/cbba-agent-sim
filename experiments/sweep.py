"""Runs a whole experiment from one spec: python -m experiments.sweep configs/sweeps/<name>.yaml.

A spec lists episode configs (one per condition, with `{env}` expanded over the spec's
environments), the seeds, and the sweep levels. The runner builds every cell, each config at
baseline and then one factor at a time (packet loss, communication range, churn), with the oracle
only at baseline. It runs every (cell, seed) episode in parallel under one launch stamp, writes
metrics.json next to each run, and prints means with 95% CIs per (environment, cell).

Logs go to `runs/<experiment>/<env>/<cell>/<stamp>/seed_<n>/`, and the spec is copied to
`runs/<experiment>/sweep_<stamp>.yaml`, so every launch is kept and can be rerun. A failed
episode is logged and the sweep goes on; failures are listed at the end and the exit code is 1.

On a cluster (`docs/design/cluster.md`), each Slurm array task runs one shard of the episodes
(`--shard I --of N`) against its own model server, and all shards share one `--stamp`, so
`experiments.analyze runs/<experiment> --stamp <stamp>` reads them as one launch.
"""

import argparse
import ctypes
import dataclasses
import logging
import operator
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

from experiments.analyze import print_summary, write_metrics
from experiments.console import (
    configure_console,
    print_banner,
    print_sweep_footer,
    print_sweep_header,
    run_checks,
)
from experiments.run_single import CODED_CONDITIONS, run_config
from sim.config import RUN_STAMP_FORMAT, EpisodeConfig, load_config
from sim.logger import format_duration

RUNS_ROOT = Path("runs")
# Windows power request flags: keep the machine out of idle sleep (Modern Standby freezes every
# process, which once stretched a one-minute run to half an hour) until the flags are cleared.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

log = logging.getLogger(__name__)


def main() -> None:
    """Command-line entry point."""
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="a sweep spec in configs/sweeps/")
    parser.add_argument("--shard", type=int, default=0, help="this shard's index, from 0")
    parser.add_argument("--of", type=int, default=1, help="the number of shards")
    parser.add_argument("--stamp", help="the launch stamp all shards share (default: now)")
    parser.add_argument(
        "--endpoint", help="this shard's model server, e.g. http://127.0.0.1:18081 (a cluster port)"
    )
    args = parser.parse_args()
    if not 0 <= args.shard < args.of:
        parser.error(f"--shard must be in [0, {args.of})")
    with args.spec.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if args.endpoint:
        # Where the server listens is infrastructure, not an experiment setting; it never
        # reaches a prompt or the response cache key.
        spec["llm"] = {**(spec.get("llm") or {}), "endpoint": args.endpoint}
    stamp = args.stamp or datetime.now().strftime(RUN_STAMP_FORMAT)
    experiment_dir = RUNS_ROOT / spec["experiment"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.spec, experiment_dir / f"sweep_{stamp}.yaml")
    all_episodes = build_cells(spec, RUNS_ROOT)
    episodes = all_episodes[args.shard :: args.of]
    print_banner(f"sweep: {args.spec}")
    if not run_checks(episodes, RUNS_ROOT):
        print("A status check failed; fix it and rerun.")
        sys.exit(2)
    print_sweep_header(spec, episodes, len(all_episodes), args.shard, args.of, stamp)
    began = time.perf_counter()
    keep_awake(True)
    run_dirs, failed = run_episodes(episodes, spec["workers"], spec["experiment"], stamp)
    keep_awake(False)
    print_sweep_footer(len(run_dirs), failed, time.perf_counter() - began)
    print_summary(write_metrics(sorted(run_dirs)))
    print(f"Logs in {experiment_dir}, stamp {stamp}")
    if failed:
        print(f"{len(failed)} episodes failed: {', '.join(failed)}")
        sys.exit(1)


def build_cells(spec: dict, output_dir: Path) -> list[EpisodeConfig]:
    """Every episode of a spec: each config at baseline, then one sweep at a time, at every seed.

    The oracle plans once at baseline and is not swept (`docs/design/conditions.md`). Episodes
    are ordered by seed, then environment, keeping the spec's order within each, so a sweep that
    stops early has every cell at the same seeds.

    Args:
        spec: The parsed spec: `episodes` (config paths, `{env}` expanded over `envs`), `seeds`
            and `first_seed`, `loss_levels`, `range_levels`, and `churn` (null for no churn cell).
            Optional: `llm` (fields that replace every episode's `llm` block, such as the model
            path on a cluster) and `only` (labels "<env>/<cell>/seed_<n>" to keep, for reruns).
        output_dir: The folder the episodes write their logs under.
    """
    seeds = range(spec["first_seed"], spec["first_seed"] + spec["seeds"])
    llm_fields = spec.get("llm") or {}
    episodes = []
    for template in spec["episodes"]:
        if "{env}" in template:
            paths = [template.format(env=env) for env in spec["envs"]]
        else:
            paths = [template]
        for path in paths:
            config = load_config(Path(path))
            llm = dataclasses.replace(config.llm, **llm_fields)
            base = dataclasses.replace(config, output_dir=output_dir, llm=llm)
            for cell in sweep_cells(base, spec):
                episodes += [dataclasses.replace(cell, seed=seed) for seed in seeds]
    episodes.sort(key=operator.attrgetter("seed", "env"))
    if spec.get("only"):
        episodes = [config for config in episodes if episode_label(config) in spec["only"]]
    return episodes


def episode_label(config: EpisodeConfig) -> str:
    """An episode's "<env>/<cell>/seed_<n>" label, as failures are reported and `only` lists."""
    return f"{config.env}/{config.name}/seed_{config.seed}"


def sweep_cells(base: EpisodeConfig, spec: dict) -> list[EpisodeConfig]:
    """One config's cells: baseline, each loss level, each range level, and the churn setting.

    A cell is named by its condition, with `_llm` added when a condition that also runs coded (G)
    runs with the LLM executor, so the two never share a folder.
    """
    label = base.condition
    if base.executor == "llm" and base.condition in CODED_CONDITIONS:
        label = f"{base.condition}_llm"
    cells = [dataclasses.replace(base, name=label)]
    if base.condition == "oracle":
        return cells
    for loss in spec["loss_levels"]:
        channel = dataclasses.replace(base.channel, base_loss=loss)
        cells.append(dataclasses.replace(base, name=f"{label}_p{loss}", channel=channel))
    for comm_range in spec["range_levels"]:
        channel = dataclasses.replace(base.channel, comm_range=comm_range)
        cells.append(dataclasses.replace(base, name=f"{label}_r{comm_range}", channel=channel))
    if spec["churn"] is not None:
        churn = dataclasses.replace(base.churn, **spec["churn"])
        cells.append(
            dataclasses.replace(base, name=f"{label}_churn{churn.num_agents}", churn=churn)
        )
    return cells


def run_episodes(
    episodes: list[EpisodeConfig], workers: int, experiment: str | None, stamp: str | None
) -> tuple[list[Path], list[str]]:
    """Runs episodes in parallel; one that raises is logged and skipped.

    Args:
        episodes: The configs to run.
        workers: Processes to run them in.
        experiment: The experiment name, or None for scratch runs.
        stamp: The launch stamp. Required with an experiment.

    Returns:
        The run directories of the episodes that finished, and "<env>/<cell>/seed_<n>" for each
        that failed.
    """
    run_dirs = []
    failed = []
    began = time.perf_counter()
    # Each worker process sets up its own console, so episodes' progress lines show there too.
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_console) as pool:
        futures = {
            pool.submit(run_config, config, experiment, stamp): config for config in episodes
        }
        for done, future in enumerate(as_completed(futures), start=1):
            config = futures[future]
            label = episode_label(config)
            elapsed = time.perf_counter() - began
            left = format_duration(elapsed / done * (len(episodes) - done))
            try:
                run_dir, final_tick = future.result()
            except Exception:  # one broken episode must not lose the rest of a long sweep
                log.exception("[%d/%d] FAILED %s; about %s left", done, len(episodes), label, left)
                failed.append(label)
                continue
            run_dirs.append(run_dir)
            log.info(
                "[%d/%d] finished %s at tick %d | %s so far, about %s left",
                done,
                len(episodes),
                label,
                final_tick,
                format_duration(elapsed),
                left,
            )
    return run_dirs, failed


def keep_awake(on: bool) -> None:
    """Asks Windows not to idle-sleep while episodes run (closing the lid still sleeps).

    Elsewhere it does nothing: simulated time never depends on it, only how long a sweep takes.
    """
    if sys.platform == "win32":
        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


if __name__ == "__main__":
    main()
