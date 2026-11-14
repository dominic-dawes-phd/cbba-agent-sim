"""Refreshes an environment's dev figures: python -m experiments.dev_refresh configs/dev/<env>.yaml.

`--all` refreshes every spec in configs/dev/. A spec names the environment, its episode configs,
the seeds, and the sweep levels; cells are built as in experiments.sweep. Each refresh first
deletes `runs/dev/<env>/` and nothing else, then writes logs to
`runs/dev/<env>/logs/<cell>/seed_<n>/` and figures with fixed names to `runs/dev/<env>/figures/`,
so the same paths always hold the latest look at that environment.

These runs check that the simulator and tooling behave. They are never results for the paper.
"""

import argparse
import shutil
from pathlib import Path

import yaml

from experiments.analyze import run_metrics
from experiments.plot import apply_style, fig_churn, fig_cost, fig_h3_loss, fig_h3_range
from experiments.plot_episode import fig_episode_map, fig_episode_progress
from experiments.sweep import build_cells, keep_awake, run_episodes

DEV_ROOT = Path("runs/dev")
DEV_SPEC_DIR = Path("configs/dev")
DEV_WORKERS = 8
SWEEP_FIGURES = (fig_h3_loss, fig_h3_range, fig_churn, fig_cost)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, nargs="?", help="a spec in configs/dev/")
    parser.add_argument("--all", action="store_true", help="refresh every spec in configs/dev/")
    args = parser.parse_args()
    if args.all == (args.spec is not None):
        parser.error("give one spec or --all")
    specs = sorted(DEV_SPEC_DIR.glob("*.yaml")) if args.all else [args.spec]
    for spec_path in specs:
        with spec_path.open(encoding="utf-8") as handle:
            figures = refresh(yaml.safe_load(handle), DEV_ROOT)
        print(f"Figures in {figures}")


def refresh(spec: dict, root: Path) -> Path:
    """Reruns one environment's dev episodes and redraws its figures.

    Args:
        spec: The parsed dev spec.
        root: The folder holding one subfolder per environment.

    Returns:
        The figures folder.
    """
    env_dir = root / spec["env"]
    shutil.rmtree(env_dir, ignore_errors=True)
    figures = env_dir / "figures"
    figures.mkdir(parents=True)
    seed = spec["first_seed"]
    episodes = build_cells(spec, env_dir / "logs")
    keep_awake(True)
    run_dirs, failed = run_episodes(episodes, DEV_WORKERS, None, None)
    keep_awake(False)
    if failed:
        raise RuntimeError(f"{len(failed)} dev episodes failed: {', '.join(failed)}")
    run_dirs.sort()
    runs = [run_metrics(run_dir) for run_dir in run_dirs]

    apply_style()
    for figure in SWEEP_FIGURES:
        print(f"{spec['env']} {figure.__name__}: {figure(runs, figures)}")
    conditions = {config.condition for config in episodes}
    at_baseline = [
        (run_dir, run)
        for run_dir, run in zip(run_dirs, runs, strict=True)
        if run["seed"] == seed and run["name"] in conditions
    ]
    fig_episode_progress(
        [run for _, run in at_baseline],
        figures,
        f"episode_progress_seed{seed}",
        f"{spec['env']}, seed {seed}",
    )
    for run_dir, run in at_baseline:
        fig_episode_map(run_dir, figures, f"episode_map_{run['condition']}_seed{seed}")
    return figures


if __name__ == "__main__":
    main()
