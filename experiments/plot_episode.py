"""Per-episode figures, for checking by eye what a run actually did.

1. fig_episode_progress: tasks outstanding over time, one line per condition at one seed.
2. fig_episode_map: the floor plan with each agent's path, from the per-tick `poses` stream.

Encoding follows experiments.plot: conditions keep their fixed slots, and agents take categorical
slots in sorted id order. Shape marks start and end, so neither relies on color alone.
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments.plot import (  # noqa: E402
    AXIS,
    DIRECT_LABEL_MAX,
    FIGURE_SIZE,
    GRID,
    INK_SECONDARY,
    LEGEND_ANCHOR,
    LEGEND_COLUMNS,
    LINE_WIDTH,
    MARKER_SIZE,
    SLOT_COLORS,
    SURFACE,
    Metrics,
    save,
    style_of,
    write_table,
)
from sim.logger import read_stream  # noqa: E402
from sim.scenario import load_scenario  # noqa: E402

INCHES_PER_CELL = 0.2
MAP_MIN_WIDTH_INCHES = 6.4
MAP_LEGEND_INCHES = 0.7
FURNITURE_LABEL_SIZE = 5
# Shifts each agent's path a little within its cells, so paths along one corridor stay visible.
PATH_OFFSET_SPREAD = 0.3
FLOOR, FURNITURE, WALL = 0, 1, 2


def fig_episode_progress(runs: list[Metrics], out: Path, name: str, title: str) -> None:
    """Tasks outstanding over time, one line per run; all lines end at 0, so the legend names them.

    Args:
        runs: Metrics of the runs to compare, usually one per condition at one seed.
        out: Where to write the figure and its table.
        name: The file name without an extension.
        title: The figure's title.
    """
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    rows = []
    for run in runs:
        color, dash = style_of(str(run["condition"]))
        ticks = [point[0] for point in run["outstanding"]]
        counts = [point[1] for point in run["outstanding"]]
        axes.step(
            ticks, counts, where="post", color=color, linestyle=dash, linewidth=LINE_WIDTH,
            label=str(run["condition"]),
        )  # fmt: skip
        rows += [[run["condition"], run["seed"], tick, count] for tick, count in run["outstanding"]]
    if len(runs) >= 2:
        axes.legend(loc="upper center", bbox_to_anchor=LEGEND_ANCHOR, ncol=LEGEND_COLUMNS)
    axes.set_title(title, color=INK_SECONDARY)
    axes.set_xlabel("tick")
    axes.set_ylabel("tasks outstanding")
    axes.set_ylim(bottom=0)
    save(figure, out, name)
    write_table(out / f"{name}.csv", ["condition", "seed", "tick", "outstanding"], rows)


def fig_episode_map(run_dir: Path, out: Path, name: str) -> None:
    """The floor plan with one path per agent, a circle at its start and a square at its end.

    Args:
        run_dir: The run's log directory.
        out: Where to write the figure.
        name: The file name without an extension.
    """
    with (run_dir / "run.json").open(encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    world = load_scenario(Path(config["map_path"]), Path(config["mission_path"])).world
    poses = read_stream(run_dir, "poses")
    agent_ids = sorted(poses[0]["agents"])

    grid = [[FLOOR] * world.width for _ in range(world.height)]
    for x, y in world.walls:
        grid[y][x] = WALL
    furniture = [obj for obj in world.objects.values() if obj.is_furniture]
    for obj in furniture:
        for x, y in obj.cells():
            grid[y][x] = FURNITURE

    width = max(MAP_MIN_WIDTH_INCHES, INCHES_PER_CELL * world.width)
    height = width * world.height / world.width + MAP_LEGEND_INCHES
    figure, axes = plt.subplots(figsize=(width, height))
    axes.imshow(grid, cmap=ListedColormap([SURFACE, GRID, AXIS]), vmin=FLOOR, vmax=WALL)
    for obj in furniture:
        cells = obj.cells()
        center = (statistics.fmean(x for x, _ in cells), statistics.fmean(y for _, y in cells))
        axes.annotate(
            obj.obj_id, center, ha="center", va="center", fontsize=FURNITURE_LABEL_SIZE,
            color=INK_SECONDARY,
        )  # fmt: skip

    handles = []
    for index, agent_id in enumerate(agent_ids):
        color = SLOT_COLORS[index]
        offset = PATH_OFFSET_SPREAD * ((index + 0.5) / len(agent_ids) - 0.5)
        xs = [row["agents"][agent_id]["cell"][0] + offset for row in poses]
        ys = [row["agents"][agent_id]["cell"][1] + offset for row in poses]
        axes.plot(xs, ys, color=color, linewidth=LINE_WIDTH, solid_joinstyle="round")
        for x, y, marker in ((xs[0], ys[0], "o"), (xs[-1], ys[-1], "s")):
            axes.plot(
                x, y, marker, color=color, markersize=MARKER_SIZE + 2, markeredgecolor=SURFACE,
                markeredgewidth=1,
            )  # fmt: skip
        if len(agent_ids) <= DIRECT_LABEL_MAX:
            axes.annotate(
                agent_id, (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points",
                color=INK_SECONDARY,
            )  # fmt: skip
        handles.append(Line2D([], [], color=color, linewidth=LINE_WIDTH, label=agent_id))
    for marker, label in (("o", "start"), ("s", "end")):
        handles.append(
            Line2D([], [], color=INK_SECONDARY, linestyle="", marker=marker, label=label)
        )
    axes.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles)
    )  # fmt: skip
    axes.set_title(run_dir.parent.name, color=INK_SECONDARY)
    axes.grid(visible=False)
    axes.set_xticks([])
    axes.set_yticks([])
    save(figure, out, name)
