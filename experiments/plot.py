"""Hero figures for the paper: python -m experiments.plot <runs dir> <output dir>.

Reads every run under the runs directory (computing metrics from logs with experiments.analyze)
and writes four figures, each as PNG and PDF plus a CSV with the plotted numbers (the figure's
table view):

1. fig_h3_loss: completion within budget, and wasted work, against packet loss p (H3).
2. fig_h3_range: completion within budget against communication range r, crossovers marked (H3).
3. fig_churn: tasks outstanding over time for one run per condition, crash ticks marked (H3).
4. fig_cost: tokens per episode against completion rate, one point per condition.

The H1 figure (outcomes against mission scale) and the H2 table (C3 minus C2) join once the
mission-scale sweep exists (docs/design/decisions.md, future work).

A figure whose sweep axis has fewer than two levels in the runs is skipped with a message.

Encoding rules (house data-viz method): each condition keeps a fixed categorical slot, so its
color never depends on which conditions are present; ledger ablations ("<condition>-noledger",
"<condition>+ledger") reuse the base condition's hue with a dashed line; one y-axis per panel;
thin marks, hairline solid grid, legends whenever there are two or more series, direct labels at
line ends for four or fewer.
"""

import argparse
import csv
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

from experiments.analyze import (  # noqa: E402
    BOOTSTRAP_SEED,
    bootstrap_ci,
    list_run_dirs,
    run_metrics,
)
from sim.logger import JsonValue  # noqa: E402

Metrics = dict[str, JsonValue]

# Validated categorical palette (light surface, adjacent pairs pass CVD and normal-vision floors).
SLOT_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
# Fixed slot per condition, in paper order. Conditions not built yet keep their slots reserved.
CONDITION_SLOTS = {
    "oracle": 0,
    "llm_only": 1,
    "numeric_cbba": 2,
    "llm_cbba": 3,
    "llm_murdoch": 4,
    "greedy": 5,
    "leader": 6,
    "random": 7,
}
# Ledger ablations (docs/design/conditions.md): C3 without the ledger, C1 with it.
ABLATION_SUFFIXES = ("-noledger", "+ledger")
SCAFFOLD_COLOR = "#898781"  # muted ink: the scripted test policy is not a paper condition
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_WIDTH = 1.5
MARKER_SIZE = 6
BAND_ALPHA = 0.15
DIRECT_LABEL_MAX = 4
LABEL_GAP = 0.06  # minimum vertical gap between end labels, as a share of the y-range
LABEL_STACK = 11  # points between stacked labels of nearby points
LEGEND_ANCHOR = (0.5, -0.2)  # legends sit below the plot, clear of the end labels
LEGEND_COLUMNS = 4
X_MARGIN = 0.1  # room for end labels past the last level
COST_X_HEADROOM = 1.2  # room right of the last point for its label
FIGURE_SIZE = (6.4, 3.6)
WIDE_FIGURE_SIZE = (10.0, 3.6)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="a folder of run directories")
    parser.add_argument("out", type=Path, help="where to write the figures")
    parser.add_argument("--stamp", help="only this launch (YYYYMMDD-HHMMSS) of an experiment")
    args = parser.parse_args()
    runs = [run_metrics(run_dir) for run_dir in list_run_dirs(args.runs, args.stamp)]
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()
    for figure in (fig_h3_loss, fig_h3_range, fig_churn, fig_cost):
        message = figure(runs, args.out)
        print(f"{figure.__name__}: {message}")


def apply_style() -> None:
    """Recessive chrome: hairline solid grid, muted axes, sans text in ink colors."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            # AAAI forbids Type 3 fonts; 42 embeds TrueType.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "legend.frameon": False,
        }
    )


def style_of(condition: str) -> tuple[str, str]:
    """(color, line style) for a condition: its fixed slot, dashed if it is a ledger ablation."""
    base, dash = condition, "-"
    for suffix in ABLATION_SUFFIXES:
        if condition.endswith(suffix):
            base, dash = condition.removesuffix(suffix), "--"
    color = SLOT_COLORS[CONDITION_SLOTS[base]] if base in CONDITION_SLOTS else SCAFFOLD_COLOR
    return color, dash


def sweep_points(
    runs: list[Metrics], axis: str, metric: str
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Per condition, (level, mean, CI low, CI high) for a metric at each level of a sweep axis."""
    grouped: dict[str, dict[float, list[float]]] = {}
    for run in runs:
        if run[metric] is None or run[axis] is None:
            continue
        levels = grouped.setdefault(str(run["condition"]), {})
        levels.setdefault(float(run[axis]), []).append(float(run[metric]))
    rng = random.Random(BOOTSTRAP_SEED)
    points = {}
    for condition, levels in sorted(grouped.items()):
        rows = []
        for level in sorted(levels):
            values = levels[level]
            low, high = bootstrap_ci(values, rng)
            rows.append((level, statistics.fmean(values), low, high))
        points[condition] = rows
    return points


def draw_lines(axes: plt.Axes, points: dict[str, list[tuple[float, float, float, float]]]) -> None:
    """One line with a CI band per condition; direct end labels when there are few series."""
    for condition, rows in points.items():
        color, dash = style_of(condition)
        levels = [row[0] for row in rows]
        axes.fill_between(
            levels, [row[2] for row in rows], [row[3] for row in rows], color=color,
            alpha=BAND_ALPHA, linewidth=0,
        )  # fmt: skip
        axes.plot(
            levels, [row[1] for row in rows], color=color, linestyle=dash, linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE, label=condition,
        )  # fmt: skip
    axes.margins(x=X_MARGIN)
    if len(points) <= DIRECT_LABEL_MAX and not _ends_collide(axes, points):
        for condition, rows in points.items():
            axes.annotate(
                condition, (rows[-1][0], rows[-1][1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK_SECONDARY,
            )  # fmt: skip
    if len(points) >= 2:
        axes.legend(loc="upper center", bbox_to_anchor=LEGEND_ANCHOR, ncol=LEGEND_COLUMNS)


def _ends_collide(
    axes: plt.Axes, points: dict[str, list[tuple[float, float, float, float]]]
) -> bool:
    """Whether two lines end too close for direct labels (then the legend alone names them)."""
    low, high = axes.get_ylim()
    ends = sorted(rows[-1][1] for rows in points.values())
    gaps = [later - earlier for earlier, later in zip(ends, ends[1:], strict=False)]
    return any(gap < LABEL_GAP * (high - low) for gap in gaps)


def write_table(path: Path, header: list[str], rows: list[list[JsonValue]]) -> None:
    """The figure's table view: the plotted numbers as CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save(figure: plt.Figure, out: Path, name: str) -> None:
    """Writes a figure as PNG and PDF."""
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(out / f"{name}.{extension}", dpi=200)
    plt.close(figure)


def baseline(runs: list[Metrics], swept: str) -> list[Metrics]:
    """Runs whose other sweep axes sit at their baseline: no loss, unlimited range, no crash.

    Each figure varies one axis; without this, runs from another sweep (all at p = 0, say) would
    be pooled into the first level of this one.
    """
    kept = []
    for run in runs:
        at_baseline = {
            "base_loss": run["base_loss"] == 0,
            "comm_range": run["comm_range"] is None,
            "churn": not run["crash_ticks"],
        }
        if all(ok for axis, ok in at_baseline.items() if axis != swept):
            kept.append(run)
    return kept


def swept_runs(runs: list[Metrics], axis: str) -> list[Metrics]:
    """Baseline runs of the conditions swept along an axis. C0 plans once and is not swept."""
    return [run for run in baseline(runs, axis) if run["condition"] != "oracle"]


def levels_of(runs: list[Metrics], axis: str) -> set[float]:
    """Distinct non-null values of a sweep axis across runs."""
    return {float(run[axis]) for run in runs if run[axis] is not None}


def fig_h3_loss(runs: list[Metrics], out: Path) -> str:
    """Completion and wasted work against packet loss, side by side (two measures, two axes)."""
    runs = swept_runs(runs, "base_loss")
    if len(levels_of(runs, "base_loss")) < 2:
        return "skipped (fewer than two packet-loss levels)"
    completion = sweep_points(runs, "base_loss", "within_budget")
    wasted = sweep_points(runs, "base_loss", "duplicate_work_ticks")
    figure, (left, right) = plt.subplots(1, 2, figsize=WIDE_FIGURE_SIZE)
    draw_lines(left, completion)
    left.set_xlabel("packet loss p")
    left.set_ylabel("episodes finished within budget")
    left.set_ylim(0, 1.05)
    draw_lines(right, wasted)
    right.set_xlabel("packet loss p")
    right.set_ylabel("wasted work (ticks on tasks others finished)")
    right.set_ylim(bottom=0)
    save(figure, out, "fig_h3_loss")
    rows = [
        [condition, level, mean, low, high, "within_budget"]
        for condition, points in completion.items()
        for level, mean, low, high in points
    ] + [
        [condition, level, mean, low, high, "duplicate_work_ticks"]
        for condition, points in wasted.items()
        for level, mean, low, high in points
    ]
    write_table(
        out / "fig_h3_loss.csv", ["condition", "p", "mean", "ci_low", "ci_high", "metric"], rows
    )
    return "written"


def fig_h3_range(runs: list[Metrics], out: Path) -> str:
    """Completion against communication range, with every crossover between conditions marked."""
    runs = swept_runs(runs, "comm_range")
    if len(levels_of(runs, "comm_range")) < 2:
        return "skipped (fewer than two finite communication ranges)"
    points = sweep_points(runs, "comm_range", "within_budget")
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    draw_lines(axes, points)
    conditions = sorted(points)
    for index, first in enumerate(conditions):
        for second in conditions[index + 1 :]:
            for crossing in crossovers(points[first], points[second]):
                axes.axvline(crossing, color=AXIS, linewidth=0.8)
                axes.annotate(
                    f"{first} / {second} cross", (crossing, 1.0), xytext=(3, -10),
                    textcoords="offset points", color=INK_SECONDARY, fontsize=8,
                )  # fmt: skip
    axes.set_xlabel("communication range r (cells)")
    axes.set_ylabel("episodes finished within budget")
    axes.set_ylim(0, 1.05)
    save(figure, out, "fig_h3_range")
    rows = [
        [condition, level, mean, low, high]
        for condition, rows_ in points.items()
        for level, mean, low, high in rows_
    ]
    write_table(out / "fig_h3_range.csv", ["condition", "r", "mean", "ci_low", "ci_high"], rows)
    return "written"


def crossovers(
    first: list[tuple[float, float, float, float]], second: list[tuple[float, float, float, float]]
) -> list[float]:
    """Levels where two mean curves cross (linear interpolation between shared levels)."""
    shared = sorted({row[0] for row in first} & {row[0] for row in second})
    first_mean = {row[0]: row[1] for row in first}
    second_mean = {row[0]: row[1] for row in second}
    crossings = []
    for low, high in zip(shared, shared[1:], strict=False):
        before = first_mean[low] - second_mean[low]
        after = first_mean[high] - second_mean[high]
        if before * after < 0:
            crossings.append(low + (high - low) * before / (before - after))
    return crossings


def fig_churn(runs: list[Metrics], out: Path) -> str:
    """Tasks outstanding over time, one run per condition, crash ticks marked."""
    with_crash = [run for run in runs if run["crash_ticks"]]
    if not with_crash:
        return "skipped (no run with a crash)"
    chosen: dict[str, Metrics] = {}
    for run in sorted(with_crash, key=_seed_of):
        chosen.setdefault(str(run["condition"]), run)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    rows = []
    for condition, run in sorted(chosen.items()):
        color, dash = style_of(condition)
        ticks = [point[0] for point in run["outstanding"]]
        counts = [point[1] for point in run["outstanding"]]
        axes.step(
            ticks, counts, where="post", color=color, linestyle=dash, linewidth=LINE_WIDTH,
            label=f"{condition} (seed {run['seed']})",
        )  # fmt: skip
        for crash in run["crash_ticks"]:
            axes.axvline(crash, color=color, linewidth=0.8, alpha=0.6)
        rows += [[condition, run["seed"], tick, count] for tick, count in run["outstanding"]]
    top = max(row[3] for row in rows)
    for crash in sorted({tick for run in chosen.values() for tick in run["crash_ticks"]}):
        axes.annotate("crash", (crash, top), xytext=(3, 0), textcoords="offset points",
                      color=INK_SECONDARY, fontsize=8)  # fmt: skip
    if len(chosen) >= 2:
        axes.legend(loc="upper center", bbox_to_anchor=LEGEND_ANCHOR, ncol=LEGEND_COLUMNS)
    axes.set_xlabel("tick")
    axes.set_ylabel("tasks outstanding")
    axes.set_ylim(bottom=0)
    save(figure, out, "fig_churn")
    write_table(out / "fig_churn.csv", ["condition", "seed", "tick", "outstanding"], rows)
    return "written"


def _seed_of(run: Metrics) -> int:
    return int(run["seed"])


def fig_cost(runs: list[Metrics], out: Path) -> str:
    """Tokens per episode against completion rate: one direct-labeled point per condition."""
    grouped: dict[str, list[Metrics]] = {}
    for run in baseline(runs, ""):
        grouped.setdefault(str(run["condition"]), []).append(run)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    rows = []
    for condition, group in sorted(grouped.items()):
        tokens = statistics.fmean(float(run["tokens"]) for run in group)
        completion = statistics.fmean(float(run["within_budget"]) for run in group)
        rows.append([condition, len(group), tokens, completion])
    top_tokens = max(row[2] for row in rows)
    placed: list[tuple[float, float]] = []
    for condition, _, tokens, completion in rows:
        color, _ = style_of(condition)
        axes.plot(tokens, completion, "o", color=color, markersize=MARKER_SIZE + 2,
                  markeredgecolor=SURFACE, markeredgewidth=2)  # fmt: skip
        # Stack the labels of points that sit close together so they never overlap.
        near = sum(
            1
            for x, y in placed
            if abs(x - tokens) < LABEL_GAP * top_tokens and abs(y - completion) < LABEL_GAP
        )
        axes.annotate(
            condition, (tokens, completion), xytext=(7, 4 + LABEL_STACK * near),
            textcoords="offset points", color=INK_SECONDARY,
        )  # fmt: skip
        placed.append((tokens, completion))
    axes.set_xlabel("tokens per episode")
    axes.set_ylabel("episodes finished within budget")
    axes.set_ylim(0, 1.05)
    axes.set_xlim(0, COST_X_HEADROOM * max(row[2] for row in rows))
    save(figure, out, "fig_cost")
    write_table(out / "fig_cost.csv", ["condition", "runs", "tokens", "within_budget"], rows)
    return "written"


if __name__ == "__main__":
    main()
