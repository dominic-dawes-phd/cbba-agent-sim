"""Computes episode metrics from logs: python -m experiments.analyze runs/<dir>.

Every metric comes from the JSONL logs after the run (hard rule 9); nothing is measured inside the
sim. The directory may hold one run or many (any folder with a run.json counts). Each run gets a
metrics.json next to its logs, and runs are grouped by environment and config name, with means and
95% bootstrap confidence intervals over seeds.

Metric definitions:
- tasks: the mission's tasks (logged `posted`, or `exists` when tasks must be discovered) plus the
  follow-up tasks discovered during the episode.
- success: every task done, none voided.
- within_budget: the episode ended (every task done or voided) before max_ticks, the tick budget.
- partial_success: done tasks / all tasks.
- makespan: last tick, for runs that finished every task (done or voided) before max_ticks.
- voided: tasks with no item left because theirs shattered. Rare now: a broken item leaves the rest
  of its task to be finished, but a voided single-item task still removes its work, so read
  makespan next to this.
- drops, items_broken: every drop, and those that shattered. Drops cost time (a pause, or the clean
  up reflex), so they add variance and penalty to makespan.
- duplicate_work_ticks: ticks agents spent on a done task that another agent finished (the
  finisher is whoever did the last tagged action on it).
- contested_tasks: tasks that more than one agent worked on.
- orphan_delay: for each crash, ticks from the crash until the task the crashed agent was working
  on got done (censored crashes counted separately).
- stale_belief_rate, hallucination_rate: share of actions with that outcome.
- blocked_ticks: ticks agents could not move (congestion).
- messages_sent, messages_dropped, llm_calls, tokens: costs; messages_per_task and
  tokens_per_task divide them by done tasks. Coverage messages (perception-layer map sharing, the
  same for every condition) are left out and counted in coverage_messages.
- move_conflicts, yields: lost same-cell conflicts and idle step-asides (congestion).
- outstanding: (tick, open task count) at every change, for the tasks-over-time figure. A hidden
  mission task is open from tick 0, before anyone has found it.
- discovery_latency_mean: mean ticks from a hidden mission task's `exists` to its first
  `discovered`; undiscovered counts hidden tasks never found. None and 0 in posted mode.
- makespan_vs_oracle: makespan / the oracle's replayed makespan at the same (mission, seed), when an
  oracle run is in the same folder. Below 1 is possible (the oracle is the best known plan, replayed
  with congestion) and is reported as it is.
- oracle_bound, oracle_planned, solver_gap, oracle_status: for oracle runs, the proven lower bound
  and planned makespan in the planning model, (planned - bound) / bound, and the solver's verdict.
- exploration_ticks: agent-ticks spent in the `explore` and `search` skills.
- coverage: (tick, share of floor some agent counts as searched) per snapshot; final_coverage is
  the last value.
- messages_by_kind: every message sent, by kind (coverage included), so consensus traffic (`cbba`,
  `ledger`) and C1's `dialogue` can be read apart from claims and reports.
- llm_calls_by_purpose: LLM calls by purpose (`act`, `utility`, `dialogue`); llm_wall_seconds is
  their summed wall-clock time, a cost only (hard rule 1).
- parse_failures: executor replies refused before the belief check (an `invalid_call` on
  `reply`): they did not parse, or a task tool named a task other than the agent's own.
  rejected_responses: executor replies refused at the belief check or before it, each a re-plan
  the agent pays `L` for. plan_rejections: C1 joint plans refused by the plan check.
- false_completions: completion claims (`report` in G, `mark_done` in C2/C3) on a task that still
  needed work: it was never done, or someone worked on it after the claim. A claim that leaves a
  wash cycle to finish the task is true. One count per (agent, task). C1 has no structured claim,
  so it is 0 there; its false-done herding shows in the dialogue text only.
- false_completion_rate: false_completions / completion claims (distinct agent-task pairs); None
  with no claim. replan_rate: rejected_responses / act calls; None with no act call.
  consensus_messages_per_task: `cbba` and `ledger` messages / done tasks (0 without consensus).
- ledger_lag_final: in the ledger conditions, the share of (agent, done task) pairs whose agent's
  final ledger copy does not show the task done; None without a ledger.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from sim.config import COMPLETION_TOOLS, EXPLORING_TOOLS, LOG_SUFFIX, ORACLE_PLAN_FILE
from sim.exploration import COVERAGE_KIND
from sim.logger import JsonValue, read_stream

BOOTSTRAP_RESAMPLES = 2000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0

# Metrics summarized across seeds, in table order.
SUMMARY_METRICS = (
    "success",
    "within_budget",
    "partial_success",
    "makespan",
    "makespan_vs_oracle",
    "oracle_bound",
    "oracle_planned",
    "solver_gap",
    "voided",
    "drops",
    "items_broken",
    "duplicate_work_ticks",
    "contested_tasks",
    "stale_belief_rate",
    "hallucination_rate",
    "blocked_ticks",
    "move_conflicts",
    "messages_sent",
    "messages_dropped",
    "llm_calls",
    "tokens",
    "messages_per_task",
    "tokens_per_task",
    "discovery_latency_mean",
    "undiscovered",
    "exploration_ticks",
    "final_coverage",
    "parse_failures",
    "rejected_responses",
    "plan_rejections",
    "false_completions",
    "ledger_lag_final",
    "llm_wall_seconds",
    "false_completion_rate",
    "replan_rate",
    "consensus_messages_per_task",
)

# Message kinds that carry consensus state (CBBA bids and ledger rows), for its cost per task.
CONSENSUS_KINDS = ("cbba", "ledger")

# Actions that do no work on a task, so they never show a claimed task still needed work.
IDLE_TOOLS = ("wait", *COMPLETION_TOOLS)

# Task transitions that open a task (it exists or is known), and that close it.
OPEN_TRANSITIONS = ("posted", "exists", "discovered")
CLOSE_TRANSITIONS = ("done", "voided")

Row = dict[str, JsonValue]


def list_run_dirs(root: Path, stamp: str | None = None) -> list[Path]:
    """Every run directory under a folder, sorted.

    Args:
        root: A run directory or a folder of them.
        stamp: Keep only this launch of a named experiment. Without it, repeated launches of a
            seed are all included.

    Raises:
        ValueError: No run under the folder has that stamp.
    """
    run_dirs = [run_json.parent for run_json in sorted(root.rglob("run.json"))]
    if stamp is None:
        return run_dirs
    stamped = [run_dir for run_dir in run_dirs if run_dir.parent.name == stamp]
    if not stamped:
        raise ValueError(f"No run under {root} has stamp {stamp!r}")
    return stamped


def run_metrics(run_dir: Path) -> dict[str, JsonValue]:
    """All per-run metrics for one run directory."""
    with (run_dir / "run.json").open(encoding="utf-8") as handle:
        run = json.load(handle)
    config = run["config"]
    events = read_stream(run_dir, "events")
    actions = read_stream(run_dir, "actions")
    messages = read_stream(run_dir, "messages")
    llm_calls = read_stream(run_dir, "llm_calls")
    states = read_stream(run_dir, "state")

    tasks = {row["task"] for row in events if row["transition"] in OPEN_TRANSITIONS}
    done_ticks = {row["task"]: row["tick"] for row in events if row["transition"] == "done"}
    coordination = [row for row in messages if row["kind"] != COVERAGE_KIND]
    latencies, undiscovered = discovery_latencies(events)
    voided_ticks = {row["task"]: row["tick"] for row in events if row["transition"] == "voided"}
    voided = set(voided_ticks)
    final_tick = max(row["tick"] for row in states)
    finished = final_tick < config["max_ticks"] - 1 and len(done_ticks) + len(voided) == len(tasks)

    duplicate_ticks, contested = duplicate_work(actions, done_ticks)
    tokens = sum(row["prompt_tokens"] + row["output_tokens"] for row in llm_calls)
    agents = states[-1]["agents"].values()
    delays, censored = orphan_delays(events, actions, done_ticks)
    outcomes = [row["outcome"] for row in actions]
    by_purpose: dict[str, int] = {}
    for row in llm_calls:
        by_purpose[row["purpose"]] = by_purpose.get(row["purpose"], 0) + 1
    by_kind: dict[str, int] = {}
    for row in messages:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    # Runs logged before the ledger and dialogue streams existed (v5.0.00 and earlier) lack them.
    dialogues = []
    if (run_dir / f"dialogues{LOG_SUFFIX}").exists():
        dialogues = read_stream(run_dir, "dialogues")
    ledgers = []
    if (run_dir / f"ledgers{LOG_SUFFIX}").exists():
        ledgers = read_stream(run_dir, "ledgers")
    false_claims, claims = false_completions(actions, {**voided_ticks, **done_ticks})
    rejected = sum(1 for row in llm_calls if row["purpose"] == "act" and row["rejected"])
    consensus = sum(by_kind.get(kind, 0) for kind in CONSENSUS_KINDS)
    plan_path = run_dir / ORACLE_PLAN_FILE
    plan: dict[str, JsonValue] = {}
    if plan_path.exists():
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
    bound, planned = plan.get("bound"), plan.get("planned")
    return {
        "name": config["name"],
        "env": config.get("env"),  # runs before v3.1.00 have no env
        "condition": config["condition"],
        "seed": run["seed"],
        "drop_prob": config["faults"]["drop_prob"],
        "base_loss": config["channel"]["base_loss"],
        "comm_range": config["channel"]["comm_range"],
        "mission": config["mission_path"],
        "max_ticks": config["max_ticks"],
        "crash_ticks": [row["tick"] for row in events if row["transition"] == "crashed"],
        "tasks": len(tasks),
        "done": len(done_ticks),
        "voided": len(voided),
        "drops": sum(1 for row in events if row["transition"] == "dropped"),
        "items_broken": sum(1 for row in events if row["transition"] == "broken"),
        "success": len(done_ticks) == len(tasks),
        "partial_success": len(done_ticks) / len(tasks),
        "finished": finished,
        "within_budget": finished,
        "makespan": final_tick if finished else None,
        "makespan_vs_oracle": None,  # filled in by attach_oracle_ratios once all runs are known
        "oracle_bound": bound,
        "oracle_planned": planned,
        "solver_gap": (planned - bound) / bound if plan else None,
        "oracle_status": plan.get("status"),
        "duplicate_work_ticks": duplicate_ticks,
        "contested_tasks": contested,
        "orphan_delays": delays,
        "orphans_never_done": censored,
        "stale_belief_rate": outcomes.count("stale_belief") / len(outcomes),
        "hallucination_rate": outcomes.count("hallucination") / len(outcomes),
        "blocked_ticks": sum(agent["blocked_ticks"] for agent in agents),
        "move_conflicts": sum(agent["move_conflicts"] for agent in agents),
        "yields": sum(agent["yields"] for agent in agents),
        "messages_sent": len(coordination),
        "messages_dropped": sum(1 for row in coordination if row["outcome"] == "dropped"),
        "coverage_messages": len(messages) - len(coordination),
        "llm_calls": len(llm_calls),
        "tokens": tokens,
        "messages_per_task": len(coordination) / len(done_ticks) if done_ticks else None,
        "tokens_per_task": tokens / len(done_ticks) if done_ticks else None,
        "outstanding": outstanding_over_time(events),
        "discovery_latency_mean": statistics.fmean(latencies) if latencies else None,
        "undiscovered": undiscovered,
        "exploration_ticks": sum(
            row["tick"] - row["start_tick"] + 1
            for row in actions
            if row["call"][0] in EXPLORING_TOOLS
        ),
        "coverage": [[row["tick"], row["coverage"]] for row in states],
        "final_coverage": states[-1]["coverage"],
        "messages_by_kind": by_kind,
        "llm_calls_by_purpose": by_purpose,
        "llm_wall_seconds": sum(row.get("wall_seconds") or 0.0 for row in llm_calls),
        "parse_failures": sum(1 for row in actions if row["call"][0] == "reply"),
        "rejected_responses": rejected,
        "plan_rejections": sum(1 for row in dialogues if row.get("problem")),
        "false_completions": false_claims,
        "ledger_lag_final": ledger_lag(ledgers, done_ticks),
        "false_completion_rate": false_claims / claims if claims else None,
        "replan_rate": rejected / by_purpose["act"] if by_purpose.get("act") else None,
        "consensus_messages_per_task": consensus / len(done_ticks) if done_ticks else None,
    }


def attach_oracle_ratios(runs: list[dict[str, JsonValue]]) -> None:
    """Fills in makespan_vs_oracle for every finished run with an oracle at its (mission, seed)."""
    oracle = {
        (run["mission"], run["seed"]): run["makespan"]
        for run in runs
        if run["condition"] == "oracle" and run["makespan"]
    }
    for run in runs:
        reference = oracle.get((run["mission"], run["seed"]))
        if reference and run["makespan"] is not None:
            run["makespan_vs_oracle"] = run["makespan"] / reference


def discovery_latencies(events: list[Row]) -> tuple[list[int], int]:
    """Ticks from each hidden mission task's `exists` to its first `discovered`.

    Returns:
        (latencies of the tasks that were found, number never found).
    """
    exists = {row["task"]: row["tick"] for row in events if row["transition"] == "exists"}
    found: dict[str, int] = {}
    for row in sorted(events, key=_tick_of):
        if row["transition"] == "discovered" and row["task"] in exists:
            found.setdefault(row["task"], row["tick"])
    latencies = [found[task] - exists[task] for task in sorted(found)]
    return latencies, len(exists) - len(found)


def outstanding_over_time(events: list[Row]) -> list[list[int]]:
    """Open task count after every change.

    A task opens at its first `posted`, `exists`, or `discovered` (a hidden mission task opens at
    `exists`, and finding it later does not open it again); `done` and `voided` close it.

    Returns:
        [tick, open count] pairs in tick order, starting at tick 0.
    """
    opened: set[str] = set()
    count = 0
    series: list[list[int]] = []
    for row in sorted(events, key=_tick_of):
        if row["transition"] in OPEN_TRANSITIONS and row["task"] not in opened:
            opened.add(row["task"])
            count += 1
        elif row["transition"] in CLOSE_TRANSITIONS:
            count -= 1
        else:
            continue
        if series and series[-1][0] == row["tick"]:
            series[-1][1] = count
        else:
            series.append([row["tick"], count])
    return series


def false_completions(actions: list[Row], finished_ticks: dict[str, int]) -> tuple[int, int]:
    """Completion claims on a task that still needed work.

    A claim is false when its task was never finished (done or voided), or when some agent worked
    on the task (anything but waiting or claiming) after the claim and before it finished. A claim
    made when a wash cycle is left to finish the task is true. Each (agent, task) counts once,
    from its first claim.

    Returns:
        The false claims, and all claims (distinct agent-task pairs).
    """
    claims: dict[tuple[str, str], int] = {}
    for row in actions:
        if row["call"][0] in COMPLETION_TOOLS and row["outcome"] == "ok":
            claims.setdefault((row["agent"], row["call"][1]), row["tick"])
    false = 0
    for (_, task_id), tick in claims.items():
        finished = finished_ticks.get(task_id)
        if finished is None or any(
            row["task"] == task_id
            and tick < row["start_tick"] < finished
            and row["call"][0] not in IDLE_TOOLS
            for row in actions
        ):
            false += 1
    return false, len(claims)


def ledger_lag(ledgers: list[Row], done_ticks: dict[str, int]) -> float | None:
    """Share of (agent, done task) pairs whose agent's final ledger copy is not `done`.

    The `ledgers` stream holds every row each agent's copy changed, in order, so the last row per
    (agent, task) is that agent's final view. None for a run with no ledger.
    """
    if not ledgers:
        return None
    final: dict[tuple[str, str], str] = {}
    for row in ledgers:
        for task_id, fields in row["rows"].items():
            final[(row["agent"], task_id)] = fields["status"]
    agents = {row["agent"] for row in ledgers}
    pairs = [(agent, task_id) for agent in agents for task_id in done_ticks]
    if not pairs:
        return None
    lagging = sum(1 for pair in pairs if final.get(pair) != "done")
    return lagging / len(pairs)


def _tick_of(row: Row) -> int:
    return row["tick"]


def duplicate_work(actions: list[Row], done_ticks: dict[str, int]) -> tuple[int, int]:
    """Ticks spent on done tasks by agents other than the finisher, and tasks worked on by many.

    Returns:
        (duplicate work ticks, number of contested tasks).
    """
    workers: dict[str, set[str]] = {}
    finisher: dict[str, str] = {}
    for row in actions:
        task = row["task"]
        if task is None:
            continue
        workers.setdefault(task, set()).add(row["agent"])
        if task in done_ticks and row["tick"] <= done_ticks[task]:
            finisher[task] = row["agent"]
    duplicate = sum(
        row["tick"] - row["start_tick"] + 1
        for row in actions
        if row["task"] in finisher and row["agent"] != finisher[row["task"]]
    )
    contested = sum(1 for agents in workers.values() if len(agents) > 1)
    return duplicate, contested


def orphan_delays(
    events: list[Row], actions: list[Row], done_ticks: dict[str, int]
) -> tuple[list[int], int]:
    """Delay from each crash to the completion of the task the crashed agent was on.

    Returns:
        (delays in ticks for orphaned tasks that got done, count never done).
    """
    delays = []
    censored = 0
    for crash in (row for row in events if row["transition"] == "crashed"):
        before = [
            row
            for row in actions
            if row["agent"] == crash["agent"] and row["tick"] <= crash["tick"] and row["task"]
        ]
        if not before:
            continue
        task = before[-1]["task"]
        if task in done_ticks and done_ticks[task] <= crash["tick"]:
            continue
        if task in done_ticks:
            delays.append(done_ticks[task] - crash["tick"])
        else:
            censored += 1
    return delays, censored


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES)
    )
    tail = (1 - CONFIDENCE) / 2
    return means[int(tail * BOOTSTRAP_RESAMPLES)], means[int((1 - tail) * BOOTSTRAP_RESAMPLES) - 1]


def summarize(group: list[dict[str, JsonValue]]) -> dict[str, str]:
    """Mean and CI per metric over a group of runs; makespan over finished runs only."""
    rng = random.Random(BOOTSTRAP_SEED)
    summary = {}
    for metric in SUMMARY_METRICS:
        values = [float(run[metric]) for run in group if run[metric] is not None]
        if not values:
            summary[metric] = "n/a"
            continue
        low, high = bootstrap_ci(values, rng)
        summary[metric] = f"{statistics.fmean(values):.3g} [{low:.3g}, {high:.3g}]"
    summary["unfinished"] = str(sum(1 for run in group if not run["finished"]))
    return summary


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="a run directory or a folder of them")
    parser.add_argument("--stamp", help="only this launch (YYYYMMDD-HHMMSS) of an experiment")
    args = parser.parse_args()
    print_summary(write_metrics(list_run_dirs(args.runs, args.stamp)))


def write_metrics(run_dirs: list[Path]) -> list[dict[str, JsonValue]]:
    """Computes every run's metrics, oracle ratios included, and writes each run's metrics.json."""
    runs = [run_metrics(run_dir) for run_dir in run_dirs]
    attach_oracle_ratios(runs)
    for run_dir, metrics in zip(run_dirs, runs, strict=True):
        with (run_dir / "metrics.json").open("w", encoding="utf-8", newline="\n") as out:
            json.dump(metrics, out, indent=2, sort_keys=True)
            out.write("\n")
    return runs


def print_summary(runs: list[dict[str, JsonValue]]) -> None:
    """Prints means and CIs per (environment, config name)."""
    groups: dict[str, list[dict[str, JsonValue]]] = {}
    for metrics in runs:
        groups.setdefault(f"{metrics['env']}/{metrics['name']}", []).append(metrics)
    for name, group in sorted(groups.items()):
        print(f"{name}  ({len(group)} seeds, condition {group[0]['condition']})")
        for metric, text in summarize(group).items():
            print(f"  {metric:22} {text}")


if __name__ == "__main__":
    main()
