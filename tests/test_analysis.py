import dataclasses
import random
from pathlib import Path

from experiments.analyze import (
    bootstrap_ci,
    duplicate_work,
    false_completions,
    ledger_lag,
    orphan_delays,
    run_metrics,
)
from experiments.run_single import run_config
from sim.config import FaultConfig, load_config
from sim.logger import read_stream

HOME = Path("configs/m2_home.yaml")
HOME_GREEDY = Path("configs/m2_home_greedy.yaml")


def action(agent, task, start, end, outcome="ok"):
    return {"agent": agent, "task": task, "start_tick": start, "tick": end, "outcome": outcome}


def test_duplicate_work_counts_ticks_of_agents_other_than_the_finisher():
    actions = [
        action("arm_a", "t_1", 0, 4),
        action("arm_b", "t_1", 0, 2),  # 3 wasted ticks
        action("arm_b", "t_1", 3, 3),  # 1 wasted tick
        action("arm_a", "t_1", 5, 6),  # arm_a finishes t_1
        action("arm_b", "t_2", 4, 9),  # t_2 never done: not counted
        action("arm_b", None, 10, 14),  # idle
    ]
    assert duplicate_work(actions, {"t_1": 6}) == (4, 1)


def test_orphan_delay_runs_from_the_crash_to_completion():
    events = [
        {"agent": "arm_a", "tick": 10, "transition": "crashed"},
        {"agent": "arm_b", "tick": 30, "transition": "crashed"},
    ]
    actions = [action("arm_a", "t_1", 5, 8), action("arm_b", "t_2", 20, 25)]
    assert orphan_delays(events, actions, {"t_1": 40}) == ([30], 1)


def call_row(agent, task, start, end, *call):
    return {**action(agent, task, start, end), "call": list(call)}


def test_a_completion_claim_is_false_only_if_the_task_still_needed_work():
    actions = [
        call_row("arm_a", "t_1", 10, 10, "report", "t_1"),  # wash left running: true
        call_row("arm_a", "t_1", 11, 15, "wait", "5"),
        call_row("arm_b", "t_2", 20, 20, "mark_done", "t_2"),  # then more work on it: false
        call_row("arm_c", "t_2", 21, 24, "pick", "cup_1"),
        call_row("arm_b", "t_2", 30, 30, "mark_done", "t_2"),  # a repeat counts once
        call_row("arm_c", "t_3", 40, 40, "report", "t_3"),  # never done: false
    ]
    assert false_completions(actions, {"t_1": 60, "t_2": 25}) == (2, 3)


def test_ledger_lag_reads_each_agents_last_row():
    ledgers = [
        {"agent": "arm_a", "rows": {"t_1": {"status": "claimed"}}},
        {"agent": "arm_b", "rows": {"t_1": {"status": "claimed"}}},
        {"agent": "arm_a", "rows": {"t_1": {"status": "done"}}},
    ]
    assert ledger_lag(ledgers, {"t_1": 9}) == 0.5
    assert ledger_lag([], {"t_1": 9}) is None


def test_bootstrap_interval_of_a_constant_is_that_constant():
    assert bootstrap_ci([3.0, 3.0, 3.0], random.Random(0)) == (3.0, 3.0)
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], random.Random(0))
    assert 1.0 <= low < 2.5 < high <= 4.0


def test_metrics_of_a_clean_scripted_run(tmp_path):
    config = dataclasses.replace(
        load_config(HOME), output_dir=tmp_path, faults=FaultConfig(drop_prob=0.0)
    )
    run_dir, final_tick = run_config(config)
    metrics = run_metrics(run_dir)
    # Six mission tasks, plus the unloads (and any cupboard overflow) the washed plates call for.
    assert metrics["tasks"] > 6 and metrics["done"] == metrics["tasks"] and metrics["voided"] == 0
    assert metrics["success"] and metrics["finished"]
    assert metrics["makespan"] == final_tick
    assert metrics["hallucination_rate"] == 0.0
    rows = read_stream(run_dir, "actions")
    tagged = {row["task"] for row in rows if row["task"] is not None}
    assert {"t_unpack", "t_clean1", "t_clean2", "t_groceries", "t_trash"} <= tagged


def test_greedy_finishes_and_its_claim_races_show_as_duplicated_work(tmp_path):
    config = dataclasses.replace(
        load_config(HOME_GREEDY), output_dir=tmp_path, faults=FaultConfig(drop_prob=0.0)
    )
    assert config.condition == "greedy"
    run_dir, final_tick = run_config(config)
    metrics = run_metrics(run_dir)
    assert metrics["success"] and final_tick < config.max_ticks - 1
    events = read_stream(run_dir, "events")
    claimers: dict[str, set[str]] = {}
    for row in events:
        if row["transition"] == "claimed":
            claimers.setdefault(row["task"], set()).add(row["agent"])
    # With no consensus, two agents claim the same task in the same tick before hearing each other.
    assert any(len(agents) > 1 for agents in claimers.values())
    assert metrics["contested_tasks"] >= 1
    assert metrics["duplicate_work_ticks"] > 0
    assert metrics["false_completions"] == 0  # the coded executor reports only finished work
    assert metrics["false_completion_rate"] == 0.0
    assert metrics["replan_rate"] == 0.0  # coded decisions are logged as act calls, none refused
    assert metrics["consensus_messages_per_task"] == 0.0  # G has no consensus traffic
