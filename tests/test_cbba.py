"""CBBA (C2, C3): bundle building, consensus, partitions, and the conditions end to end."""

import collections
import dataclasses
from pathlib import Path

from allocation.cbba import COMMITTED_BID, NO_WINNER, CbbaState, decode_cbba
from experiments.analyze import run_metrics
from experiments.run_single import load_episode_scenario, run_config
from sim.config import FaultConfig, load_config
from sim.logger import read_stream
from tests.fake_llm import CodedBrainClient

HOME = Path("configs/m2_home.yaml")
HOUSE = Path("configs/generated/house_48x32_s0/episode_greedy.yaml")


def exchange(states, rounds, tick=0, links=None):
    """Synchronous rounds in which every linked pair swaps its CBBA state."""
    for step in range(rounds):
        payloads = {state.agent_id: state.payload(tick + step) for state in states}
        for state in states:
            for sender, payload in payloads.items():
                linked = links is None or (sender, state.agent_id) in links
                if sender != state.agent_id and linked:
                    state.merge(sender, *decode_cbba(payload), tick + step)


def test_connected_agents_agree_on_one_winner_per_task():
    scores = {
        "a": {"t1": 0.9, "t2": 0.5, "t3": 0.4, "t4": 0.2},
        "b": {"t1": 0.8, "t2": 0.7, "t3": 0.3, "t4": 0.6},
        "c": {"t1": 0.7, "t2": 0.6, "t3": 0.9, "t4": 0.1},
    }
    states = [CbbaState(agent, bundle_size=2) for agent in scores]
    for _ in range(4):
        for state in states:
            state.build(scores[state.agent_id])
        exchange(states, rounds=3)
    owners = collections.Counter(task for state in states for task in state.bundle)
    assert set(owners.values()) == {1}  # conflict-free among connected agents
    assert all(state.winners == states[0].winners for state in states)
    assert states[0].winners["t1"] == "a" and states[0].winners["t3"] == "c"


def test_agents_cut_off_from_each_other_can_both_win_a_task():
    # Hard rule 10: CBBA's guarantee holds only among agents that can reach each other.
    left, right = CbbaState("a", 1), CbbaState("b", 1)
    left.build({"t1": 0.5})
    right.build({"t1": 0.6})
    exchange([left, right], rounds=3, links=set())
    assert left.bundle == right.bundle == ["t1"]
    exchange([left, right], rounds=2)  # the partition heals
    assert left.bundle == [] and right.bundle == ["t1"]


def test_an_outbid_task_drops_with_every_task_added_after_it():
    state = CbbaState("b", 3)
    state.build({"t1": 0.9, "t2": 0.8, "t3": 0.7})
    assert state.bundle == ["t1", "t2", "t3"]
    rival = CbbaState("a", 1)
    rival.build({"t2": 0.95})
    state.merge("a", *decode_cbba(rival.payload(5)), 5)
    assert state.bundle == ["t1"]
    assert state.winners["t2"] == "a" and state.winners["t3"] == NO_WINNER


def test_a_committed_task_is_not_outbid():
    state = CbbaState("b", 1)
    state.build({"t1": 0.2})
    state.commit("t1")
    assert state.bids["t1"] == COMMITTED_BID
    rival = CbbaState("a", 1)
    rival.build({"t1": 0.99})
    state.merge("a", *decode_cbba(rival.payload(3)), 3)
    assert state.bundle == ["t1"]


def auction_run(tmp_path, path, condition, seed=0):
    config = dataclasses.replace(
        load_config(path),
        condition=condition,
        executor="llm",
        output_dir=tmp_path,
        seed=seed,
        faults=FaultConfig(0.0),
    )
    client = CodedBrainClient(config.scripted, load_episode_scenario(config).tasks)
    run_dir, final_tick = run_config(config, client=client)
    return config, run_dir, final_tick


def test_numeric_cbba_finishes_the_kitchen_with_the_ledger_and_no_utility_calls(tmp_path):
    config, run_dir, final_tick = auction_run(tmp_path, HOME, "numeric_cbba")
    assert final_tick < config.max_ticks - 1 and run_metrics(run_dir)["finished"]
    calls = read_stream(run_dir, "llm_calls")
    assert {call["purpose"] for call in calls} == {"act"}
    kinds = {row["kind"] for row in read_stream(run_dir, "messages")}
    assert {"cbba", "ledger"} <= kinds and "claim" not in kinds
    done = {
        task
        for row in read_stream(run_dir, "ledgers")
        for task, fields in row["rows"].items()
        if fields["status"] == "done"
    }
    truth = {row["task"] for row in read_stream(run_dir, "events") if row["transition"] == "done"}
    # The coded brain marks a task done only when it is (a task can also finish as a side effect of
    # other work, unmarked, so the converse need not hold).
    assert done and done <= truth


def test_llm_cbba_scores_each_task_once_per_agent(tmp_path):
    config, run_dir, final_tick = auction_run(tmp_path, HOME, "llm_cbba")
    assert final_tick < config.max_ticks - 1
    calls = read_stream(run_dir, "llm_calls")
    utility = [call for call in calls if call["purpose"] == "utility"]
    assert utility and all(call["latency_ticks"] > 0 for call in utility)
    scored = collections.Counter()
    for call in utility:
        for line in call["prompt"].splitlines():
            if line.startswith("  ") and ":" in line:
                scored[call["agent"], line.split(":")[0].strip()] += 1
    assert set(scored.values()) == {1}


def test_numeric_cbba_finishes_a_house(tmp_path):
    config, run_dir, final_tick = auction_run(tmp_path, HOUSE, "numeric_cbba")
    assert final_tick < config.max_ticks - 1
