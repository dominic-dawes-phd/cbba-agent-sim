"""C1's dialogue rounds: membership, turn order, EXECUTE rules, adoption, and runs end to end."""

import dataclasses
import json
from pathlib import Path

from allocation.dialogue import DialogueRounds, check_plan
from experiments.analyze import run_metrics
from experiments.run_single import load_episode_scenario, run_config
from sim.config import FaultConfig, load_config
from sim.logger import read_stream
from tests.fake_llm import CodedBrainClient

HOME = Path("configs/m2_home.yaml")


def test_a_round_holds_the_triggers_radio_component_trigger_first():
    rounds = DialogueRounds(load_config(HOME).dialogue, comm_range=5)
    cells = {"a": (0, 0), "b": (4, 0), "c": (8, 0), "d": (30, 0)}
    online = {"a": True, "b": True, "c": True, "d": True}
    opened = rounds.try_open("b", 10, cells, online)
    assert opened.members == ["b", "a", "c"]  # c reaches b's component through a chain of range
    assert rounds.try_open("a", 11, cells, online) is None  # its component is already talking
    other = rounds.try_open("d", 11, cells, online)
    assert other.members == ["d"]  # a partition talks in its own round
    rounds.close(opened)
    assert rounds.round_of("a") is None and rounds.round_of("d") is other


def test_offline_agents_are_not_members():
    rounds = DialogueRounds(load_config(HOME).dialogue, comm_range=None)
    opened = rounds.try_open("a", 0, {"a": (0, 0), "b": (1, 0)}, {"a": True, "b": False})
    assert opened.members == ["a"]


def test_a_plan_is_checked_for_turn_order_members_and_known_tasks():
    members = ["a", "b"]
    ok = [{"agent": "a", "assignment": "t_1"}, {"agent": "b", "assignment": "WAIT"}]
    assert check_plan(ok, members, {"t_1"}, may_execute=True) is None
    assert (
        check_plan(ok, members, {"t_1"}, may_execute=False)
        == "every member must speak before EXECUTE"
    )
    stranger = [{"agent": "z", "assignment": "t_1"}]
    assert check_plan(stranger, members, {"t_1"}, True) == "z is not in this round"
    unknown = [{"agent": "a", "assignment": "t_9"}]
    assert check_plan(unknown, members, {"t_1"}, True) == "t_9 is not a task you know of"


def dialogue_run(tmp_path, **changes):
    config = dataclasses.replace(
        load_config(HOME),
        condition="llm_only",
        executor="llm",
        output_dir=tmp_path,
        faults=FaultConfig(0.0),
        **changes,
    )
    scenario = load_episode_scenario(config)
    client = CodedBrainClient(config.scripted, scenario.tasks, scenario.world)
    return config, *run_config(config, client=client, scenario=scenario)


def test_llm_only_divides_work_by_talking_with_no_structured_messages(tmp_path):
    config, run_dir, final_tick = dialogue_run(tmp_path)
    # The fake's plans are crude, so the bar is real work done by adopted plans, not a finish.
    assert run_metrics(run_dir)["partial_success"] >= 0.5
    kinds = {row["kind"] for row in read_stream(run_dir, "messages")}
    assert "dialogue" in kinds
    assert not kinds & {"task_found", "claim", "report", "release", "ledger", "cbba"}
    calls = read_stream(run_dir, "llm_calls")
    assert {"act", "dialogue"} <= {call["purpose"] for call in calls}
    turns = [row for row in read_stream(run_dir, "dialogues") if row["event"] == "turn"]
    executed = [row for row in turns if row["end"] == "EXECUTE"]
    assert executed and all(json.loads(row["plan"]) for row in executed)


def test_turns_follow_member_order_and_execute_waits_for_everyone(tmp_path):
    _, run_dir, _ = dialogue_run(tmp_path, max_ticks=200)
    rows = read_stream(run_dir, "dialogues")
    rounds = {}
    for row in rows:
        rounds.setdefault(row["round"], []).append(row)
    for events in rounds.values():
        members = events[0]["members"]
        speakers = [row["agent"] for row in events if row["event"] == "turn"]
        assert speakers == [members[index % len(members)] for index in range(len(speakers))]
        executes = [index for index, row in enumerate(events[1:]) if row["end"] == "EXECUTE"]
        assert all(index + 1 >= len(members) for index in executes)


def test_a_lost_execute_is_not_adopted(tmp_path):
    # Every message is lost: only the speaker of an EXECUTE holds the plan.
    config, run_dir, _ = dialogue_run(tmp_path, max_ticks=120)
    lossy = dataclasses.replace(config.channel, base_loss=1.0)
    config = dataclasses.replace(config, channel=lossy, output_dir=tmp_path / "lossy")
    scenario = load_episode_scenario(config)
    client = CodedBrainClient(config.scripted, scenario.tasks, scenario.world)
    run_dir, _ = run_config(config, client=client, scenario=scenario)
    turns = [row for row in read_stream(run_dir, "dialogues") if row.get("end") == "EXECUTE"]
    tagged = {row["agent"] for row in read_stream(run_dir, "actions") if row["task"]}
    assert tagged <= {row["agent"] for row in turns}
