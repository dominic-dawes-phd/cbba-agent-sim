import dataclasses
import json
from pathlib import Path

from experiments.run_single import run_config
from sim.agent import DecisionRequest, DecisionResponse, ToolCall
from sim.config import ChurnConfig, LatencyConfig, load_config
from sim.logger import LOG_STREAMS, EpisodeLogger
from sim.loop import run_episode
from sim.scenario import load_scenario

SMOKE = Path("configs/m1_smoke.yaml")
SMOKE_P05 = Path("configs/m1_smoke_p05.yaml")
TASK_IDS = {"t_open", "t_plate1", "t_plate2", "t_mug", "t_pot", "t_close"}


class FixedPolicy:
    """Returns the same calls every decision with a fixed output token count."""

    def __init__(self, calls, output_tokens):
        self.calls = calls
        self.output_tokens = output_tokens

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        return DecisionResponse(list(self.calls), "fixed", 0, self.output_tokens)


def read_rows(run_dir, stream):
    return [json.loads(line) for line in (run_dir / f"{stream}.jsonl").read_text().splitlines()]


def without_wall_fields(rows):
    return [
        {key: value for key, value in row.items() if not key.startswith("wall_")} for row in rows
    ]


def done_tasks(run_dir):
    return {row["task"] for row in read_rows(run_dir, "events") if row["transition"] == "done"}


def test_smoke_mission_completes(tmp_path):
    config = dataclasses.replace(load_config(SMOKE), output_dir=tmp_path)
    run_dir, final_tick = run_config(config)
    assert done_tasks(run_dir) == TASK_IDS
    assert final_tick < config.max_ticks - 1


def test_same_seed_gives_identical_logs(tmp_path):
    base = load_config(SMOKE_P05)
    first, _ = run_config(dataclasses.replace(base, output_dir=tmp_path / "first"))
    second, _ = run_config(dataclasses.replace(base, output_dir=tmp_path / "second"))
    for stream in LOG_STREAMS:
        assert without_wall_fields(read_rows(first, stream)) == without_wall_fields(
            read_rows(second, stream)
        ), stream


def test_packet_loss_drops_messages_and_slows_the_team(tmp_path):
    perfect_dir, perfect_tick = run_config(
        dataclasses.replace(load_config(SMOKE), output_dir=tmp_path / "perfect")
    )
    lossy_dir, lossy_tick = run_config(
        dataclasses.replace(load_config(SMOKE_P05), output_dir=tmp_path / "lossy")
    )
    reasons = {row["reason"] for row in read_rows(lossy_dir, "messages")}
    assert "loss" in reasons
    assert {row["outcome"] for row in read_rows(perfect_dir, "messages")} == {"delivered"}
    assert done_tasks(lossy_dir) == TASK_IDS
    assert lossy_tick > perfect_tick


def test_actions_apply_after_latency_while_teammates_act(tmp_path):
    # decode_rate 1 tok/s at 1 s/tick: L equals the policy's output token count.
    config = dataclasses.replace(
        load_config(SMOKE),
        output_dir=tmp_path,
        max_ticks=12,
        latency=LatencyConfig(prefill_rate=1.0e9, decode_rate=1.0, seconds_per_tick=1.0),
    )
    scenario = load_scenario(config.map_path, config.mission_path)
    policies = {
        "arm_a": FixedPolicy([ToolCall("wait", ("1",))], output_tokens=7),
        "arm_b": FixedPolicy([ToolCall("move_to", ("table_1",))], output_tokens=0),
        "carrier_c": FixedPolicy([ToolCall("wait", ("100",))], output_tokens=0),
    }
    with EpisodeLogger(tmp_path) as logger:
        run_episode(config, scenario, policies, logger)
    calls = read_rows(tmp_path, "llm_calls")
    first_call = next(row for row in calls if row["agent"] == "arm_a")
    assert (first_call["tick"], first_call["latency_ticks"], first_call["tick_applied"]) == (
        0,
        7,
        7,
    )
    actions = read_rows(tmp_path, "actions")
    first_action_a = next(row for row in actions if row["agent"] == "arm_a")
    assert first_action_a["tick"] == 7  # a one-tick wait started at t + L
    assert any(row["agent"] == "arm_b" and row["tick"] < 7 for row in actions)


def test_churn_crashes_and_rejoins_an_agent(tmp_path):
    config = dataclasses.replace(
        load_config(SMOKE),
        output_dir=tmp_path,
        churn=ChurnConfig(num_agents=1, rejoin_delay=10, earliest_tick=5, latest_tick=5),
    )
    run_dir, _ = run_config(config)
    events = read_rows(run_dir, "events")
    crashed = [row for row in events if row["transition"] == "crashed"]
    rejoined = [row for row in events if row["transition"] == "rejoined"]
    assert [row["tick"] for row in crashed] == [5]
    assert [(row["agent"], row["tick"]) for row in rejoined] == [(crashed[0]["agent"], 15)]
    assert done_tasks(run_dir) == TASK_IDS
