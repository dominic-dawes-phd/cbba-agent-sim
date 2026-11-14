import dataclasses
import random
from pathlib import Path

import pytest
import yaml

from allocation.scripted import _already_placed
from experiments.analyze import run_metrics
from experiments.run_single import run_config
from sim.agent import DecisionRequest, DecisionResponse, Skill, ToolCall
from sim.beliefs import Belief
from sim.config import FaultConfig, load_config
from sim.exploration import (
    coverage_payload,
    is_searched,
    merge_coverage,
    nearest_frontier,
    retarget_explorers,
)
from sim.logger import EpisodeLogger, read_stream
from sim.loop import reveal_objects_of, run_episode
from sim.pathing import PathPlanner
from sim.scenario import GoalCondition, TaskSpec, draw_spawns, load_scenario
from tests.helpers import make_agent, make_world

HOME = Path("configs/m2_home.yaml")
HOME_GREEDY = Path("configs/m2_home_greedy.yaml")
SMOKE = Path("configs/m1_smoke.yaml")
MISSION_TASKS = {"t_unpack", "t_clean1", "t_clean2", "t_groceries", "t_trash"}


def read_rows(run_dir, stream):
    return read_stream(run_dir, stream)


class ExplorePolicy:
    """Explores while any floor is unsearched, then waits."""

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.frontier_steps is None:
            return DecisionResponse([ToolCall("wait", ("100",))], "wait", 0, 0)
        return DecisionResponse([ToolCall("explore", ())], "explore", 0, 0)


def test_hidden_tasks_are_revealed_by_sight_once_each(tmp_path):
    config = dataclasses.replace(
        load_config(HOME_GREEDY),
        output_dir=tmp_path,
        task_visibility="discovered",
        faults=FaultConfig(drop_prob=0.0),
    )
    run_dir, final_tick = run_config(config)
    events = read_rows(run_dir, "events")
    assert {row["task"] for row in events if row["transition"] == "exists"} == MISSION_TASKS
    assert not [row for row in events if row["transition"] == "posted"]
    discovered = [row["task"] for row in events if row["transition"] == "discovered"]
    assert sorted(task for task in discovered if task in MISSION_TASKS) == sorted(MISSION_TASKS)
    # Reveal runs before triggers, so a mission's dirty plate or grocery never becomes a second
    # task. (plate_3 has no mission task on purpose; its clean_plate_3 is a real discovery.)
    mission_objects = ("plate_1", "plate_2", "milk_1", "bread_1", "apple_1")
    copies = [task for task in discovered if task.startswith(("clean_", "pack_"))]
    assert not [task for task in copies if task.endswith(mission_objects)]
    assert "clean_plate_3" in discovered
    assert final_tick < config.max_ticks - 1
    metrics = run_metrics(run_dir)
    assert metrics["undiscovered"] == 0 and metrics["discovery_latency_mean"] is not None
    # Hidden tasks are outstanding from tick 0 (plus plate_3's cleanup if it is in view at once).
    assert metrics["outstanding"][0][0] == 0
    assert metrics["outstanding"][0][1] >= len(MISSION_TASKS)


def test_posted_runs_report_no_discovery_latency(tmp_path):
    config = dataclasses.replace(
        load_config(HOME), output_dir=tmp_path, faults=FaultConfig(drop_prob=0.0)
    )
    run_dir, _ = run_config(config)
    metrics = run_metrics(run_dir)
    assert metrics["discovery_latency_mean"] is None and metrics["undiscovered"] == 0


def test_reveal_objects_default_to_movable_goal_objects():
    config = load_config(HOME)
    scenario = load_scenario(config.map_path, config.mission_path)
    reveal = reveal_objects_of(scenario.tasks, scenario.world)
    assert reveal["t_clean1"] == ("plate_1",)  # the dishwasher goal is furniture, left out
    assert reveal["t_unpack"] == ("dishwasher_1",)  # explicit reveal_on: the bowls are inside
    assert set(reveal["t_groceries"]) == {"milk_1", "bread_1", "apple_1"}


def test_coverage_decays_and_merges_by_latest_sighting():
    searched = {(1, 1): 10}
    assert is_searched(searched, (1, 1), 15, None) and is_searched(searched, (1, 1), 15, 10)
    assert not is_searched(searched, (1, 1), 20, 10)
    assert not is_searched(searched, (2, 2), 15, None)
    merge_coverage(searched, coverage_payload({(1, 1): 5, (2, 2): 12}))
    assert searched == {(1, 1): 10, (2, 2): 12}


def test_nearest_frontier_is_the_closest_unsearched_floor():
    world = make_world(["#######", "#.....#", "#######"])
    floor = frozenset(world.rooms)
    nothing = frozenset()
    assert nearest_frontier((1, 1), floor, nothing, {(1, 1): 0, (2, 1): 0}, 5, None) == (3, 1)
    everything = {cell: 0 for cell in floor}
    assert nearest_frontier((1, 1), floor, nothing, everything, 5, None) is None


def test_floor_seen_from_afar_stays_a_frontier_until_the_furniture_beside_it_is_seen():
    # Seen at the edge of view range, a floor cell says nothing about the shelf just beyond it.
    world = make_world(["#######", "#.....#", "#######"])
    shelf = frozenset({(5, 0)})
    floor = frozenset(world.rooms)
    everything = {cell: 0 for cell in floor}
    assert nearest_frontier((1, 1), floor, shelf, everything, 5, None) == (5, 1)
    assert nearest_frontier((1, 1), floor, shelf, {**everything, (5, 0): 3}, 5, None) is None


def test_search_starts_a_new_round_but_explore_ends():
    world = make_world(["######", "#....#", "######"])
    floor = frozenset(world.rooms)
    explorer = make_agent("a", (1, 1))
    searcher = make_agent("b", (4, 1))
    for agent, tool in ((explorer, "explore"), (searcher, "search")):
        agent.skill = Skill(ToolCall(tool, ()), 0, interruptible=True)
        agent.searched = {cell: 0 for cell in floor}
        agent.own_searched = {cell: 0 for cell in floor}
    results = retarget_explorers({"a": explorer, "b": searcher}, floor, frozenset(), 5, None)
    assert results["a"].text == "nothing left to explore"
    assert "b" not in results and searcher.own_searched == {}
    assert searcher.skill.targets == (searcher.cell,)


def test_a_search_that_has_swept_its_round_ends():
    world = make_world(["######", "#....#", "######"])
    floor = frozenset(world.rooms)
    searcher = make_agent("b", (4, 1))
    searcher.skill = Skill(ToolCall("search", ()), 0, interruptible=True, targets=((1, 1),))
    searcher.own_searched = {cell: 0 for cell in floor}
    results = retarget_explorers({"b": searcher}, floor, frozenset(), 5, None)
    # An object inside a closed container is never seen: the agent gets a decision point.
    assert results["b"].text == "searched everywhere you can reach"


def test_explorers_cover_the_map_and_coverage_messages_never_prompt_decisions(tmp_path):
    config = dataclasses.replace(load_config(SMOKE), output_dir=tmp_path, max_ticks=150)
    scenario = load_scenario(config.map_path, config.mission_path)
    policies = {spec.agent_id: ExplorePolicy() for spec in scenario.agents}
    with EpisodeLogger(tmp_path) as logger:
        run_episode(config, scenario, policies, logger)
    assert read_rows(tmp_path, "state")[-1]["coverage"] == 1.0
    actions = read_rows(tmp_path, "actions")
    assert any(row["call"] == ["explore"] for row in actions)
    coverage = [row for row in read_rows(tmp_path, "messages") if row["kind"] == "coverage"]
    assert any(row["outcome"] == "delivered" for row in coverage)
    inbox_lines = [
        line
        for call in read_rows(tmp_path, "llm_calls")
        for line in call["prompt"].splitlines()
        if line.startswith("  from ")
    ]
    assert not [line for line in inbox_lines if "coverage" in line]


def test_uniform_spawns_are_seeded_valid_and_apart():
    config = load_config(HOME)
    scenario = load_scenario(config.map_path, config.mission_path)
    first = draw_spawns(scenario.world, scenario.agents, random.Random("0:spawn"), 4)
    again = draw_spawns(scenario.world, scenario.agents, random.Random("0:spawn"), 4)
    other = draw_spawns(scenario.world, scenario.agents, random.Random("1:spawn"), 4)
    assert [spec.cell for spec in first] == [spec.cell for spec in again]
    assert [spec.cell for spec in first] != [spec.cell for spec in other]
    planner = PathPlanner(scenario.world)
    cells = [spec.cell for spec in first]
    for index, cell in enumerate(cells):
        for other_cell in cells[index + 1 :]:
            assert planner.steps(cell, other_cell) >= 4


def test_battery_must_stay_disabled(tmp_path):
    assert load_config(HOME).battery.enabled is False
    raw = yaml.safe_load(HOME.read_text(encoding="utf-8"))
    raw["battery"]["enabled"] = True
    path = tmp_path / "battery_on.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="battery"):
        load_config(path)


def test_an_item_already_where_the_task_wants_it_is_not_taken_back_out():
    task = TaskSpec(
        "t_wash",
        "wash",
        (
            GoalCondition("plate_1", "in", "dishwasher_1"),
            GoalCondition("bowl_1", "in_kind", "cabinet"),
        ),
        (),
        (),
    )
    loaded = Belief("plate", [("plate_1", "in", "dishwasher_1")], {}, (1, 1), 0, "seen")
    on_table = Belief("plate", [("plate_1", "on", "table_1")], {}, (2, 2), 0, "seen")
    cabinet = Belief("cabinet", [], {}, (3, 3), 0, "map")
    stored = Belief("bowl", [("bowl_1", "in", "cabinet_1")], {}, (3, 3), 0, "seen")
    assert _already_placed("plate_1", task, {"plate_1": loaded})
    assert not _already_placed("plate_1", task, {"plate_1": on_table})
    missing = dataclasses.replace(loaded, missing=True)
    assert not _already_placed("plate_1", task, {"plate_1": missing})
    assert _already_placed("bowl_1", task, {"bowl_1": stored, "cabinet_1": cabinet})
