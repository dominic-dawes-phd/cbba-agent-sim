"""Shared dishwashers: washed goals, the countdown, splits, pick cancels, executor rules."""

import dataclasses
import random
from pathlib import Path

import yaml

from allocation.scripted import ScriptedPolicy, all_items_shattered
from sim.agent import ActionResult, DecisionRequest, Skill, ToolCall
from sim.beliefs import Belief
from sim.channel import Network
from sim.config import RETURN_ITEM_TOOL, TASK_TEMPLATES_PATH, ScriptedConfig, load_config
from sim.logger import EpisodeLogger, read_stream
from sim.loop import cancel_stale_picks, handle_cycle_starts, stop_picks_of
from sim.motion import advance_moves
from sim.pathing import PathPlanner
from sim.perception import observed_attrs
from sim.primitives import finish_return_item
from sim.scenario import GoalCondition, Scenario, goal_holds, task_from_entry
from sim.triggers import find_triggered_tasks
from tests.helpers import add_furniture, add_item, make_agent, make_world
from tests.test_primitives import run_call

SMOKE = Path("configs/m1_smoke.yaml")
SCRIPTED = ScriptedConfig(wait_ticks=5)


def templates():
    with TASK_TEMPLATES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def wash(task_id, items, appliance="dishwasher_1"):
    entry = {"id": task_id, "kind": "clean_dishes", "items": items, "appliance": appliance}
    return task_from_entry(entry, templates())


def kitchen():
    """A dishwasher at the west end, a counter at the east end, dirty plates on the counter."""
    world = make_world(["........", "........"])
    add_furniture(world, "dishwasher_1", (0, 0), container=True).washing_time = 60
    add_furniture(world, "counter_1", (7, 0), surface=True)
    for index in (1, 2, 3):
        add_item(world, f"plate_{index}", (7, 0), parent_id="counter_1").visual_attrs["dirty"] = (
            "yes"
        )
    return world


def belief(obj_type, relations, cell, last_seen=0, **attrs):
    return Belief(obj_type, relations, dict(attrs), cell, last_seen, "seen")


def request(agent_id, tick, beliefs, known=(), last_result=None):
    steps = {obj_id: 3 for obj_id in beliefs}
    card = make_agent(agent_id, (0, 0)).card
    return DecisionRequest(
        agent_id, tick, "", [], last_result, beliefs, steps, tuple(known), (), None, None, card, ()
    )


def test_a_dish_counts_as_washed_once_any_cycle_that_held_it_ends():
    world = kitchen()
    goal = GoalCondition("plate_1", "washed", "yes")
    world.attach("plate_1", "dishwasher_1", (0, 0))
    assert not goal_holds(world, goal)
    world.start_cycle("dishwasher_1", "arm_a")
    for _ in range(60):
        world.advance_cycles()
    assert goal_holds(world, goal)


def test_the_countdown_is_part_of_what_anyone_sees():
    world = kitchen()
    world.start_cycle("dishwasher_1", "arm_a")
    world.advance_cycles()
    attrs = observed_attrs(world, "dishwasher_1")
    assert (attrs["cycle"], attrs["cycle_left"]) == ("running", "59")


def split_setup(tmp_path):
    world = kitchen()
    world.attach("plate_1", "dishwasher_1", (0, 0))
    holder = make_agent("arm_b", (6, 1))
    world.hold("plate_2", "arm_b", holder.cell)
    holder.held.append("plate_2")
    holder.picked_from["plate_2"] = ("counter_1", (7, 0))
    starter = make_agent("arm_a", (1, 1))
    task = wash("t_wash", ["plate_1", "plate_2", "plate_3"])
    scenario = Scenario(world, [], [task], {}, templates())
    world.cycles_started.append(("dishwasher_1", "arm_a"))
    return world, scenario, {"arm_a": starter, "arm_b": holder}


def test_a_cycle_started_mid_load_splits_the_task_and_the_holder_puts_its_dish_back(tmp_path):
    world, scenario, agents = split_setup(tmp_path)
    with EpisodeLogger(tmp_path) as logger:
        network = Network(load_config(SMOKE).channel, world.walls, 0, logger)
        children = handle_cycle_starts(world, scenario, agents, set(), set(), network, logger)
    parent, child = scenario.tasks
    assert children == [child] and child.task_id == "t_wash_r1"
    assert [goal.obj_id for goal in parent.goals] == ["plate_1"] and parent.steps == ()
    assert [goal.obj_id for goal in child.goals] == ["plate_2", "plate_3"]
    assert child.steps[-1] == ToolCall("start", ("dishwasher_1",))
    assert all("t_wash_r1" in agent.known_tasks for agent in agents.values())
    assert world.cycles_started == []
    events = read_stream(tmp_path, "events")
    assert [row["transition"] for row in events] == ["split", "exists", "discovered"]
    holder = agents["arm_b"]
    assert holder.skill.call.name == RETURN_ITEM_TOOL and holder.queue == []
    planner = PathPlanner(world)
    rng = random.Random(0)
    result = None
    while result is None:
        world.tick += 1
        result = advance_moves({"arm_b": holder}, world, planner, rng).get("arm_b")
    returned = finish_return_item(holder, world)
    assert returned.ok and holder.held == []
    assert world.objects["plate_2"].parent_id == "counter_1"


def test_nothing_splits_when_nothing_or_everything_is_loaded(tmp_path):
    for loaded in ([], ["plate_1", "plate_2", "plate_3"]):
        world = kitchen()
        for plate in loaded:
            world.attach(plate, "dishwasher_1", (0, 0))
        task = wash("t_wash", ["plate_1", "plate_2", "plate_3"])
        scenario = Scenario(world, [], [task], {}, templates())
        world.cycles_started.append(("dishwasher_1", "arm_a"))
        agents = {"arm_a": make_agent("arm_a", (1, 1))}
        with EpisodeLogger(tmp_path / str(len(loaded))) as logger:
            network = Network(load_config(SMOKE).channel, world.walls, 0, logger)
            assert handle_cycle_starts(world, scenario, agents, set(), set(), network, logger) == []
        assert scenario.tasks == [task]


def test_a_queued_pick_is_cancelled_once_the_item_is_seen_elsewhere(tmp_path):
    agent = make_agent("arm_a", (3, 1))
    agent.beliefs["plate_1"] = belief("plate", [("plate_1", "in", "dishwasher_1")], (0, 0))
    agent.pick_places = {"plate_1": ("on", "counter_1")}
    agent.queue = [ToolCall("pick", ("plate_1",))]
    with EpisodeLogger(tmp_path) as logger:
        cancel_stale_picks(agent, ["plate_1"], 12, logger)
    assert agent.queue == [] and agent.decision_due
    assert (agent.last_result.outcome, agent.last_result.text) == (
        "stale_belief",
        "plate_1 is no longer on counter_1",
    )


def test_a_pick_fails_when_its_item_moved_unseen():
    world = kitchen()
    agent = make_agent("arm_a", (6, 0))
    agent.pick_places = {"plate_1": ("on", "dishwasher_1")}  # believed loaded, still on the counter
    result = run_call(ToolCall("pick", ("plate_1",)), agent, world)
    assert (result.outcome, result.text) == ("stale_belief", "plate_1 is no longer on dishwasher_1")


def dishwasher_beliefs(tick, running_until=None, second=False, holding=False):
    """Beliefs of an agent next to the counter; dishwasher_1 mid-cycle until `running_until`."""
    beliefs = {
        "counter_1": belief("counter", [], (7, 0)),
        "dishwasher_1": belief("dishwasher", [], (0, 0), cycle="idle", state="closed"),
        "plate_1": belief(
            "plate",
            [("plate_1", "held_by", "arm_a")] if holding else [("plate_1", "on", "counter_1")],
            (7, 0),
            dirty="yes",
        ),
    }
    if running_until is not None:
        left = str(running_until - tick)
        beliefs["dishwasher_1"] = belief(
            "dishwasher", [], (0, 0), tick, cycle="running", cycle_left=left, state="closed"
        )
    if second:
        beliefs["dishwasher_2"] = belief("dishwasher", [], (0, 1), cycle="idle", state="closed")
    return beliefs


def test_a_wash_moves_to_another_dishwasher_when_its_own_is_busy():
    task = wash("t_wash", ["plate_1"])
    policy = ScriptedPolicy([task], SCRIPTED, [task])
    policy.decide(request("arm_a", 10, dishwasher_beliefs(10, running_until=60, second=True)))
    assert ToolCall("start", ("dishwasher_2",)) in policy._tasks[0].steps


def tidy_task():
    entry = {
        "id": "t_tidy",
        "kind": "tidy",
        "item": "sock_1",
        "relation": "on",
        "dest": "counter_1",
    }
    return task_from_entry(entry, templates())


def test_with_no_dishwasher_free_an_adopted_wash_is_put_back_and_released():
    task = wash("t_wash", ["plate_1"])
    policy = ScriptedPolicy([], SCRIPTED, [task, tidy_task()])
    known = ["t_wash", "t_tidy"]
    first = policy.decide(request("arm_a", 0, dishwasher_beliefs(0), known=["t_wash"]))
    assert first.tool_calls[-1] == ToolCall("pick", ("plate_1",))
    busy = dishwasher_beliefs(10, running_until=60, holding=True)
    busy["sock_1"] = belief("sock", [], (3, 1))  # other work to do meanwhile
    picked = ActionResult(ToolCall("pick", ("plate_1",)), "ok", "picked up plate_1")
    calls = policy.decide(request("arm_a", 10, busy, known, picked)).tool_calls
    assert ToolCall("place", ("plate_1", "counter_1")) in calls
    assert ToolCall("release", ("t_wash",)) in calls
    # Not taken again while the machine is believed busy; free once its countdown has run out.
    loose = dishwasher_beliefs(20, running_until=60)
    assert not policy._available(task, ("t_wash",), loose)
    policy._now = 61
    assert policy._available(task, ("t_wash",), loose)


def test_an_agent_with_nothing_else_to_do_waits_out_the_cycle_holding_its_dish():
    task = wash("t_wash", ["plate_1"])
    policy = ScriptedPolicy([], SCRIPTED, [task])
    policy.decide(request("arm_a", 0, dishwasher_beliefs(0), known=["t_wash"]))
    busy = dishwasher_beliefs(10, running_until=60, holding=True)
    picked = ActionResult(ToolCall("pick", ("plate_1",)), "ok", "picked up plate_1")
    calls = policy.decide(request("arm_a", 10, busy, ["t_wash"], picked)).tool_calls
    assert calls == [ToolCall("wait", ("50",))]


def test_a_cycle_about_to_end_is_waited_for_even_with_other_work():
    task = wash("t_wash", ["plate_1"])
    policy = ScriptedPolicy([], SCRIPTED, [task, tidy_task()])
    known = ["t_wash", "t_tidy"]
    policy.decide(request("arm_a", 0, dishwasher_beliefs(0), known=["t_wash"]))
    busy = dishwasher_beliefs(58, running_until=60, holding=True)  # 2 ticks left
    busy["sock_1"] = belief("sock", [], (3, 1))
    picked = ActionResult(ToolCall("pick", ("plate_1",)), "ok", "picked up plate_1")
    calls = policy.decide(request("arm_a", 58, busy, known, picked)).tool_calls
    assert calls == [ToolCall("wait", ("2",))]


def test_washed_dishes_seen_inside_a_dishwasher_call_for_an_unload():
    beliefs = {
        "dishwasher_1": belief("dishwasher", [("dishwasher_1", "in_room", "kitchen")], (0, 0)),
        "cabinet_1": belief("cabinet", [("cabinet_1", "in_room", "hall")], (5, 0)),
        "cabinet_2": belief("cabinet", [("cabinet_2", "in_room", "kitchen")], (1, 0)),
        "plate_1": belief("plate", [("plate_1", "in", "dishwasher_1")], (0, 0), dirty="no"),
        "plate_2": belief("plate", [("plate_2", "in", "dishwasher_1")], (0, 0), dirty="yes"),
    }
    # The wash task that covered plate_1 does not hide its unload: that is a follow-up.
    entries = find_triggered_tasks(beliefs, {("clean_dishes", "plate_1")})
    assert entries == [
        {
            "id": "unload_dishwasher_1_plate_1",
            "kind": "unpack_dishwasher",
            "items": ["plate_1"],
            "appliance": "dishwasher_1",
            "dest": "cabinet_2",
        }
    ]
    assert find_triggered_tasks(beliefs, {("unpack_dishwasher", "plate_1")}) == []


def test_a_door_already_believed_open_needs_no_call():
    task = wash("t_wash", ["plate_1"])
    policy = ScriptedPolicy([task], SCRIPTED, [task])
    policy.decide(request("arm_a", 0, dishwasher_beliefs(0)))
    beliefs = dishwasher_beliefs(5, holding=True)
    beliefs["dishwasher_1"].visual_attrs["state"] = "open"
    picked = ActionResult(ToolCall("pick", ("plate_1",)), "ok", "picked up plate_1")
    calls = policy.decide(request("arm_a", 5, beliefs, last_result=picked)).tool_calls
    assert calls[-1] == ToolCall("place", ("plate_1", "dishwasher_1"))
    assert ToolCall("open", ("dishwasher_1",)) not in calls


def test_teammates_assignments_and_items_in_a_teammates_hand_are_left_alone():
    task = wash("t_wash", ["plate_1"])
    reserved = ScriptedPolicy([], SCRIPTED, [task], frozenset({"t_wash"}))
    reserved._me = "arm_a"
    assert not reserved._available(task, ("t_wash",), dishwasher_beliefs(0))
    free = ScriptedPolicy([], SCRIPTED, [task])
    free._me = "arm_a"
    held = dishwasher_beliefs(0)
    held["plate_1"].relations = [("plate_1", "held_by", "arm_b")]
    assert free._available(task, ("t_wash",), dishwasher_beliefs(0))
    assert not free._available(task, ("t_wash",), held)


def test_a_wash_whose_dishes_all_broke_is_over_for_the_executor():
    # A break removes the dish's goal; with none left, restarting the machine would loop forever.
    task = wash("t_wash", ["plate_1"])
    voided = dataclasses.replace(task, goals=())
    assert all_items_shattered(voided, {}) and not all_items_shattered(task, {})


def test_a_split_stops_a_teammates_pick_of_a_dish_that_moved_to_the_new_task(tmp_path):
    agent = make_agent("arm_c", (6, 1))
    agent.skill = Skill(ToolCall("pick", ("plate_3",)), 2)
    agent.queue = [ToolCall("move_to", ("dishwasher_1",))]
    with EpisodeLogger(tmp_path) as logger:
        stop_picks_of(agent, ["plate_2", "plate_3"], "t_wash", 40, logger)
    assert agent.skill is None and agent.queue == [] and agent.decision_due
    assert agent.last_result.text == "t_wash split; not picking plate_3"
