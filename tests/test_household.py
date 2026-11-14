import dataclasses
import random
from pathlib import Path
from types import SimpleNamespace

from allocation.scripted import resolve_place_target
from experiments.run_single import run_config
from sim.agent import ToolCall
from sim.beliefs import initial_beliefs, update_beliefs
from sim.config import (
    BLOCKED_PATIENCE_TICKS,
    CLEAN_UP_TOOL,
    DROP_RECOVERY_TICKS,
    RETURN_TO_TASK_TOOL,
    FaultConfig,
    load_config,
)
from sim.logger import EpisodeLogger, read_stream
from sim.loop import crash, handle_breaks
from sim.motion import advance_moves, drop_held
from sim.pathing import PathPlanner
from sim.primitives import (
    check_against_beliefs,
    nearest_trashcan,
    return_to_task_area,
    start_clean_up,
)
from sim.scenario import GoalCondition, TaskSpec, load_scenario
from sim.triggers import find_triggered_tasks
from sim.world import SHARDS_TYPE
from tests.helpers import add_furniture, add_item, make_agent, make_world
from tests.test_primitives import run_call

HOME = Path("configs/m2_home.yaml")
RACK = Path("configs/m2_rack.yaml")
HOME_TASKS = {"t_unpack", "t_clean1", "t_clean2", "t_groceries", "t_trash", "clean_plate_3"}
SHATTER_SEARCH_SEEDS = 15


def read_rows(run_dir, stream):
    return read_stream(run_dir, stream)


def transitions(run_dir):
    return {(row["task"], row["transition"]) for row in read_rows(run_dir, "events")}


def add_dishwasher(world, cell=(1, 0)):
    dishwasher = add_furniture(world, "dishwasher_1", cell, container=True, is_open=False)
    dishwasher.washing_time = 60
    return dishwasher


def add_cabinet(world, cells, slot_capacity=1, is_open=True):
    cabinet = add_furniture(world, "cabinet_1", cells[0], container=True, is_open=is_open)
    cabinet.footprint = tuple(cells)
    cabinet.slot_capacity = slot_capacity
    return cabinet


def test_trashcan_counts_each_item_and_refuses_when_full():
    world = make_world(["....."])
    add_furniture(world, "trashcan_1", (1, 0), container=True).capacity = 2
    for index in range(3):
        add_item(world, f"scrap_{index}", (0, 0))
    agent = make_agent("arm_a", (0, 0))
    for index in range(2):
        assert run_call(ToolCall("pick", (f"scrap_{index}",)), agent, world).ok
        assert run_call(ToolCall("place", (f"scrap_{index}", "trashcan_1")), agent, world).ok
    assert world.contents_count("trashcan_1") == 2
    assert run_call(ToolCall("pick", ("scrap_2",)), agent, world).ok
    result = run_call(ToolCall("place", ("scrap_2", "trashcan_1")), agent, world)
    assert (result.outcome, result.text) == ("stale_belief", "trashcan_1 full")
    assert agent.held == ["scrap_2"]


def test_plates_stack_up_to_the_slot_capacity():
    world = make_world(["....", "...."])
    counter = add_furniture(world, "counter_1", (1, 0), surface=True)
    counter.slot_capacity = 3
    agent = make_agent("arm_a", (1, 1))
    for index in range(4):
        add_item(world, f"plate_{index}", (0, 1))
    for index in range(3):
        assert run_call(ToolCall("pick", (f"plate_{index}",)), agent, world).ok
        assert run_call(ToolCall("place", (f"plate_{index}", "counter_1")), agent, world).ok
    assert world.slot_count("counter_1", (1, 0)) == 3
    assert run_call(ToolCall("pick", ("plate_3",)), agent, world).ok
    assert run_call(ToolCall("place", ("plate_3", "counter_1")), agent, world).text == (
        "counter_1 full"
    )


def test_place_uses_the_closest_reachable_slot_and_names_full_ones():
    world = make_world(["......", "......"])
    add_cabinet(world, [(0, 0), (1, 0), (2, 0), (3, 0)])
    add_item(world, "cup_1", (5, 1))
    add_item(world, "cup_2", (5, 1))
    agent = make_agent("arm_a", (0, 1), hands=2)
    for cup in ("cup_1", "cup_2"):
        world.hold(cup, "arm_a", agent.cell)
        agent.held.append(cup)
    first = run_call(ToolCall("place", ("cup_1", "cabinet_1")), agent, world)
    assert first.text == "placed cup_1 in cabinet_1[0]"  # (0, 0) is closer than (1, 0)
    second = run_call(ToolCall("place", ("cup_2", "cabinet_1")), agent, world)
    assert second.text == "placed cup_2 in cabinet_1[1]"
    add_item(world, "cup_3", (0, 1))
    world.hold("cup_3", "arm_a", agent.cell)
    agent.held.append("cup_3")
    full = run_call(ToolCall("place", ("cup_3", "cabinet_1")), agent, world)
    assert (full.outcome, full.text) == ("stale_belief", "cabinet_1[0], cabinet_1[1] full")
    far = run_call(ToolCall("place", ("cup_3", "cabinet_1[3]")), agent, world)
    assert far.text == "place failed: cabinet_1[3] is not here"
    agent.cell = (2, 1)
    assert run_call(ToolCall("place", ("cup_3", "cabinet_1[3]")), agent, world).ok


def test_slot_references_are_checked_against_beliefs():
    world = make_world(["......", "......"])
    add_cabinet(world, [(0, 0), (1, 0), (2, 0)])
    agent = make_agent("arm_a", (0, 1))
    agent.beliefs = initial_beliefs(world)
    assert check_against_beliefs(ToolCall("move_to", ("cabinet_1[2]",)), agent, ["arm_a"]) is None
    bad_slot = check_against_beliefs(ToolCall("move_to", ("cabinet_1[3]",)), agent, ["arm_a"])
    assert bad_slot.outcome == "invalid_call"
    unknown = check_against_beliefs(ToolCall("move_to", ("cabinet_9[0]",)), agent, ["arm_a"])
    assert unknown.outcome == "hallucination"
    not_a_slot_arg = check_against_beliefs(ToolCall("open", ("cabinet_1[0]",)), agent, ["arm_a"])
    assert not_a_slot_arg.outcome == "hallucination"


def test_slot_fill_is_unknown_until_seen_and_remembered_after_closing():
    world = make_world(["....", "...."])
    cabinet = add_cabinet(world, [(0, 0), (1, 0)], is_open=False)
    add_item(world, "cup_1", (0, 0), parent_id="cabinet_1")
    agent = make_agent("arm_a", (1, 1))
    agent.beliefs = initial_beliefs(world)
    assert "slots" not in agent.beliefs["cabinet_1"].visual_attrs
    cabinet.is_open = True
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    assert agent.beliefs["cabinet_1"].visual_attrs["slots"] == "1/1,0/1"
    cabinet.is_open = False
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    assert agent.beliefs["cabinet_1"].visual_attrs["slots"] == "1/1,0/1"


def test_lock_reserves_the_workspace_against_other_agents():
    world = make_world([".......", ".......", "......."])
    add_dishwasher(world, (3, 0))
    holder = make_agent("arm_a", (3, 1))
    other = make_agent("arm_b", (5, 1))
    team = {"arm_a": holder, "arm_b": other}
    assert run_call(ToolCall("lock", ("dishwasher_1",)), holder, world, team).ok
    # The dishwasher cell plus every free floor cell around it.
    assert sorted(world.reservations) == [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
    blocked = run_call(ToolCall("open", ("dishwasher_1",)), other, world, team)
    assert (blocked.outcome, blocked.text) == ("stale_belief", "dishwasher_1 is reserved by arm_a")
    assert run_call(ToolCall("lock", ("dishwasher_1",)), other, world, team).outcome == (
        "stale_belief"
    )
    assert run_call(ToolCall("unlock", ("dishwasher_1",)), other, world, team).outcome == (
        "invalid_call"
    )
    assert run_call(ToolCall("open", ("dishwasher_1",)), holder, world, team).ok
    assert run_call(ToolCall("unlock", ("dishwasher_1",)), holder, world, team).ok
    assert world.reservations == {}


def test_moving_agents_route_around_reserved_floor_but_static_steps_do_not_change():
    world = make_world([".......", ".......", "......."])
    add_furniture(world, "table_1", (6, 1), surface=True)
    world.reservations = {(3, 0): ("arm_b", "x"), (3, 1): ("arm_b", "x")}
    agent = make_agent("arm_a", (0, 1))
    agent.beliefs = initial_beliefs(world)
    planner = PathPlanner(world)
    assert planner.steps((0, 1), (6, 1)) == 5
    assert run_call(ToolCall("move_to", ("table_1",)), agent, world).ok
    assert run_call.ticks == 6  # the detour through row 2 costs one extra step
    assert planner.steps((0, 1), (6, 1)) == 5


def test_agents_block_each_other_and_never_swap_cells():
    world = make_world(["#####", ".....", "#####"])
    add_furniture(world, "shelf_1", (4, 1), surface=True)
    walker = make_agent("arm_a", (0, 1))
    walker.beliefs = initial_beliefs(world)
    blocker = make_agent("arm_b", (2, 1))
    team = {"arm_a": walker, "arm_b": blocker}
    result = run_call(ToolCall("move_to", ("shelf_1",)), walker, world, team)
    assert (result.outcome, result.text) == ("stale_belief", "path to shelf_1 blocked by arm_b")
    # The walker closes in along its path, queues behind arm_b, and never enters arm_b's cell.
    assert (walker.cell, blocker.cell) == ((1, 1), (2, 1))
    assert walker.blocked_ticks == 6


def test_an_idle_agent_in_the_way_steps_aside_but_a_busy_one_does_not():
    world = make_world(["#####", ".....", "#.###"])
    add_furniture(world, "shelf_1", (4, 1), surface=True)
    walker = make_agent("arm_a", (0, 1))
    walker.beliefs = initial_beliefs(world)
    idle = make_agent("arm_b", (1, 1))  # nothing queued: yields into the side pocket at (1, 2)
    team = {"arm_a": walker, "arm_b": idle}
    assert run_call(ToolCall("move_to", ("shelf_1",)), walker, world, team).ok
    assert (idle.cell, idle.yields) == ((1, 2), 1)
    busy = make_agent("arm_b", (2, 1))
    busy.queue = [ToolCall("wait", ("5",))]
    walker.cell = (0, 1)
    result = run_call(
        ToolCall("move_to", ("shelf_1",)), walker, world, {"arm_a": walker, "arm_b": busy}
    )
    assert result.text == "path to shelf_1 blocked by arm_b"
    assert busy.cell == (2, 1)


def test_starting_a_cycle_hands_the_lock_to_the_appliance_until_it_finishes():
    world = make_world([".....", "....."])
    dishwasher = add_dishwasher(world)
    agent = make_agent("arm_a", (1, 1))
    other = make_agent("arm_b", (2, 1))
    team = {"arm_a": agent, "arm_b": other}
    assert run_call(ToolCall("lock", ("dishwasher_1",)), agent, world, team).ok
    assert run_call(ToolCall("start", ("dishwasher_1",)), agent, world, team).ok
    assert {holder for holder, _ in world.reservations.values()} == {"dishwasher_1"}
    blocked = run_call(ToolCall("open", ("dishwasher_1",)), other, world, team)
    assert blocked.text == "dishwasher_1 is reserved by dishwasher_1"
    for _ in range(60):
        world.advance_cycles()
    assert dishwasher.cycle == "done"
    assert world.reservations == {}
    assert run_call(ToolCall("open", ("dishwasher_1",)), other, world, team).ok


def test_wash_cycle_runs_for_washing_time_and_cleans_contents():
    world = make_world(["....."])
    dishwasher = add_dishwasher(world)
    dishwasher.is_open = True
    plate = add_item(world, "plate_1", (1, 0), parent_id="dishwasher_1")
    plate.visual_attrs["dirty"] = "yes"
    agent = make_agent("arm_a", (0, 0))
    assert run_call(ToolCall("start", ("dishwasher_1",)), agent, world).outcome == "stale_belief"
    assert run_call(ToolCall("close", ("dishwasher_1",)), agent, world).ok
    assert run_call(ToolCall("start", ("dishwasher_1",)), agent, world).ok
    assert run_call(ToolCall("open", ("dishwasher_1",)), agent, world).outcome == "stale_belief"
    for _ in range(59):
        world.advance_cycles()
    assert dishwasher.cycle == "running"
    world.advance_cycles()
    assert dishwasher.cycle == "done"
    assert plate.visual_attrs["dirty"] == "no"
    assert run_call(ToolCall("open", ("dishwasher_1",)), agent, world).ok
    assert dishwasher.cycle == "idle"


def test_a_dropped_plate_shatters_on_the_agent_tile_and_awaits_the_reflex():
    world = make_world(["...."])
    add_item(world, "plate_1", (0, 0))
    agent = make_agent("arm_a", (1, 0), hands=2)
    result = run_call(ToolCall("pick", ("plate_1",)), agent, world, drop_prob=1.0)
    assert result.outcome == "skill_failure"
    assert result.text == "plate_1 dropped right after pickup and shattered"
    assert world.objects["plate_1"].obj_type == SHARDS_TYPE
    assert world.objects["plate_1"].cell == (1, 0)
    assert agent.held == [] and agent.shards == "plate_1"
    assert world.broken == ["plate_1"] and world.lost == {"plate_1"}
    assert world.drops == [("arm_a", "plate_1")]


def test_a_non_fragile_drop_is_only_a_pause():
    world = make_world(["...."])
    add_furniture(world, "shelf_1", (3, 0), surface=True)
    add_item(world, "pot_1", (0, 0))
    agent = make_agent("arm_a", (1, 0))
    agent.beliefs = initial_beliefs(world)
    picked = run_call(ToolCall("pick", ("pot_1",)), agent, world, drop_prob=1.0)
    assert picked.ok and agent.held == ["pot_1"]
    assert agent.thinking_until == world.tick + 1 + DROP_RECOVERY_TICKS
    agent.cell = (2, 0)
    placed = run_call(ToolCall("place", ("pot_1", "shelf_1")), agent, world, drop_prob=1.0)
    assert placed.ok and world.objects["pot_1"].parent_id == "shelf_1"
    assert len(world.drops) == 2 and world.objects["pot_1"].obj_type == "pot"


def test_a_non_fragile_drop_while_moving_costs_exactly_the_pause():
    ticks = {}
    for drop_prob in (0.0, 1.0):
        world = make_world([".........."])
        add_furniture(world, "shelf_1", (9, 0), surface=True)
        add_item(world, "pot_1", (0, 0))
        agent = make_agent("arm_a", (0, 0))
        agent.beliefs = initial_beliefs(world)
        world.hold("pot_1", "arm_a", agent.cell)
        agent.held.append("pot_1")
        call = ToolCall("move_to", ("shelf_1",))
        moved = run_call(call, agent, world, drop_prob=drop_prob, seed=3)
        assert moved.ok and agent.held == ["pot_1"]
        ticks[drop_prob] = run_call.ticks
    assert ticks[1.0] == ticks[0.0] + DROP_RECOVERY_TICKS


def test_tie_bag_collapses_the_trash_into_one_carriable_bag():
    world = make_world(["...."])
    add_furniture(world, "trashcan_1", (1, 0), container=True).capacity = 5
    add_item(world, "scrap_1", (1, 0), parent_id="trashcan_1")
    add_item(world, "scrap_2", (1, 0), parent_id="trashcan_1")
    bag = add_item(world, "trashcan_1_bag", (1, 0), parent_id="trashcan_1")
    bag.obj_type = "trash_bag"
    agent = make_agent("arm_a", (0, 0))
    assert run_call(ToolCall("tie_bag", ("trashcan_1",)), agent, world).ok
    assert sorted(world.objects) == ["trashcan_1", "trashcan_1_bag"]
    assert bag.visual_attrs["tied"] == "yes"
    assert run_call(ToolCall("pick", ("trashcan_1_bag",)), agent, world).ok


def test_triggers_fire_from_beliefs_only_and_ids_are_deterministic():
    world = make_world(["......", "......"])
    add_furniture(world, "trashcan_1", (5, 0), container=True).capacity = 4
    add_dishwasher(world, (0, 0))
    add_item(world, "plate_1", (2, 1)).visual_attrs["dirty"] = "yes"
    add_item(world, "plate_2", (0, 0), parent_id="dishwasher_1").visual_attrs["dirty"] = "yes"
    add_item(world, "plate_3", (3, 1))
    world.break_object("plate_3")  # shards are not a task: the reflex cleans them up
    first = make_agent("arm_a", (2, 0))
    second = make_agent("arm_b", (3, 0))
    for agent in (first, second):
        agent.beliefs = initial_beliefs(world)
        update_beliefs(agent.beliefs, world, agent.viewpoint())
    found = find_triggered_tasks(first.beliefs, set())
    # plate_2 is in the closed dishwasher, so nobody can see it to trigger a task.
    assert [entry["id"] for entry in found] == ["clean_plate_1"]
    assert found == find_triggered_tasks(second.beliefs, set())
    assert find_triggered_tasks(first.beliefs, {("clean_plate", "plate_1")}) == []


def test_trash_trigger_needs_a_gauge_above_half():
    world = make_world(["....."])
    can = add_furniture(world, "trashcan_1", (0, 0), container=True, is_open=False)
    can.capacity = 4
    add_furniture(world, "bin_1", (4, 0), container=True)
    for index in range(2):
        add_item(world, f"scrap_{index}", (0, 0), parent_id="trashcan_1")
    agent = make_agent("arm_a", (2, 0))
    agent.beliefs = initial_beliefs(world)
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    assert find_triggered_tasks(agent.beliefs, set()) == []
    add_item(world, "scrap_2", (0, 0), parent_id="trashcan_1")
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    assert [entry["id"] for entry in find_triggered_tasks(agent.beliefs, set())] == [
        "trash_trashcan_1"
    ]


def test_place_target_falls_back_to_the_rack_or_an_unseen_cupboard():
    world = make_world(["......", "......"])
    add_cabinet(world, [(0, 0), (1, 0)])
    other = add_furniture(world, "cabinet_2", (4, 0), container=True, is_open=False)
    other.slot_capacity = 1
    add_furniture(world, "drying_rack_1", (5, 1), surface=True).obj_type = "drying_rack"
    add_item(world, "cup_1", (0, 0), parent_id="cabinet_1")
    add_item(world, "bowl_1", (1, 1))
    agent = make_agent("arm_a", (1, 1))
    agent.beliefs = initial_beliefs(world)
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    beliefs = agent.beliefs
    assert resolve_place_target("bowl_1", "cabinet_1", "unpack_dishwasher", beliefs) == (
        "cabinet_1[1]"
    )
    add_item(world, "cup_2", (1, 0), parent_id="cabinet_1").cell = (1, 0)  # slot 1
    update_beliefs(beliefs, world, agent.viewpoint())
    assert resolve_place_target("bowl_1", "cabinet_1", "unpack_dishwasher", beliefs) == (
        "drying_rack_1"
    )
    assert resolve_place_target("bowl_1", "cabinet", "store_dish", beliefs) == "cabinet_2"


def test_templates_unroll_into_flat_steps_and_goals():
    scenario = load_scenario(
        Path("configs/maps/home_medium.txt"), Path("configs/missions/home_day.yaml")
    )
    unpack = next(task for task in scenario.tasks if task.task_id == "t_unpack")
    assert unpack.kind == "unpack_dishwasher"
    assert [call.name for call in unpack.steps] == [
        "lock", "open", "open", "pick", "place", "pick", "place", "close", "close", "unlock",
    ]  # fmt: skip
    assert [(goal.obj_id, goal.value) for goal in unpack.goals] == [
        ("bowl_1", "cabinet|drying_rack"),
        ("bowl_2", "cabinet|drying_rack"),
    ]
    assert scenario.world.objects["cabinet_2"].cells() == ((6, 6), (7, 6), (8, 6))


def test_home_mission_completes_with_discovery_and_contention(tmp_path):
    config = dataclasses.replace(
        load_config(HOME), output_dir=tmp_path, faults=FaultConfig(drop_prob=0.0)
    )
    run_dir, final_tick = run_config(config)
    done = {task for task, transition in transitions(run_dir) if transition == "done"}
    # Washed plates are unloaded (and racked plates stored) as follow-up tasks.
    assert done >= HOME_TASKS
    assert all(task.startswith(("unload_", "store_")) for task in done - HOME_TASKS)
    assert ("clean_plate_3", "discovered") in transitions(run_dir)
    assert final_tick < config.max_ticks - 1
    texts = [row["text"] for row in read_rows(run_dir, "actions")]
    # Teammates are kept off the dishwasher while a cycle holds its workspace.
    assert any("reserved by" in text or "blocked by dishwasher_1" in text for text in texts)


def test_home_mission_finishes_under_the_default_drop_rate(tmp_path):
    config = load_config(HOME)
    assert config.faults.drop_prob == 0.025
    for seed in range(4):
        seeded = dataclasses.replace(config, output_dir=tmp_path, seed=seed)
        _, final_tick = run_config(seeded)
        assert final_tick < config.max_ticks - 1, seed


def test_a_broken_single_item_voids_its_task_and_the_breaker_cleans_up_by_reflex(tmp_path):
    # At a high drop rate, find a seed where plate_2 shatters; it happens in about 1 seed in 3.
    for seed in range(SHATTER_SEARCH_SEEDS):
        config = dataclasses.replace(
            load_config(HOME), output_dir=tmp_path, seed=seed, faults=FaultConfig(0.2)
        )
        run_dir, final_tick = run_config(config)
        events = transitions(run_dir)
        if ("t_clean2", "voided") in events:
            break
    assert ("t_clean2", "voided") in events, "no seed shattered plate_2"
    assert ("t_clean2", "done") not in events
    assert not [task for task, _ in events if task and task.startswith("cleanup")]
    assert final_tick < config.max_ticks - 1
    actions = read_rows(run_dir, "actions")
    cleanups = [row for row in actions if row["call"][0] == CLEAN_UP_TOOL]
    assert cleanups and all(row["outcome"] == "ok" for row in cleanups)
    final_objects = read_rows(run_dir, "state")[-1]["objects"]
    for row in cleanups:
        # Shards end in the can, or inside its bag once someone has tied it (and are gone as items).
        assert row["call"][1] not in final_objects or (
            final_objects[row["call"][1]]["parent"] == row["call"][2]
        )
    # The reflex takes no LLM call: nothing is decided between its start and its end.
    calls = read_rows(run_dir, "llm_calls")
    for row in cleanups:
        assert not [
            call
            for call in calls
            if call["agent"] == row["agent"] and row["start_tick"] < call["tick"] <= row["tick"]
        ]


def test_a_broken_item_leaves_the_rest_of_its_task_with_the_same_agent(tmp_path):
    # A generated wash task covers a cluster of dishes, so one break leaves the rest to finish.
    generated = Path("configs/generated/house_48x32_s1/episode_greedy.yaml")
    for seed in range(SHATTER_SEARCH_SEEDS):
        config = dataclasses.replace(
            load_config(generated), output_dir=tmp_path, seed=seed, faults=FaultConfig(0.1)
        )
        run_dir, final_tick = run_config(config)
        rows = read_rows(run_dir, "events")
        broken = {row["obj"] for row in rows if row["transition"] == "broken"}
        mission = load_scenario(config.map_path, config.mission_path)
        washes = {
            task.task_id: {goal.obj_id for goal in task.goals if goal.relation == "washed"}
            for task in mission.tasks
            if task.kind == "clean_dishes"
        }
        hit = [task for task, items in washes.items() if 0 < len(items & broken) < len(items)]
        if hit:
            break
    assert hit, "no seed broke part of a cluster"
    assert final_tick < config.max_ticks - 1
    done = {row["task"] for row in rows if row["transition"] == "done"}
    voided = {row["task"] for row in rows if row["transition"] == "voided"}
    assert hit[0] in done and hit[0] not in voided
    state = read_rows(run_dir, "state")[-1]["objects"]
    for survivor in washes[hit[0]] - broken:
        assert state[survivor]["parent"].startswith("dishwasher")


def reflex_run(tmp_path, seeds=SHATTER_SEARCH_SEEDS):
    """A kitchen run at a high drop rate in which at least one dish shatters."""
    for seed in range(seeds):
        config = dataclasses.replace(
            load_config(HOME), output_dir=tmp_path, seed=seed, faults=FaultConfig(0.2)
        )
        run_dir, _ = run_config(config)
        actions = read_rows(run_dir, "actions")
        if any(row["call"][0] == CLEAN_UP_TOOL for row in actions):
            return run_dir, actions
    raise AssertionError("no seed shattered a dish")


def test_the_reflex_returns_to_the_task_area_before_any_decision(tmp_path):
    run_dir, actions = reflex_run(tmp_path)
    calls = read_rows(run_dir, "llm_calls")
    for index, row in enumerate(actions):
        if row["call"][0] != CLEAN_UP_TOOL:
            continue
        back = next(
            later
            for later in actions[index + 1 :]
            if later["agent"] == row["agent"] and later["call"][0] == RETURN_TO_TASK_TOOL
        )
        assert back["outcome"] == "ok" and back["call"][1:] == row["call"][1:]
        # Nothing is decided from the shatter until the agent is back.
        assert not [
            call
            for call in calls
            if call["agent"] == row["agent"] and row["start_tick"] < call["tick"] <= back["tick"]
        ]


def test_a_blocked_return_keeps_trying_and_never_hands_back_a_failure():
    world = make_world(["#####", ".....", "#####"])
    walker = make_agent("arm_a", (0, 1))
    walker.task_area = (4, 1)
    return_to_task_area(walker, world, ToolCall(CLEAN_UP_TOOL, ("shards_1", "trashcan_1")))
    busy = make_agent("arm_b", (2, 1))
    busy.queue = [ToolCall("wait", ("5",))]
    team = {"arm_a": walker, "arm_b": busy}
    planner = PathPlanner(world)
    rng = random.Random(0)
    for _ in range(3 * BLOCKED_PATIENCE_TICKS):
        assert advance_moves(team, world, planner, rng) == {}
        world.tick += 1
    assert walker.skill.call.name == RETURN_TO_TASK_TOOL and walker.cell == (1, 1)
    busy.cell = (2, 0)  # the way clears
    busy.queue = []
    results = {}
    for _ in range(5):
        results |= advance_moves({"arm_a": walker}, world, planner, rng)
        world.tick += 1
    assert results["arm_a"].ok


def test_the_reflex_finds_the_trashcan_without_the_agent_ever_having_seen_it(tmp_path):
    world = make_world(["........", "........"])
    add_furniture(world, "trashcan_1", (7, 0), container=True).capacity = 5
    add_furniture(world, "trashcan_2", (7, 1), container=True).capacity = 5
    add_item(world, "plate_1", (0, 0))
    agent = make_agent("arm_a", (1, 0))
    assert agent.beliefs == {}  # no map knowledge, no sightings
    planner = PathPlanner(world)
    assert nearest_trashcan(agent.cell, world, planner) == "trashcan_1"
    world.hold("plate_1", "arm_a", agent.cell)
    agent.held.append("plate_1")
    assert drop_held(ToolCall("pick", ("plate_1",)), agent, world, "plate_1", "here") is not None
    assert start_clean_up(agent, world, planner)
    assert agent.skill.call.name == CLEAN_UP_TOOL and agent.task_area == (1, 0)
    assert agent.held == ["plate_1"] and agent.queue == []


def test_a_map_with_no_trashcan_leaves_the_shards_and_hands_back_a_decision():
    world = make_world(["...."])
    add_item(world, "plate_1", (0, 0))
    agent = make_agent("arm_a", (1, 0))
    world.hold("plate_1", "arm_a", agent.cell)
    agent.held.append("plate_1")
    drop_held(ToolCall("pick", ("plate_1",)), agent, world, "plate_1", "here")
    assert not start_clean_up(agent, world, PathPlanner(world))
    assert agent.skill is None and agent.shards is None and agent.held == []


def test_a_crash_leaves_shards_as_debris_with_no_task(tmp_path):
    world = make_world(["...."])
    add_item(world, "plate_1", (0, 0))
    add_item(world, "pot_1", (0, 0))
    agent = make_agent("arm_a", (1, 0), hands=2)
    for obj_id in ("plate_1", "pot_1"):
        world.hold(obj_id, "arm_a", agent.cell)
        agent.held.append(obj_id)
    with EpisodeLogger(tmp_path) as logger:
        crash(agent, world, logger)
    assert world.objects["plate_1"].obj_type == SHARDS_TYPE
    assert world.objects["plate_1"].cell == (1, 0)
    assert world.objects["pot_1"].held_by is None and world.objects["pot_1"].obj_type == "pot"
    assert agent.held == [] and agent.shards is None and world.lost == {"plate_1"}


def test_breaking_an_item_removes_only_its_goals_and_voids_only_an_emptied_task(tmp_path):
    world = make_world(["...."])
    add_furniture(world, "dishwasher_1", (0, 0), container=True)
    add_item(world, "plate_1", (1, 0))
    add_item(world, "plate_2", (2, 0))
    add_item(world, "plate_3", (3, 0))
    cycle = GoalCondition("dishwasher_1", "cycle", "done")
    batch = TaskSpec(
        "t_batch",
        "wash",
        (
            GoalCondition("plate_1", "in", "dishwasher_1"),
            GoalCondition("plate_2", "in", "dishwasher_1"),
            cycle,
        ),
        (),
        (),
    )
    single = TaskSpec(
        "t_single", "wash", (GoalCondition("plate_3", "in", "dishwasher_1"), cycle), (), ()
    )
    scenario = SimpleNamespace(tasks=[batch, single])
    voided: set[str] = set()
    world.break_object("plate_1")
    world.break_object("plate_3")
    with EpisodeLogger(tmp_path) as logger:
        handle_breaks(world, scenario, {}, voided, logger)
    assert voided == {"t_single"}
    assert [goal.obj_id for goal in scenario.tasks[0].goals] == ["plate_2", "dishwasher_1"]
    assert world.broken == []


def test_full_cupboard_routes_dishes_through_the_rack_to_another_cupboard(tmp_path):
    config = dataclasses.replace(load_config(RACK), output_dir=tmp_path)
    scenario = load_scenario(config.map_path, config.mission_path)
    run_dir, final_tick = run_config(config)
    events = transitions(run_dir)
    for bowl in ("bowl_1", "bowl_2"):
        assert (f"store_{bowl}", "discovered") in events
        assert (f"store_{bowl}", "done") in events
    assert final_tick < config.max_ticks - 1
    final_state = read_rows(run_dir, "state")[-1]["objects"]
    assert {final_state[bowl]["parent"] for bowl in ("bowl_1", "bowl_2")} == {"cabinet_2"}
    assert scenario.world.objects["cabinet_2"].slot_capacity == 1
