import random

from sim.agent import ToolCall
from sim.beliefs import update_beliefs
from sim.pathing import PathPlanner
from sim.primitives import advance_skill, check_against_beliefs, start_skill
from tests.helpers import add_furniture, add_item, make_agent, make_world


def run_call(call, agent, world, agents=None):
    """Starts a call and advances it until it finishes. The channel is unused by these calls."""
    planner = PathPlanner(world)
    agent.skill = start_skill(call, agent, planner)
    rng = random.Random(0)
    while True:
        result = advance_skill(agent, agents or {agent.agent_id: agent}, world, planner, None, rng)
        if result is not None:
            agent.skill = None
            return result


def test_unknown_object_is_a_hallucination():
    agent = make_agent("arm_a", (0, 0))
    result = check_against_beliefs(ToolCall("pick", ("plate_9",)), agent, ["arm_a"])
    assert result.outcome == "hallucination"


def test_bad_tool_and_arity_are_invalid_calls():
    agent = make_agent("arm_a", (0, 0))
    assert check_against_beliefs(ToolCall("fly", ()), agent, ["arm_a"]).outcome == "invalid_call"
    assert check_against_beliefs(ToolCall("wait", ()), agent, ["arm_a"]).outcome == "invalid_call"
    assert (
        check_against_beliefs(ToolCall("wait", ("x",)), agent, ["arm_a"]).outcome == "invalid_call"
    )
    assert check_against_beliefs(ToolCall("wait", ("3",)), agent, ["arm_a"]) is None


def test_pick_of_moved_object_is_a_stale_belief():
    world = make_world(["......"])
    add_item(world, "plate_1", (1, 0))
    agent = make_agent("arm_a", (0, 0))
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    assert check_against_beliefs(ToolCall("pick", ("plate_1",)), agent, ["arm_a"]) is None
    world.objects["plate_1"].cell = (5, 0)  # moved after the agent last looked
    result = run_call(ToolCall("pick", ("plate_1",)), agent, world)
    assert result.outcome == "stale_belief"
    assert agent.held == []


def test_two_hand_object_needs_two_free_hands():
    world = make_world(["..."])
    add_item(world, "pot_1", (1, 0), hands=2)
    one_hand = make_agent("arm_b", (0, 0), hands=1)
    assert run_call(ToolCall("pick", ("pot_1",)), one_hand, world).outcome == "invalid_call"
    two_hands = make_agent("arm_a", (0, 0), hands=2)
    assert run_call(ToolCall("pick", ("pot_1",)), two_hands, world).ok
    assert world.objects["pot_1"].held_by == "arm_a"


def test_failed_place_follows_drop_rule():
    world = make_world(["....."])
    add_furniture(world, "table_1", (1, 0), surface=True)
    add_furniture(world, "shelf_1", (4, 0), surface=True)
    add_item(world, "plate_1", (0, 0))
    clumsy = make_agent("arm_a", (0, 0), success={"place": 0.0})
    assert run_call(ToolCall("pick", ("plate_1",)), clumsy, world).ok
    clumsy.cell = (3, 0)  # next to the shelf and not next to the table
    result = run_call(ToolCall("place", ("plate_1", "shelf_1")), clumsy, world)
    assert result.outcome == "skill_failure"
    assert result.text == "place failed: plate_1 dropped, now on shelf_1"
    assert clumsy.held == []


def test_drop_rule_prefers_adjacent_surface_else_floor():
    world = make_world([".....", "....."])
    add_furniture(world, "table_1", (0, 0), surface=True)
    add_item(world, "mug_1", (4, 1))
    world.hold("mug_1", "arm_a", (1, 1))
    assert world.drop("mug_1", (1, 1)) == "on table_1"
    assert world.objects["mug_1"].parent_id == "table_1"
    world.hold("mug_1", "arm_a", (4, 1))
    assert world.drop("mug_1", (4, 1)) == "on the floor"
    assert world.objects["mug_1"].cell == (4, 1)
    assert world.objects["mug_1"].parent_id is None


def test_place_into_closed_container_is_a_stale_belief():
    world = make_world(["..."])
    add_furniture(world, "dishwasher_1", (1, 0), container=True, is_open=False)
    add_item(world, "plate_1", (0, 0))
    agent = make_agent("arm_a", (0, 0))
    assert run_call(ToolCall("pick", ("plate_1",)), agent, world).ok
    result = run_call(ToolCall("place", ("plate_1", "dishwasher_1")), agent, world)
    assert result.outcome == "stale_belief"
    assert agent.held == ["plate_1"]


def test_move_to_takes_ceil_steps_over_speed_ticks():
    world = make_world(["........"])
    add_item(world, "mug_1", (7, 0))
    agent = make_agent("drone_c", (0, 0), speed=2)
    update_beliefs(agent.beliefs, world, agent.viewpoint())
    planner = PathPlanner(world)
    agent.skill = start_skill(ToolCall("move_to", ("mug_1",)), agent, planner)
    assert agent.skill.ticks_left == 3  # 6 steps to (6, 0) at 2 cells per tick
    result = run_call(ToolCall("move_to", ("mug_1",)), agent, world)
    assert result.ok
    assert agent.cell == (6, 0)
