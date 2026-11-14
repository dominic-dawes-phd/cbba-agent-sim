import random

from sim.agent import ToolCall
from sim.beliefs import update_beliefs
from sim.motion import advance_moves
from sim.pathing import PathPlanner
from sim.primitives import advance_skill, check_against_beliefs, start_skill
from tests.helpers import add_furniture, add_item, make_agent, make_world


def run_call(call, agent, world, agents=None, drop_prob=0.0, seed=0):
    """Starts a call and advances it until it finishes. The channel is unused by these calls.

    Returns:
        The result, with `.ticks` set to how many ticks the skill took.
    """
    planner = PathPlanner(world)
    rng = random.Random(seed)
    agent.skill = start_skill(call, agent, planner, rng, drop_prob)
    team = agents or {agent.agent_id: agent}
    ticks = 0
    while True:
        ticks += 1
        world.tick += 1
        if call.name == "move_to":
            result = advance_moves(team, world, planner, rng).get(agent.agent_id)
        else:
            result = advance_skill(agent, team, world, planner, None, rng, drop_prob)
        if result is not None:
            agent.skill = None
            run_call.ticks = ticks
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


def test_fumbled_place_keeps_the_object_in_hand():
    world = make_world(["....."])
    add_furniture(world, "shelf_1", (2, 0), surface=True)
    add_item(world, "mug_1", (0, 0))
    clumsy = make_agent("arm_a", (1, 0), success={"place": 0.0})
    assert run_call(ToolCall("pick", ("mug_1",)), clumsy, world).ok
    result = run_call(ToolCall("place", ("mug_1", "shelf_1")), clumsy, world)
    assert (result.outcome, result.text) == (
        "skill_failure",
        "place failed: fumbled, still holding mug_1",
    )
    assert clumsy.held == ["mug_1"]


def test_drop_rule_lands_on_the_agents_own_tile():
    world = make_world([".....", "....."])
    add_furniture(world, "table_1", (0, 0), surface=True)
    add_item(world, "mug_1", (4, 1))
    world.hold("mug_1", "arm_a", (1, 1))
    world.drop("mug_1", (1, 1))  # next to the table, still lands on the floor
    assert world.objects["mug_1"].cell == (1, 1)
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
    result = run_call(ToolCall("move_to", ("mug_1",)), agent, world)
    assert result.ok
    assert run_call.ticks == 3  # 6 steps to (6, 0) at 2 cells per tick
    assert agent.cell == (6, 0)
