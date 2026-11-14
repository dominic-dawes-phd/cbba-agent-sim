import random

from sim.agent import ToolCall
from sim.beliefs import initial_beliefs
from sim.config import BACKOFF_AFTER_STUCK_TICKS
from sim.motion import advance_moves
from sim.pathing import PathPlanner
from sim.primitives import start_skill
from tests.helpers import add_furniture, make_agent, make_world

# Two agents at the top corners of a T; both must pass through (1, 0) to reach the shelf at (1, 4).
T_JUNCTION = ["...", "#.#", "#.#", "#.#", "#.#"]


def start_move(agent, target, planner):
    agent.skill = start_skill(ToolCall("move_to", (target,)), agent, planner, random.Random(0), 0.0)


def t_junction():
    world = make_world(T_JUNCTION)
    add_furniture(world, "shelf_1", (1, 4), surface=True)
    planner = PathPlanner(world)
    left = make_agent("arm_a", (0, 0))
    right = make_agent("arm_b", (2, 0))
    for agent in (left, right):
        agent.beliefs = initial_beliefs(world)
        start_move(agent, "shelf_1", planner)
    return world, planner, {"arm_a": left, "arm_b": right}


def test_a_contested_cell_goes_to_exactly_one_agent_chosen_by_the_seed():
    winners = []
    for seed in range(2):
        world, planner, team = t_junction()
        advance_moves(team, world, planner, random.Random(seed))
        on_junction = [agent_id for agent_id, agent in team.items() if agent.cell == (1, 0)]
        assert len(on_junction) == 1
        loser = "arm_b" if on_junction[0] == "arm_a" else "arm_a"
        assert team[loser].move_conflicts == 1
        winners.append(on_junction[0])
    world, planner, team = t_junction()
    advance_moves(team, world, planner, random.Random(0))
    assert team[winners[0]].cell == (1, 0)  # same seed, same winner


def test_a_mover_stuck_for_a_while_sits_out_some_rounds_at_random():
    # Two speed-2 movers detouring in step undid each other's moves every tick, forever (found in a
    # generated house). A coin flip for movers that have made no progress breaks the symmetry.
    moved = []
    for seed in range(30):
        world, planner, team = t_junction()
        agent = team["arm_a"]
        del team["arm_b"]
        agent.skill.stuck_ticks = BACKOFF_AFTER_STUCK_TICKS
        advance_moves(team, world, planner, random.Random(seed))
        moved.append(agent.cell != (0, 0))
    assert any(moved) and not all(moved)
    for seed in range(30):
        world, planner, team = t_junction()
        del team["arm_b"]
        advance_moves(team, world, planner, random.Random(seed))
        assert team["arm_a"].cell != (0, 0)  # a mover making progress never sits out


def test_conflicts_are_won_without_id_bias():
    wins = 0
    for seed in range(200):
        world, planner, team = t_junction()
        advance_moves(team, world, planner, random.Random(seed))
        wins += team["arm_a"].cell == (1, 0)
    assert 80 <= wins <= 120


def test_both_agents_get_through_the_junction_one_after_the_other():
    world, planner, team = t_junction()
    rng = random.Random(0)
    arrived = set()
    for _ in range(20):
        for agent_id in advance_moves(team, world, planner, rng):
            arrived.add(agent_id)
            team[agent_id].skill = None
        assert team["arm_a"].cell != team["arm_b"].cell
    # Only one cell is next to the shelf, so the first to arrive holds it and the other waits.
    assert len(arrived) == 2


def test_head_on_agents_never_swap_or_overlap():
    world = make_world(["......"])
    add_furniture(world, "shelf_w", (0, 0), surface=True)
    add_furniture(world, "shelf_e", (5, 0), surface=True)
    planner = PathPlanner(world)
    west_bound = make_agent("arm_a", (3, 0))
    east_bound = make_agent("arm_b", (2, 0))
    team = {"arm_a": west_bound, "arm_b": east_bound}
    for agent, target in ((west_bound, "shelf_w"), (east_bound, "shelf_e")):
        agent.beliefs = initial_beliefs(world)
        start_move(agent, target, planner)
    results = {}
    for _ in range(10):
        results.update(advance_moves(team, world, planner, random.Random(0)))
        assert east_bound.cell[0] < west_bound.cell[0]  # never passed through each other
    assert {result.outcome for result in results.values()} == {"stale_belief"}


def test_a_fast_agent_one_step_away_moves_one_cell_and_stops():
    world = make_world(["....."])
    add_furniture(world, "shelf_1", (4, 0), surface=True)
    planner = PathPlanner(world)
    fast = make_agent("carrier_c", (1, 0), speed=2)
    fast.beliefs = initial_beliefs(world)
    start_move(fast, "shelf_1", planner)
    results = advance_moves({"carrier_c": fast}, world, planner, random.Random(0))
    assert results["carrier_c"].ok
    assert fast.cell == (3, 0)  # next to the shelf, never on it
    assert fast.skill.steps_taken == 2
    start_move(fast, "shelf_1", planner)
    fast.cell = (2, 0)
    results = advance_moves({"carrier_c": fast}, world, planner, random.Random(0))
    assert results["carrier_c"].ok and fast.cell == (3, 0)
    assert fast.skill.steps_taken == 1
