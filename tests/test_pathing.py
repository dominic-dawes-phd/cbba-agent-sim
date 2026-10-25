from sim.pathing import PathPlanner, travel_ticks
from tests.helpers import add_furniture, make_world


def test_open_grid_distance_reaches_adjacent_cell():
    world = make_world(["....."] * 5)
    planner = PathPlanner(world)
    assert planner.steps((0, 0), (4, 4)) == 6  # to (3, 3), diagonal to the target
    assert planner.steps((4, 4), (4, 4)) == 0


def test_wall_forces_detour_through_door():
    world = make_world(
        [
            ".....",
            "####.",
            ".....",
        ]
    )
    planner = PathPlanner(world)
    # East along row 0, down through the gap at (4, 1), west to (1, 2), which is next to (0, 2).
    assert planner.steps((0, 0), (0, 2)) == 9


def test_unreachable_returns_none():
    world = make_world(
        [
            "..#..",
            "..#..",
            "..#..",
        ]
    )
    planner = PathPlanner(world)
    assert planner.steps((0, 0), (4, 1)) is None
    assert planner.next_cell((0, 0), (4, 1)) is None


def test_furniture_is_impassable_but_reachable():
    world = make_world(["...", "...", "..."])
    add_furniture(world, "table_1", (1, 1), surface=True)
    planner = PathPlanner(world)
    # The table blocks the direct route, so walk around it to (1, 0), which is next to (2, 1).
    assert planner.steps((0, 1), (2, 1)) == 2
    assert planner.steps((0, 1), (1, 1)) == 0  # standing next to the table reaches it
    assert planner.steps((1, 1), (2, 1)) is None  # the furniture cell itself is not free


def test_next_cell_decreases_distance():
    world = make_world(["......", "......", "......"])
    planner = PathPlanner(world)
    cell = (0, 0)
    target = (5, 2)
    while planner.steps(cell, target) > 0:
        step = planner.next_cell(cell, target)
        assert planner.steps(step, target) == planner.steps(cell, target) - 1
        cell = step


def test_cached_distances_match_fresh_bfs():
    world = make_world(["......", ".##...", "......"])
    cached = PathPlanner(world)
    first = [cached.steps((x, y), (5, 0)) for x in range(6) for y in range(3)]
    second = [cached.steps((x, y), (5, 0)) for x in range(6) for y in range(3)]
    fresh = [PathPlanner(world).steps((x, y), (5, 0)) for x in range(6) for y in range(3)]
    assert first == second == fresh


def test_travel_ticks_rounds_up_and_is_at_least_one():
    assert travel_ticks(0, 1) == 1
    assert travel_ticks(4, 2) == 2
    assert travel_ticks(5, 2) == 3
    assert travel_ticks(7, 1) == 7
