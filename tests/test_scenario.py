import pytest

from sim.agent import CapabilityCard
from sim.scenario import AgentSpec, validate_spawns
from tests.helpers import add_furniture, make_world

CARD = CapabilityCard(
    hands=1, speed=1, view_range=8.0, view_angle=360.0, can_scan=False, success={}
)


def spec(agent_id, cell):
    return AgentSpec(agent_id, CARD, cell, 0.0)


def room_with_shelf():
    # A sealed pocket at (4, 0): walls on its left and below.
    world = make_world(["...#.", "....#"])
    add_furniture(world, "shelf_1", (0, 0), surface=True)
    return world


def test_valid_spawns_pass():
    validate_spawns(room_with_shelf(), [spec("arm_a", (2, 1)), spec("arm_b", (1, 1))])


@pytest.mark.parametrize(
    ("cells", "message"),
    [
        ([(3, 0)], "a wall or off the map"),
        ([(9, 9)], "a wall or off the map"),
        ([(0, 0)], "on furniture"),
        ([(2, 1), (2, 1)], "both start at"),
        ([(4, 0)], "cannot reach shelf_1"),
    ],
)
def test_invalid_spawns_are_rejected_with_the_reason(cells, message):
    agents = [spec(f"agent_{index}", cell) for index, cell in enumerate(cells)]
    with pytest.raises(ValueError, match=message):
        validate_spawns(room_with_shelf(), agents)
