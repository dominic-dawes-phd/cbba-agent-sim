from sim.perception import (
    Viewpoint,
    cell_visible,
    observed_relations,
    visible_object_ids,
    walls_between,
)
from tests.helpers import add_furniture, add_item, make_world


def view(cell, heading=0.0, view_range=10.0, view_angle=360.0):
    return Viewpoint(cell, heading, view_range, view_angle)


def test_walls_block_sight_and_are_counted():
    world = make_world(
        [
            ".....",
            "..#..",
            ".....",
        ]
    )
    assert walls_between(world.walls, (0, 1), (4, 1)) == 1
    assert not cell_visible(world.walls, view((0, 1)), (4, 1))
    assert cell_visible(world.walls, view((0, 0)), (4, 0))


def test_view_cone_and_range():
    world = make_world(["........."] * 3)
    facing_east = view((4, 1), heading=0.0, view_range=3.0, view_angle=90.0)
    assert cell_visible(world.walls, facing_east, (6, 1))
    assert not cell_visible(world.walls, facing_east, (2, 1))  # behind
    assert not cell_visible(world.walls, facing_east, (8, 1))  # out of range
    assert cell_visible(world.walls, facing_east, (4, 1))  # own cell


def test_closed_container_hides_contents():
    world = make_world(["....."])
    add_furniture(world, "cabinet_1", (3, 0), container=True, is_open=False)
    add_item(world, "cup_1", (0, 0), parent_id="cabinet_1")
    assert visible_object_ids(world, view((0, 0))) == {"cabinet_1"}
    world.objects["cabinet_1"].is_open = True
    assert visible_object_ids(world, view((0, 0))) == {"cabinet_1", "cup_1"}


def test_relations_are_sorted_and_deterministic():
    world = make_world(["....."])
    add_furniture(world, "table_1", (2, 0), surface=True)
    add_item(world, "plate_1", (0, 0), parent_id="table_1")
    add_item(world, "mug_1", (3, 0))
    add_item(world, "pot_1", (0, 0))
    world.hold("pot_1", "arm_a", (0, 0))
    visible = visible_object_ids(world, view((0, 0)))
    relations = observed_relations(world, visible)
    assert relations == observed_relations(world, set(sorted(visible, reverse=True)))
    assert relations == [
        ("mug_1", "in_room", "room"),
        ("plate_1", "on", "table_1"),
        ("pot_1", "held_by", "arm_a"),
        ("table_1", "in_room", "room"),
        ("mug_1", "next_to", "plate_1"),
        ("mug_1", "next_to", "table_1"),
    ]
