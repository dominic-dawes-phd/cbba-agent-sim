from sim.beliefs import Belief, initial_beliefs, merge_told, serialize_beliefs, update_beliefs
from sim.pathing import PathPlanner
from sim.perception import Viewpoint
from tests.helpers import add_furniture, add_item, make_world


def view(cell, heading=0.0, view_angle=360.0):
    return Viewpoint(cell, heading, 10.0, view_angle)


def test_seen_object_is_added_then_marked_missing_when_moved():
    world = make_world([".....", "....."])
    add_item(world, "plate_1", (3, 0))
    beliefs = {}
    assert update_beliefs(beliefs, world, view((0, 0))) == [("plate_1", "added")]
    world.objects["plate_1"].cell = (3, 1)
    world.tick = 5
    # Facing east with a narrow cone: (3, 0) is in view, (3, 1) is not.
    changes = update_beliefs(beliefs, world, view((0, 0), heading=0.0, view_angle=10.0))
    assert changes == [("plate_1", "missing")]
    assert beliefs["plate_1"].missing
    assert beliefs["plate_1"].last_seen == 0


def test_object_out_of_view_goes_stale_but_is_kept():
    world = make_world(["......."])
    add_item(world, "mug_1", (6, 0))
    beliefs = {}
    update_beliefs(beliefs, world, view((5, 0)))
    world.tick = 20
    assert update_beliefs(beliefs, world, view((0, 0), heading=180.0, view_angle=90.0)) == []
    assert beliefs["mug_1"].last_seen == 0
    assert not beliefs["mug_1"].missing


def test_closed_container_contents_are_not_marked_missing():
    world = make_world(["....."])
    add_furniture(world, "dishwasher_1", (3, 0), container=True, is_open=True)
    add_item(world, "plate_1", (0, 0), parent_id="dishwasher_1")
    beliefs = {}
    update_beliefs(beliefs, world, view((0, 0)))
    world.objects["dishwasher_1"].is_open = False
    world.tick = 3
    changes = update_beliefs(beliefs, world, view((0, 0)))
    assert ("plate_1", "missing") not in changes
    assert not beliefs["plate_1"].missing


def test_merge_told_newer_wins():
    own = {"cup_7": Belief("cup", [], {}, (1, 1), 10, "seen")}
    newer = {"cup_7": Belief("cup", [("cup_7", "on", "shelf_2")], {}, (4, 4), 15, "seen")}
    older = {"cup_7": Belief("cup", [], {}, (9, 9), 5, "seen")}
    assert merge_told(own, older, "drone_b") == []
    assert own["cup_7"].believed_cell == (1, 1)
    assert merge_told(own, newer, "drone_b") == ["cup_7"]
    assert own["cup_7"].believed_cell == (4, 4)
    assert own["cup_7"].source == "told"
    assert own["cup_7"].told_by == "drone_b"


def test_furniture_known_from_map_and_serialized_with_steps():
    world = make_world(["......", "......"])
    add_furniture(world, "table_1", (5, 0), surface=True)
    add_item(world, "plate_1", (2, 0))
    beliefs = initial_beliefs(world)
    assert set(beliefs) == {"table_1"}
    assert beliefs["table_1"].source == "map"
    update_beliefs(beliefs, world, view((0, 0)))
    world.objects["plate_1"].cell = (2, 1)  # off the narrow sightline along row 0
    world.tick = 4
    update_beliefs(beliefs, world, view((0, 0), heading=0.0, view_angle=10.0))
    text = serialize_beliefs(beliefs, 4, (0, 0), PathPlanner(world))
    assert "  plate_1 - MISSING (not at last known spot, last seen 4 ticks ago)" in text
    assert "SEEN NOW:\n  table_1 in_room room - 4 steps" in text
