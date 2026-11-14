import dataclasses
import itertools
from pathlib import Path

import yaml

from experiments.generate_mission import generate
from experiments.run_single import run_config
from sim.config import load_config
from sim.scenario import load_scenario
from sim.world import chebyshev_distance

SETTINGS = Path("configs/generator/house_48x32.yaml")
GENERATED_FILES = ("map.txt", "mission.yaml", "episode_greedy.yaml", "episode_scripted.yaml")


def settings(seed):
    # One budget probe instead of three keeps the test fast; the budget logic is the same.
    with SETTINGS.open(encoding="utf-8") as handle:
        return {**yaml.safe_load(handle), "seed": seed, "budget_probe_seeds": 1}


def test_dirty_dishes_form_separate_clusters_of_stacks_one_task_each(tmp_path):
    base = settings(0)
    for seed in (0, 1, 2):
        out_dir = generate({**base, "seed": seed}, tmp_path)
        scenario = load_scenario(out_dir / "map.txt", out_dir / "mission.yaml")
        world = scenario.world
        washes = [task for task in scenario.tasks if task.kind == "clean_dishes"]
        assert len(washes) == base["dirty_clusters"]
        clusters = []
        for task in washes:
            dishes = [goal.obj_id for goal in task.goals if goal.relation == "washed"]
            tiles = {}
            for dish in dishes:
                tiles.setdefault(world.objects[dish].cell, []).append(dish)
            low, high = base["stack_size"]
            assert all(low <= len(stack) <= high for stack in tiles.values())
            assert len(tiles) <= base["cluster_tiles"][1]
            assert len({world.objects[dish].parent_id for dish in dishes}) == 1
            # Connected: every tile reaches the others through adjacent tiles.
            cells = list(tiles)
            reached = {cells[0]}
            for _ in cells:
                reached |= {
                    cell
                    for cell in cells
                    if any(chebyshev_distance(cell, other) <= 1 for other in reached)
                }
            assert reached == set(cells)
            clusters.append(cells)
        for first, second in itertools.combinations(clusters, 2):
            assert all(chebyshev_distance(a, b) > 1 for a in first for b in second)


def test_same_seed_gives_byte_identical_files(tmp_path):
    out_dir = generate(settings(1), tmp_path)
    first = {name: (out_dir / name).read_bytes() for name in GENERATED_FILES}
    generate(settings(1), tmp_path)  # overwrites the same folder
    for name in GENERATED_FILES:
        assert (out_dir / name).read_bytes() == first[name], name


def test_generated_mission_is_valid_and_greedy_finishes_within_budget(tmp_path):
    out_dir = generate(settings(2), tmp_path)
    config = load_config(out_dir / "episode_greedy.yaml")
    assert config.task_visibility == "discovered" and config.spawn.mode == "uniform"
    scenario = load_scenario(config.map_path, config.mission_path)  # validates spawns
    rooms = set(scenario.world.rooms.values())
    assert len(rooms) >= 7  # 6-8 rooms plus outside
    assert len(scenario.tasks) >= 20
    dishes = [
        obj for obj in scenario.world.objects.values() if obj.obj_type in ("plate", "cup", "bowl")
    ]
    cabinets = [obj for obj in scenario.world.objects.values() if obj.obj_type == "cabinet"]
    assert sum(len(cab.cells()) * cab.slot_capacity for cab in cabinets) >= len(dishes)
    no_drops = dataclasses.replace(
        config,
        output_dir=tmp_path / "runs",
        faults=dataclasses.replace(config.faults, drop_prob=0.0),
    )
    _, final_tick = run_config(no_drops)
    assert final_tick < config.max_ticks - 1
