import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from experiments.analyze import list_run_dirs  # noqa: E402
from experiments.dev_refresh import refresh  # noqa: E402
from experiments.run_single import run_config, run_dir_of  # noqa: E402
from experiments.sweep import build_cells, episode_label, run_episodes  # noqa: E402
from sim.config import load_config  # noqa: E402

SMOKE = Path("configs/m1_smoke.yaml")
HOME_SPEC = {
    "env": "m2_home",
    "episodes": ["configs/m2_home_oracle.yaml", "configs/m2_home_greedy.yaml"],
    "first_seed": 0,
    "seeds": 1,
    "loss_levels": [0.3],
    "range_levels": [8],
    "churn": {"num_agents": 1, "earliest_tick": 40, "latest_tick": 40, "rejoin_delay": 60},
}
SMOKE_SPEC = {**HOME_SPEC, "env": "m1_smoke", "episodes": [SMOKE.as_posix()]}


def test_cells_sweep_one_factor_at_a_time_and_never_the_oracle(tmp_path):
    episodes = build_cells(HOME_SPEC, tmp_path)
    cells = {config.name: config for config in episodes}
    assert set(cells) == {"oracle", "greedy", "greedy_p0.3", "greedy_r8", "greedy_churn1"}
    assert cells["greedy_p0.3"].channel.base_loss == 0.3
    assert cells["greedy_p0.3"].channel.comm_range is None
    assert cells["greedy_r8"].channel.comm_range == 8
    assert cells["greedy_churn1"].churn.num_agents == 1


def test_env_placeholders_expand_and_seeds_count_from_the_first(tmp_path):
    spec = {
        **HOME_SPEC,
        "episodes": ["configs/generated/{env}/episode_greedy.yaml"],
        "envs": ["house_48x32_s0", "house_48x32_s1"],
        "first_seed": 5,
        "seeds": 2,
        "loss_levels": [],
        "range_levels": [],
        "churn": None,
    }
    episodes = build_cells(spec, tmp_path)
    assert [(config.env, config.name, config.seed) for config in episodes] == [
        ("house_48x32_s0", "greedy", 5),
        ("house_48x32_s1", "greedy", 5),
        ("house_48x32_s0", "greedy", 6),
        ("house_48x32_s1", "greedy", 6),
    ]


def test_a_sweep_runs_every_condition_at_a_seed_before_the_next_seed(tmp_path):
    spec = {
        **HOME_SPEC,
        "episodes": [
            "configs/generated/{env}/episode_greedy.yaml",
            "configs/generated/{env}/episode_scripted.yaml",
        ],
        "envs": ["house_48x32_s0"],
        "first_seed": 0,
        "seeds": 2,
        "loss_levels": [],
        "range_levels": [],
        "churn": None,
    }
    order = [(config.name, config.seed) for config in build_cells(spec, tmp_path)]
    assert order == [("greedy", 0), ("scripted", 0), ("greedy", 1), ("scripted", 1)]


CLUSTER_SPEC = {
    **HOME_SPEC,
    "episodes": [
        "configs/generated/{env}/episode_greedy_llm.yaml",
        "configs/generated/{env}/episode_llm_cbba.yaml",
    ],
    "envs": ["house_48x32_s0", "house_48x32_s1"],
    "first_seed": 0,
    "seeds": 2,
    "loss_levels": [],
    "range_levels": [],
    "churn": None,
}


def test_a_spec_llm_block_replaces_the_model_path_of_every_episode(tmp_path):
    spec = {**CLUSTER_SPEC, "llm": {"model_path": "/models/Qwen3-8B-Q4_K_M.gguf"}}
    episodes = build_cells(spec, tmp_path)
    assert {config.llm.model_path for config in episodes} == {"/models/Qwen3-8B-Q4_K_M.gguf"}
    untouched = build_cells(CLUSTER_SPEC, tmp_path)[0].llm
    assert episodes[0].llm == dataclasses.replace(untouched, model_path=episodes[0].llm.model_path)


def test_only_keeps_the_listed_episodes(tmp_path):
    only = ["house_48x32_s1/greedy_llm/seed_0", "house_48x32_s0/llm_cbba/seed_1"]
    episodes = build_cells({**CLUSTER_SPEC, "only": only}, tmp_path)
    assert sorted(episode_label(config) for config in episodes) == sorted(only)


def test_shards_split_the_episodes_exactly_once(tmp_path):
    episodes = [episode_label(config) for config in build_cells(CLUSTER_SPEC, tmp_path)]
    shards = [episodes[shard::3] for shard in range(3)]
    assert sorted(label for shard in shards for label in shard) == sorted(episodes)
    # Seed-major order deals each shard a mix of conditions, not one condition each.
    assert all(len({label.split("/")[1] for label in shard}) == 2 for shard in shards)


def test_greedy_with_the_llm_executor_gets_its_own_cells(tmp_path):
    spec = {
        **HOME_SPEC,
        "episodes": ["configs/m2_home_greedy.yaml", "configs/m2_home_greedy_llm.yaml"],
        "churn": None,
    }
    names = {config.name for config in build_cells(spec, tmp_path)}
    assert names == {"greedy", "greedy_p0.3", "greedy_r8"} | {
        "greedy_llm",
        "greedy_llm_p0.3",
        "greedy_llm_r8",
    }


def test_a_failing_episode_is_reported_and_the_rest_still_run(tmp_path):
    good = dataclasses.replace(load_config(SMOKE), output_dir=tmp_path, max_ticks=20)
    broken = dataclasses.replace(good, name="broken", condition="no_such_condition")
    run_dirs, failed = run_episodes([good, broken], 2, "exp_a", "20261108-120000")
    assert len(run_dirs) == 1 and failed == ["m1_smoke/broken/seed_0"]


def test_refresh_replaces_only_its_own_environment_and_keeps_fixed_names(tmp_path):
    stray = tmp_path / "m1_smoke" / "figures" / "old_figure.png"
    stray.parent.mkdir(parents=True)
    stray.write_text("stale")
    neighbour = tmp_path / "m2_home" / "figures" / "keep.png"
    neighbour.parent.mkdir(parents=True)
    neighbour.write_text("keep")

    figures = refresh(SMOKE_SPEC, tmp_path)

    assert figures == tmp_path / "m1_smoke" / "figures"
    assert not stray.exists() and neighbour.exists()
    for name in (
        "fig_cost.png",
        "episode_progress_seed0.png",
        "episode_progress_seed0.csv",
        "episode_map_scripted_seed0.png",
    ):
        assert (figures / name).exists(), name
    assert (tmp_path / "m1_smoke" / "logs" / "scripted_r8" / "seed_0" / "run.json").exists()


def test_a_named_experiment_lands_in_its_own_stamped_folder(tmp_path):
    config = dataclasses.replace(load_config(SMOKE), output_dir=tmp_path, max_ticks=20)
    assert run_dir_of(config) == tmp_path / config.name / "seed_0"
    expected = tmp_path / "exp_a" / "m1_smoke" / config.name / "20261107-120000" / "seed_0"
    assert run_dir_of(config, "exp_a", "20261107-120000") == expected
    run_dir, _ = run_config(config, "exp_a", "20261107-120000")
    assert run_dir == expected
    record = json.loads((run_dir / "run.json").read_text())
    assert (record["experiment"], record["stamp"]) == ("exp_a", "20261107-120000")


def test_a_stamp_selects_one_launch_of_an_experiment(tmp_path):
    config = dataclasses.replace(load_config(SMOKE), output_dir=tmp_path, max_ticks=20)
    first, _ = run_config(config, "exp_a", "20261107-120000")
    second, _ = run_config(config, "exp_a", "20261107-130000")
    root = tmp_path / "exp_a"
    assert list_run_dirs(root) == [first, second]
    assert list_run_dirs(root, "20261107-130000") == [second]
    with pytest.raises(ValueError, match="no-such-stamp"):
        list_run_dirs(root, "no-such-stamp")
