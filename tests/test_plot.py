import pytest

pytest.importorskip("matplotlib")

from experiments import plot  # noqa: E402

FIGURES = ("fig_h3_loss", "fig_h3_range", "fig_churn", "fig_cost")


def run(condition, seed, **overrides):
    metrics = {
        "condition": condition,
        "seed": seed,
        "base_loss": 0.0,
        "comm_range": None,
        "mission": "mission_a.yaml",
        "crash_ticks": [],
        "within_budget": True,
        "duplicate_work_ticks": 10.0,
        "makespan": 300,
        "tokens": 70000,
        "outstanding": [[0, 5], [100, 3], [250, 0]],
    }
    return {**metrics, **overrides}


def synthetic_runs():
    runs = []
    for condition, slope in (("greedy", 1.0), ("llm_cbba", 0.2)):
        for seed in range(3):
            for loss in (0.0, 0.3, 0.6):
                runs.append(
                    run(
                        condition,
                        seed,
                        base_loss=loss,
                        within_budget=loss < 0.5 * slope + 0.1,
                        duplicate_work_ticks=100 * loss * slope,
                    )  # fmt: skip
                )
            for comm_range in (4.0, 8.0, 12.0):
                runs.append(
                    run(condition, seed, comm_range=comm_range, within_budget=comm_range > 6)
                )
            runs.append(run(condition, seed, crash_ticks=[60]))
    return runs


def test_all_hero_figures_render_with_their_table_twins(tmp_path):
    plot.apply_style()
    runs = synthetic_runs()
    for name in FIGURES:
        assert getattr(plot, name)(runs, tmp_path) == "written", name
        for extension in ("png", "pdf", "csv"):
            assert (tmp_path / f"{name}.{extension}").stat().st_size > 0


def test_each_figure_holds_the_other_sweeps_at_baseline():
    runs = synthetic_runs()
    loss_runs = plot.baseline(runs, "base_loss")
    assert all(run["comm_range"] is None and not run["crash_ticks"] for run in loss_runs)
    assert {run["base_loss"] for run in loss_runs} == {0.0, 0.3, 0.6}


def test_condition_colors_are_fixed_and_ablations_are_dashed():
    assert plot.style_of("greedy") == (plot.SLOT_COLORS[5], "-")
    assert plot.style_of("llm_cbba-noledger") == (plot.SLOT_COLORS[3], "--")
    assert plot.style_of("llm_only+ledger") == (plot.SLOT_COLORS[1], "--")
    assert plot.style_of("scripted")[0] == plot.SCAFFOLD_COLOR


def test_crossovers_are_found_between_shared_levels():
    first = [(4.0, 0.2, 0, 0), (8.0, 0.8, 0, 0)]
    second = [(4.0, 0.6, 0, 0), (8.0, 0.4, 0, 0)]
    assert plot.crossovers(first, second) == [6.0]
