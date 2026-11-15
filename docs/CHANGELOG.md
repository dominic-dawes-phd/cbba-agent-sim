# Changelog

What changed in the simulator and experiment tooling, newest first. Design rationale lives in `design/decisions.md`; the experimental design lives in `DOE.md`.

**Versions are `MAJOR.MINOR.PATCH`**, written with a two-digit patch (`1.0.01`). `pyproject.toml` holds the same number in PEP 440 form (`1.0.1`), and every `run.json` records it as `sim_version`, so a result can be traced to an entry here.

- **MAJOR:** breaks comparability with earlier results, such as tick semantics, metric definitions, or the paper's condition set. Earlier runs must be re-run before they can be compared.
- **MINOR:** a new capability, condition, sweep factor, or mission family. Earlier results stay comparable.
- **PATCH:** fixes, refactors, docs, and tooling.

Each entry is 3-6 bullets and names its date.

## [5.4.01] PATCH (Nov 14, 2026)
- **`README.md`:** a hands-on guide for running without an AI assistant: directory map, venv setup, first coded runs, LLM runs on the laptop (llama.cpp install, weights, preflight, server), container and cluster sequence, a command cheat sheet, and troubleshooting for the failures seen so far.
- **`scripts/`:** `check.sh` (tests + lint), `start_server.sh` (laptop llama-server with the `llm_setup.md` flags; checks the binary and model first), `preflight.ps1` (free RAM, shared GPU memory, zombie processes, `docker.exe` polling; PASS/FAIL with fixes).
- **`fetch_model.sh` moved** from `cluster/` to `scripts/` and takes a target folder, so the laptop and the cluster share it; `cluster/setup.sh` and `cluster.md` follow.
- **Measured:** the server at `--ctx-size 32768` uses 7.2 of the RTX 4070's 8.2 GB (`llm_setup.md`). No result changes.

## [5.4.00] MINOR (Nov 14, 2026)
- **Cluster runs** (`docs/design/cluster.md`): an Apptainer-ready image (`Dockerfile`: llama.cpp's official CUDA server at b12211 plus Python 3.12 and PyYAML), built to GHCR by `.github/workflows/container.yml`. The code is a git checkout mounted at `/work`, so code fixes need no rebuild.
- **`cluster/` scripts:** `cluster.env` (the only file to edit), `setup.sh` and `fetch_model.sh` (image and model, sha256-checked), `submit.sh` (a Slurm array under one stamp), `sweep.sbatch` and `run_shard.sh` (one GPU, one server on its own port, one shard), `pack_results.sh` (a tarball for the laptop).
- **Sweep runner** (`experiments/sweep.py`): `--shard`, `--of`, `--stamp`, `--endpoint`; spec keys `llm:` (replaces fields of every episode's `llm` block) and `only:` (rerun named episodes); `episode_label()`. Tests in `tests/test_dev_refresh.py`.
- **Specs:** `configs/sweeps/cluster_pilot_llm.yaml` (the house pilot rerun on the cluster) and `cluster_smoke_llm.yaml` (one kitchen C3 episode).
- **Comparability rule** (`design/decisions.md`): a comparison never mixes laptop and cluster runs. Results of earlier runs are unchanged.

## [5.3.02] PATCH (Nov 14, 2026)
- **House LLM pilot** (Qwen3-8B, 24 episodes, `runs/pilot_llm/`): 21 ran, 3 stopped when the server's shared context overflowed. Share of tasks done: C3 0.84, G-llm 0.80, C2 0.79, C1 0.46; recorded in `DOE.md` §7.
- **Server context raised to 32,768 tokens** (`design/llm_setup.md`): house prompts reach 4.4k tokens, and six slots each keep their last prompt cached, so 16,384 overflowed (107 HTTP 500 answers).
- **`LLMServerError`** (`llm/client.py`): after its last retry the client raises a plain error with the server's message. The `HTTPError` it raised before holds an open response, so the sweep could not pickle it and logged a `TypeError` instead. Test in `tests/test_llm.py`.
- **Memory leak found:** about 279,000 zombie processes (90 KB each) filled RAM during the pilot, mostly `docker.exe` polls from VS Code's Docker extensions. `llm_setup.md` adds a zombie-process preflight check, and `.vscode/settings.json` keeps `runs/` out of VS Code's file watcher.

## [5.3.01] PATCH (Nov 12, 2026)
- **Sweeps run seed by seed** (`experiments/sweep.py` `build_cells`): episodes are ordered by seed, then environment, keeping the spec's condition order within each. A long local sweep that stops early has every cell at the same seeds, not some conditions finished and others never started.
- **Test:** `tests/test_dev_refresh.py` checks the new order. Results are unchanged; only the run order differs.
- **Preflight before the house LLM pilot:** free RAM had fallen from 19.5 GB (Nov 9 restart) to 7.9 GB, with about 7 GB in use outside any process, so the pilot waits for a Windows Restart (`design/llm_setup.md` memory checks).

## [5.3.00] MINOR (Nov 10, 2026)
- **`rho` removed.** The generator no longer writes language-only notes, hidden attributes, or a `rho` key; `language_only` and `note` are gone from tasks (`sim/scenario.py`, `sim/loop.py`), `rho` from `metrics.json`, and `fig_h1_rho` from `experiments/plot.py`. The three houses were regenerated: maps, episode configs, and tick budgets (2100 / 2450 / 3500) are byte-identical, and only the mission files lost the `rho` content. The `scan` and `hidden_attrs` mechanism stays. Coded results are unchanged. LLM prompts change (task text lost the notes), so earlier LLM runs do not replay.
- **Hypotheses revised** to structure (H1, scale), reasoning (H2, C3 vs C2), and resilience (H3, loss, range, churn), with no threshold hypothesis. `CLAUDE.md`, `DOE.md`, `decisions.md`, `conditions.md`, `outline.md`, and `main.tex` follow. `fig_h2_loss` is now `fig_h3_loss`.
- **Mission-scale sweep proposed, not built:** levels for tasks, team size, and difficulty in `decisions.md` (Nov 10) and `DOE.md`; battery and richer utility inputs recorded as future work.
- **Paper:** `main.tex` takes the revised abstract and Introduction, and its Approach and Experiments sections are drafted from the code, with `metrics.json` field names as the metric names.

## [5.2.00] MINOR (Nov 10, 2026)
- **Reply schema fixes each tool's arity** (`llm/prompts.py` `tool_call_schema`): `anyOf` over one object per tool, with exactly its number of arguments and whole ticks for `wait`. A refused call quotes the tool's usage line. Qwen3.5-4B's refused replies fell from 159 to 0. This changes the prompts' cache keys, so Nov 9 LLM runs do not replay.
- **`llm.thinking`** (default false, `sim/config.py`): lets a reasoning model think, via `enable_thinking`. The probe configs set it with 2,048 max tokens.
- **Model bake-off** after the Windows restart, with no crashes in ten live episodes. C3 in the kitchen: Ministral 3 8B 8/11 tasks, Qwen3-8B 7/10, Qwen3.5-4B 3/6, and 1/6 with thinking (27 ticks per decision). Details in `design/llm_models.md`.
- **Docs:** four more candidates (Granite 4.2 3B, DeepSeek-R1-0528-Qwen3-8B, LFM2.5-1.2B-Thinking, Qwen3-4B-Thinking-2507) with a reasoning column; `llm_setup.md` memory checks.

## [5.1.01] PATCH (Nov 9, 2026)
- **Model candidates** (`docs/design/llm_models.md`): 17 lightweight open models with tool calling, with GGUF sizes and licenses (checked or reported). Qwen3.5-4B and Ministral 3 8B were downloaded and hashed, and bake-off configs `configs/m2_home_{llm_cbba,greedy_llm}_{qwen35_4b,ministral3_8b}.yaml` were added.
- **Bake-off blocked:** every llama-server crashed with CUDA "out of memory" while VRAM was free. The laptop had 3.6 GB of RAM free, with `dwm.exe` holding 5 GB of shared GPU memory. Rerun after a restart. The server now runs with `--cache-ram 0`, and `llm_setup.md` has pre-run memory checks and GPU monitoring commands.
- **Analyzer rates for the paper:** `false_completion_rate`, `replan_rate`, `consensus_messages_per_task`. `false_completions()` now also returns the number of claims.
- **Docs:** a metric reference table in `DOE.md` §3, linked from `paper/outline.md`.

## [5.1.00] MINOR (Nov 8, 2026)
- **LLM conditions built.** `LLMPolicy` wraps an allocator around the LLM executor:
  - **G** with `executor: llm`;
  - **C2** (`numeric_cbba`) and **C3** (`llm_cbba`), with the replicated ledger (`sim/ledger.py`) and CBBA (`allocation/cbba.py`);
  - **C1** (`llm_only`), with RoCo-style dialogue rounds (`allocation/dialogue.py`).

  Coded results are unchanged (kitchen regression identical).
- **LLM plumbing:**
  - **Client:** an OpenAI-compatible client (stdlib only) with JSON-schema replies, a response cache with a replay-only mode, and the model hash in `run.json`.
  - **Prompts:** pinned in `configs/prompts/`.
  - **Loop:** a per-tick `on_tick` coordination hook, concurrent decisions applied in agent order, and several logged calls per decision (`purpose`: act, utility, dialogue).
  - **Server:** llama.cpp b12211 with Qwen3-8B Q4_K_M, using a unified 16k KV cache after two out-of-memory crashes (`design/llm_setup.md`).
- **New streams and responses:**
  - **Streams:** `ledgers` (every change to every copy) and `dialogues`.
  - **Analyzer:** false completion claims, rejected responses and parse failures, C1 plan rejections, final ledger lag, LLM calls by purpose, and messages by kind.
- **Tooling:**
  - The generator writes `episode_{greedy_llm,numeric_cbba,llm_cbba,llm_only}.yaml` per house.
  - Sweeps name LLM-executor G cells `greedy_llm`.
  - `configs/sweeps/pilot_llm.yaml` is the house pilot.
- **Live kitchen pilots** (Qwen3-8B, one seed each, 900 ticks; not results):
  - **Tasks done:** C3 7/8, G-llm 5/8, C2 4/7, C1 4/10.
  - **Coordination:** C2 and C3 duplicated no work and their ledgers converged; C1 duplicated 106 ticks.
  - **Fixes they prompted:** code unlocks workspaces the current task does not lock; voided tasks are flagged; task tools take only the agent's own task; G retakes a released task after 120 ticks; a `search` that has swept its round ends; the server caps its host prompt cache. Coded regressions are unchanged (houses within 2 ticks).

## [5.0.00] MAJOR (Nov 8, 2026)
- **Washed dishes are unloaded as a found task.** An agent that sees clean dishes inside an open dishwasher announces `unload_<dishwasher>_<dish>` and carries on with its own load. The unload keeps its whole-task lock, so a loader waits for it.
- **Waiting at a busy dishwasher.** The coded executor waits out a cycle that ends sooner than the cheapest way out, or when it has nothing else to do. The latter saves about 3% in the kitchen; the former is neutral.
- **C0 plans unloads.** Between consecutive washes on a shared machine, an unload gates the next wash's first hold (`PlanProblem.gates`, in the heuristic and CP-SAT). Every order of those washes is solved. Results: 15/15 proven optimal, with no unplanned unload in any replay.
- **Regressions:** G 119/120 (1001 / 1460 / 1604; one claim-race gridlock), scripted 120/120, kitchen 240/240. Budgets are now 2100 / 2450 / 3500.
- **Docs:** DOE gains perception settings (range, field of view, line of sight, what is hidden). The old v3.2.00 pilot (15 GB) is deleted. **Not comparable with v4.0.00:** missions now include unload work.

## [4.0.00] MAJOR (Nov 8, 2026)
- **Shared dishwashers.** Wash goals are per dish (`washed`, in any dishwasher). A cycle started mid-load splits the task: loaded dishes finish with it, and the rest become a new task, announced as found, with a held dish carried back by a no-LLM `return_item` reflex. The cycle countdown is perceivable.
- **Pick cancels and executor fixes.** A queued pick stops once its dish is seen elsewhere, and the pick re-checks ground truth. The coded executor no longer:
  - takes over a teammate's fixed assignment;
  - targets a dish a teammate holds;
  - re-opens an open door;
  - idles at a busy dishwasher (it moves to another, or puts the dish back and gives the task up).

  C0's model holds a dishwasher for a whole wash, and its replay keeps plan order per machine.
- **Found while verifying:** agents now know their own hands (shards in a closed can left "held" beliefs); a floor cell next to unseen furniture stays a search frontier (a shelf beyond view range was never searched again); four executor races.
- **Compact logs:** gzipped streams, belief changes carrying the new belief, a per-tick `poses` stream, and state keyframes every 50 ticks. A house run is 0.3-0.6 MB, down from 7-17. The coded pilot rerun (720 episodes, no failures) wrote 360 MB instead of 15 GB; G takes 1.99 / 2.39 / 1.64x C0's replay at baseline.
- **Results:** C0 15/15 proven optimal with no replay splits. G 120/120 at 877 / 1147 / 1227, scripted 120/120 at 1120 / 1160 / 1435, kitchen 240/240. **Not comparable with earlier versions:** wash goals, exploration, executor behavior, and the log format all changed.

## [3.2.00] MINOR (Nov 8, 2026)
- **Sweep runner:** `python -m experiments.sweep configs/sweeps/<experiment>.yaml` runs every cell (baseline, then each loss, range, and churn level; C0 at baseline only) at every seed in parallel under one stamp, copies the spec next to the runs, and prints a per-(environment, cell) summary. A failed episode is logged and skipped, and the exit code reports it.
- **Coded pilot run** (`configs/sweeps/pilot_coded.yaml`, C0 and coded G, 3 houses, 20 seeds, 720 episodes, 23 minutes): no failures. Its logs took 15 GB.
- **Sweeps and dev refreshes keep Windows awake.** A refresh had stretched to 32 minutes because Modern Standby froze it (Kernel-Power 506/507 in the System log).
- **Shared code:** `dev_refresh` builds its cells with the sweep's `build_cells`. Dev specs now use `first_seed` and `seeds` (a count). `analyze` groups by environment and cell and exposes `write_metrics` and `print_summary`.
- **Docs:** `DOE.md` gains a Key Definitions/Concepts section. The scripted s1 seed 24 failure is traced to a duplicated wash task in the coded executor (`decisions.md`, Nov 8; not fixed yet).

## [3.1.00] MINOR (Nov 7, 2026)
- **Dirty dishes come in clusters.** Generated houses place 3 clusters of 1-3 adjacent surface tiles, each a stack of 1-5 dishes. One cluster is one wash task, loaded and then washed in one cycle. This replaces `dish_batch_size`, so **house results are not comparable with earlier ones**; budgets are now 1900/2150/3050.
- **Reflex refinements.** `nearest_trashcan` reads ground truth, a deliberate cheat so a fragile drop can never become a search, and `return_to_task` walks back to the drop tile before any decision.
- **C0 measured (5 seeds per house):** 15/15 proven optimal, mean planned / replay 400/448, 453/489, 688/747, and G at drop 0 takes 1.74 / 1.87 / 1.58x the replay. One long task now dominates the makespan: the longest wash alone is 50-67% of it (`DOE.md`, open question 0a).
- **Regression (drop 0.025):** G 120/120 (849 / 1059 / 1218), scripted 119/120 (752 / 1039 / 1086); kitchen 240/240 (scripted 298 / 294 / 248, G 303 / 303 / 264 at drop 0 / 0.025 / 0.2).
- **Dev figures and run layout.** `experiments/dev_refresh.py` and `plot_episode.py` write `runs/dev/<env>/figures/` with fixed names from `configs/dev/*.yaml`. `run_single --run-name <exp>` keeps runs in `runs/<exp>/<env>/<cell>/<stamp>/seed_<n>/`. New config field `env`; `analyze` and `plot` take `--stamp`.
- **Docs:** C0's scope is recorded (no search cost, no `rho`, no posted-task variants), and the CP-SAT citations are in `references.bib`.

## [3.0.00] MAJOR (Nov 7, 2026)
- **Drops are now a time penalty.** Non-fragile: a 2-tick pause, item kept, no failure and no decision. Fragile: a `clean_up` reflex in every condition with no LLM call (3 ticks, walk to nearest trashcan, 1 tick), then back to the same task.
- **Only the broken item is lost;** the task stays with its agent and is voided only if no item remains. The discovered `cleanup` task and template are removed. A crash leaves shards as debris with no task.
- **Result:** makespan is flat at the paper's drop rate (houses, G: 672 vs 674, 877 vs 866, 737 vs 737). It still falls at drop 0.2 in the kitchen, where a break voids a single-plate task and skips its wash.
- **Analyzer:** new `drops` and `items_broken`; events log `dropped` and `broken`.
- **Regression (drop 0.025):** G 119/120, scripted 120/120, kitchen 240/240. Comparability breaks for every earlier run with drops.

## [2.1.00] MINOR (Nov 6, 2026)
- **C0 oracle built** (`condition: oracle`): a seeded heuristic seeds OR-Tools CP-SAT (optional `[solver]` extra), planning who does what and when, replayed in the sim at `L = 0` with ground-truth beliefs. Each run logs bound, planned makespan, and release ticks to `oracle_plan.json`.
- **Measured (5 seeds per house):** proven optimal on s1; gaps of 25-31% on s0 and 36-39% on s2. Replays land 6-11% above plan, and G takes 1.4-1.6x the replay at drop 0.
- **Generator bug fixed:** `_nearest` always returned the first option, sending every wash and pack to one appliance since Oct 28. Budgets are now 4x the oracle replay (1950/1950/2100), not 2x greedy.
- **Two movement livelocks fixed for all conditions:** persistent detours and seeded randomized backoff.
- **Analyzer and figures:** `makespan_vs_oracle`, `oracle_bound`, `oracle_planned`, `solver_gap`; `fig_h1_rho` uses the oracle as its denominator.
- **Docs:** `conditions.md` now says what a found task becomes in each condition; corrected the explanation of the packet-loss test (delivered messages cost re-decisions).

## [2.0.00] MAJOR (Nov 4, 2026)
- **Discovery-only tasks** (`task_visibility: discovered`): agents start knowing only the floor plan, and a mission task is revealed when an agent sees its reveal objects, even mid-task. It is logged `exists`/`discovered` and announced by `task_found`, with one id per task and no duplicates. Hand-written kitchens stay `posted`.
- **Exploration:** nearest-frontier `explore` (team coverage) and `search` (own coverage, restarting rounds because objects move). Coverage is shared every 10 ticks as `coverage` messages that never prompt a decision, with an optional `search_ttl` decay. New module `sim/exploration.py`.
- **Seeded uniform spawns** (`spawn.mode: uniform`, `min_separation`). Houses are regenerated with budgets of 2x the slowest greedy run over 3 spawns: 1750, 2400, and 1850 ticks.
- **Battery** config parsed but must stay disabled. Battery, scheduled disconnect, and continuous arrivals are designed in `simulator.md`.
- **Coded-executor fixes:**
  - stale re-broadcast reports skipped steps;
  - teammates' loaded items were taken back out;
  - held items were abandoned on a teammate's report.

  G now finishes 120/120 house runs at drop 0.025.
- **Analyzer:** discovery latency, undiscovered count, exploration ticks, coverage over time; coverage messages counted apart. The RoCo bib entry gains its DOI.

## [1.0.02] PATCH (Nov 2, 2026)
- New `docs/design/conditions.md`, the canonical condition spec:
  - a layer model: fixed perception, S1 skills, and recipes; per condition, a choice of executor, allocator, utility source, and ledger;
  - one card per condition, with allowed LLM tools and no-bleed rules;
  - a comparison map.
- C1 is now RoCo-style with no ledger: lockstep dialogue rounds per communication component, with `k` partitions giving `k` transcripts. The H2 ablations are C1+ledger and C3-noledger.
- C3/C4 LLMs supply `u_ij` and status reports only; auction code bids. G and R use the LLM executor in paper runs. `scripted` is dev only.
- Docs updated: CLAUDE.md, `decisions.md`, `simulator.md` (planned `dialogues.jsonl`), DOE.md (budget 513, two new planned responses), the outline, a sentence in `main.tex`, and the RoCo notes row (read in full).
- `plot.style_of` dashes both `-noledger` and `+ledger` ablations.

## [1.0.01] PATCH (Oct 28, 2026)
- Added `docs/CHANGELOG.md` (this file) and `docs/DOE.md` (factors, levels, responses, analysis plan, executed runs).
- `CLAUDE.md`: after every implementation pass, add a changelog entry and bump the version; update `DOE.md` whenever the experimental design changes.
- `run.json` records `sim_version`, read from `pyproject.toml`. The version moved from `0.1.0` to `1.0.1`.

## [1.0.00] START (Oct 28, 2026)
Everything built before versioning began. This is the LLM-free simulator (M1, M2) with two non-LLM policies.

- **Simulated time:** tick-based and event-driven. Decisions happen only at decision points. Deliberation costs `L` ticks through the token latency model. Observations are frozen at query time, and preconditions are re-checked against ground truth when an action applies.
- **Perception and beliefs:** deterministic perception (view range, cone, line of sight, closed containers hide their contents) feeds a per-agent belief graph with `seen` / `told` / `map` sources and negative evidence. Failures are classified as hallucination, invalid call, stale belief, or skill failure.
- **Lossy channel:** base loss `p`, hard range `r`, optional distance falloff and wall penalty, and latency. Churn crashes `k` agents and rejoins them with their memory kept.
- **Pathing:** one static BFS distance function shared by bids, prompt text, and future solvers (hard rule 8). Travel time is `ceil(steps / speed)`.
- **Household world:** an ASCII map plus a YAML mission. Furniture spans several cells, each cell a slot with `slot_capacity`. The trashcan has a capacity and a fill gauge. The dishwasher runs a 60-tick wash cycle. Objects nest inside furniture slots.
- **Task templates** (`configs/task_templates.yaml`): clean_plate, clean_dishes, unpack_dishwasher, pack_groceries, take_out_trash, tidy, store_dish, cleanup. Each unrolls into primitive steps and is judged done by a ground-truth goal check (`in`, `on`, `is`, `cycle`, `in_kind`).
- **Primitives:** move_to, pick, place, open, close, scan, handoff, report, wait, start, lock, unlock, tie_bag, claim, release.
- **Workspace reservations:** `lock` reserves the reachable cells of the target plus the floor around them. A started dishwasher holds the reservation until its wash ends.
- **Faults:** `faults.drop_prob` rolls on every pick, place, and move while holding. Plates and cups shatter, which voids the tasks that needed them. Other objects are re-picked.
- **Task discovery by perception:** shards, dirty dishes, groceries, rack dishes, and full bins create tasks with deterministic ids, broadcast as `task_found`.
- **Movement:** agents move simultaneously, with a seeded 4-way stop for contested cells and no swaps. Movers replan locally around obstacles. Idle agents yield, a seeded coin flip settles head-on meetings, and patience runs out after 6 ticks without progress toward the target. Docks give idle agents a place to park.
- **Spawn validation:** start cells must be floor, distinct, and able to reach every piece of furniture.
- **Missions:**
  - hand-written: `m1_smoke` kitchen, `m2_home` (22x8 kitchen, 3 agents), `m2_rack`, and a tight-doorway map variant;
  - generated by a seeded generator: 48x32 houses with 7 rooms, about 23 tasks, 6 heterogeneous agents, and `rho`. The tick budget is 2x the greedy makespan, and the same seed gives byte-identical files.
- **Policies:** `scripted`, test scaffolding that follows the mission's fixed assignment, and `greedy` (baseline G), which takes the nearest available task and broadcasts its claim with no consensus. **No oracle (C0), auction, or LLM condition exists yet.**
- **Logging:** append-only JSONL streams (events, actions, llm_calls, messages, beliefs, state), plus `run.json` with the config, seed, and git state. Every action is tagged with its task and start tick.
- **Analyzer** (`experiments/analyze.py`): computes from the logs success, completion within budget, makespan, voided tasks, duplicated-work ticks, contested tasks, orphan delay, stale-belief and hallucination rates, blocked ticks, move conflicts, messages, tokens, and tasks outstanding over time. It reports means with 95% bootstrap CIs.
- **Headroom pilot** (`experiments/headroom.py`): runs all static assignments, or a seeded sample of them, against greedy. The best of the sampled assignments is a **weak reference, not an oracle**.
- **Hero figures** (`experiments/plot.py`): H2 against loss, H3 against range with crossovers, H1 against `rho`, churn shown as tasks outstanding over time, and cost against completion. Each is written as PNG, PDF, and a CSV twin. Optional `[analysis]` extra: matplotlib.
