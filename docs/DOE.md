# Design of experiments

What the experiments vary, what they measure, how runs are replicated and compared, and which runs have actually been executed. Update this file whenever a factor, a level, a response, a condition, or the analysis changes, and add a line to the DOE change log at the end. Code versions refer to `CHANGELOG.md`; design rationale lives in `design/decisions.md`.

**Status (v5.4.03):** pilot stage. C0, G (coded or LLM executor), C1, C2, and C3 run; C4, L, R, and the ablations are not built. The LLM conditions have run live on Qwen3-8B only as single kitchen pilots so far. Every number below comes from pilots, not from the paper's experiment.

## Key Definitions/Concepts

**Time and runs**

| Term | Meaning |
|---|---|
| **Tick** | One step of simulated time, the unit of every reported time (hard rule 1). A speed-1 agent moves one cell per tick, a primitive takes a fixed number of ticks, and an LLM call costs `L` ticks from the latency model (tokens converted at a fixed reference throughput). Wall-clock seconds are logged as a cost only, so no result depends on hardware. |
| **Episode** (or **run**) | One simulation of one mission under one condition at one seed. It runs from tick 0 until every task is done or voided, or until the tick budget runs out. It writes one run directory of logs. |
| **Episode seed** | The integer behind every random draw in an episode: spawn cells, packet loss, drops, motion tie-breaks, crash times. Each purpose has its own stream (`<seed>:spawn`, ...). Runs of different conditions at the same seed start from the same cells and face the same draws until their actions diverge (**common random numbers**), so results are compared per (mission, seed) pair. |
| **Generator seed** | Which house the generator builds (`s0`, `s1`, `s2`). It is a different number from the episode seed: house s1 at episode seed 24 means house 1, run 24. |
| **Tick budget** (`max_ticks`) | Ticks an episode may use before it is cut off. For each house it is 4x the slowest C0 replay over 3 spawns, so no competing condition sets the yardstick. |
| **Makespan** | The tick when the last task is finished (done or voided). It exists only for runs that finished within the budget. An unfinished run is censored: it counts as a failure for completion, has no makespan, and the number censored is reported alongside. |
| **Completion** | `within_budget`: the run finished every task within the budget. `success`: every task done and none voided. `partial_success`: the share of tasks done. |

**The world and the work**

| Term | Meaning |
|---|---|
| **Mission** / **environment** (`env`) | A map plus its objects, agents, and tasks. The paper uses the generated houses `house_48x32_s0`-`s2`; the hand-written kitchens `m1_smoke` and `m2_home` are for development. `env` names the run folders. |
| **Unload** | Washed dishes stay in the dishwasher until someone unloads them. An agent that sees clean dishes inside an open dishwasher finds an unload task (`unload_<dishwasher>_<dish>`), announced like any found task; the unload holds the dishwasher, so a loader waits until it is done. |
| **Task** | One unit of work built from a template (`clean_dishes`, `tidy`, `pack_groceries`, ...), such as "load and wash this cluster of dishes". It is **done** when its ground-truth goal check holds, never on an agent's say-so. It is **voided** when every item it moves has shattered. |
| **Posted / discovered** | Posted: every agent knows every task at tick 0 (kitchens, and C0). Discovered: agents start knowing only the floor plan, and a task becomes known when an agent sees its objects or hears a teammate announce it (paper runs). |
| **Ground truth vs beliefs** | Ground truth is the world as it is. Each agent acts on its own belief graph, built by deterministic perception, which can go stale (hard rule 5). Only C0 starts from ground truth. |
| **Decision point** | A moment when an agent's executor chooses what to do next: a skill finished or failed, a message arrived, a task changed, or the episode started. LLM calls happen only here (hard rule 2). |

**Perception: what updates an agent's scene graph** (`sim/perception.py`, per-agent capability card)

| Setting | Value |
|---|---|
| **Range** | Euclidean distance from the agent's cell to the target cell at most `view_range` tiles: **8** in the generated houses and the `m2_*` kitchens, **6** in `m1_smoke`. |
| **Field of view** | A cone of `view_angle` degrees centered on the agent's heading: **360°** in the houses and `m2_*` (heading irrelevant), **180°** in `m1_smoke`. The heading is the direction of the agent's last step. |
| **Line of sight** | No wall cell on the Bresenham line between the two cells. Furniture and other agents do not block sight. |
| **What it can and cannot see** | An object is seen when any cell it occupies is in view. Contents of a closed container are hidden. Slot fill (`3/3,1/3`) shows only when the inside is visible (open container or surface). A trashcan's fill gauge and a dishwasher's cycle and countdown are readable even closed. Hidden attributes (`fragile`) need `scan`. |
| **When** | Every tick, for every online agent, before decisions, so a prompt at tick `t` shows what is in view at `t`. |
| **How it updates beliefs** | Seen objects are overwritten with fresh relations and attributes (`added` or `updated`). A believed object whose cell is in view but which is not seen is marked `missing`, unless believed inside a container now closed. Everything else goes stale. An agent always knows what it holds. |
| **Search coverage** | The cells in view are marked searched. A floor cell next to furniture not yet seen stays a frontier. |

**Conditions and comparisons**

| Term | Meaning |
|---|---|
| **Condition** | The coordination logic every agent runs: C0-C4, baselines G, L, R, ablations, or dev-only `scripted` (canonical definitions: `design/conditions.md`). |
| **C0 numbers** | **Bound**: a proven lower limit on makespan in C0's planning model. **Planned**: the makespan of its plan in that model. **Replay**: the makespan when the plan is executed in the simulator. The replay is the denominator: `makespan_vs_oracle` = a run's makespan / C0's replay at the same (mission, seed). |
| **Factors** | `p`: packet loss probability. `r`: communication range in cells. `k`: number of agents that crash and rejoin. `m`: number of tasks. `n`: team size. **Difficulty**: easy, medium, or hard, set by dish-cluster and stack size, which fix how long each wash task is and how long it holds an exclusive dishwasher (`m` stays fixed). `drop_prob`: chance of dropping a held item per pick, place, or move. |
| **Baseline point** | `p = 0`, `r = ∞`, `k = 0`, with drop 0.025, `m` = 22, `n` = 6, medium difficulty. Each sweep moves one factor away from it. |
| **Cell** | One condition at one setting of the swept factors, named `<condition>` at baseline or `<condition>_p0.3`, `_r8`, `_churn1`. A cell is run at many seeds. |
| **Experiment, launch, stamp** | An experiment is a named set of runs (`runs/<experiment>/`). Each launch of it gets a stamp (`YYYYMMDD-HHMMSS`), a wall-clock folder name only, so relaunches never overwrite earlier runs. |
| **Wasted work** | `duplicate_work_ticks`: ticks agents spent on a task that another agent finished. An **orphan** is a task left unfinished by an agent that crashed. |

## 1. What we are estimating

| Hypothesis | Estimand | Comparison | Needs |
|---|---|---|---|
| **H1** (structure) | Completion, `makespan_vs_oracle`, and wasted work (`duplicate_work_ticks`, orphans) as `m`, difficulty, and `n` grow | C1 vs C2/C3 (and G): an **interaction** of condition with scale. C1 should degrade, auction teams should not. | C0, C1, C2, C3, G; the mission-scale sweep (planned) |
| **H2** (reasoning) | Gap to the oracle (`makespan_vs_oracle`, completion) and token cost | C3 vs C2, paired per (mission, seed); they differ only in the utility source | C0, C2, C3; baseline and each scale level |
| **H3** (resilience) | Degradation slope and **normalized performance drop** (loss relative to the condition's own perfect-comm score), and recovery (`orphan_delays`), as `p` rises, `r` shrinks, and `k` grows | C3 vs C2 (both have the ledger, so the gap is the utility source), plus the 2x2 {dialogue, auction} x {ledger, none} (C1, C1+ledger, C3-noledger, C3) to split the gain between consensus and ledger | C1, C2, C3, C4, C1+ledger, C3-noledger |
| **Cost** | Tokens, messages, and LLM calls per completed task | All conditions at the baseline point | Real token counts (M4) |

Baselines G (greedy), L (elected-leader LLM), and R (random) bound these comparisons: any coordination vs consensus, leaderless vs leader, and a sanity floor.

## 2. Factors and Conditions

### 2A. Conditions

**Mission scale** (`m`, `n`, difficulty) is the main lever for H1. The proposed levels are in `design/decisions.md` (Nov 10) and the factor table below; the sweep is not built yet. (The earlier `rho` knob, language-only constraints, was removed Nov 10, 2026.)

**C0–C4** are the five main conditions. **The canonical definitions are in `design/conditions.md`**: the layer model, allowed LLM tools, no-bleed rules, and the C1 round protocol. Summary:

| ID | Name | Allocation | Ledger | Executor | LLM's role |
|---|---|---|---|---|---|
| C0 | Oracle | A central solver that knows the ground truth | n/a | coded | none (the reference to measure gaps against) |
| C1 | LLM-only | RoCo-style lockstep dialogue per comm component | none | llm | negotiates and executes |
| C2 | Numeric CBBA | CBBA auction using travel-cost utilities | yes | llm | executes only |
| C3 | LLM+CBBA | CBBA auction using LLM-generated utilities | yes | llm | supplies utilities and executes |
| C4 | LLM+MURDOCH | Single-item auctions using LLM utilities | yes | llm | supplies utilities and executes |

Baselines G (greedy), L (elected leader), and R (random) use the LLM executor in paper runs. `scripted` (fixed or round-robin assignment) is dev only.

CBBA is the consensus auction where agents bid on bundles of tasks and reach conflict-free agreement among connected agents. MURDOCH is a simpler second auction family that sells one task at a time.

The condition factor is really four layers: **allocator × utility source × ledger × executor**. Every planned comparison changes exactly one of them; see the comparison map in `conditions.md`.

Each condition runs under perfect communication and then under the sweeps (mission scale, packet loss `p`, range `r`, and churn `k`).

### 2B. Factors
"Run" lists the levels actually executed so far; "Planned" is a proposal until the LLM conditions exist.

| Factor | Config field | Planned levels | Run so far | Status |
|---|---|---|---|---|
| Condition | `condition` | C0, C1, C2, C3, C4; G, L, R; C1+ledger, C3-noledger | `scripted` (dev only, not a paper condition), `greedy` (G), `oracle` (C0); live kitchen pilots of `greedy` (LLM executor), `numeric_cbba` (C2), `llm_cbba` (C3), `llm_only` (C1) | C0, G, C1, C2, C3 built (v5.1.00); C4, L, R, ablations planned |
| Executor | `executor` (`coded \| llm`) | llm for every paper condition except C0; coded for dev references | coded; llm in pilots | built (v5.1.00). C1-C3 exist only with llm |
| LLM settings | `llm:` block (endpoint, model, `model_path`, temperature, seed, `max_tokens`, `history_keep`, `thinking`, cache) | fixed: one model Q4_K_M on llama.cpp, temperature 0, seed 0, history 8, thinking off (`design/llm_setup.md`); the model is open (`design/llm_models.md`) | Qwen3-8B, Qwen3.5-4B, Ministral 3 8B in pilots; thinking on in one probe | built (v5.2.00) |
| CBBA settings | `cbba:` block | fixed: bundle 3, discount 0.97, gossip 10, lease 60, give-up 120 ticks (not swept) | as planned | built (v5.1.00) |
| C1 dialogue settings | `dialogue:` block | fixed: `M` = 2 x members, `K` = 2, retry 30 ticks, 400-character messages (tune in the pilot, not swept) | as planned | built (v5.1.00) |
| Packet loss `p` | `channel.base_loss` | 0, 0.1, 0.3, 0.5, 0.7 | 0, 0.3, 0.6 | built |
| Comm range `r` (cells) | `channel.comm_range` | ∞, 16, 12, 8, 6, 4 (finer near `r*`) | ∞, 12, 8, 4 | built |
| Distance falloff, walls | `channel.d0`, `alpha`, `wall_penalty` | off for the main sweeps; one realism check | off | built |
| Message latency `d` | `channel.latency_ticks` | 0 (sensitivity check only) | 0 | built |
| Churn: agents crashed `k` | `churn.num_agents` | 0, 1, 2 | 0, 1 | built |
| Churn: crash timing | `churn.earliest_tick`, `latest_tick` | early / mid / late windows (open question) | tick 40 | built |
| Churn: rejoin delay | `churn.rejoin_delay` | fixed (30 or 60) | 60 | built |
| Churn: memory on rejoin | not yet | kept, wiped | kept | **planned** (decisions.md open question 7) |
| Drop rate | `faults.drop_prob` | 0.025 fixed (a nuisance factor adding time penalty and variance, not a sweep) | 0, 0.025, 0.2 | built |
| Tasks `m` | generator `tidy_items`, `dirty_clusters`, `groceries_per_entry`, `scraps_per_trashcan`, `rooms`, `width`, `height` | about 11, **22**, about 44, with house 32x24, 48x32, 64x48 and 5, 7, 10 rooms (proposed, Nov 10) | 22 | **planned**; no sweep axis yet |
| Team size `n` | generator `agents` | 3, **6**, 12 (same mix of arms and carriers; `n = 12` needs 12 server slots) | 6 | **planned** |
| Difficulty | generator `cluster_tiles`, `stack_size` | easy [1,1] and [1,2]; **medium** [1,3] and [1,5]; hard [3,4] and [3,5] (proposed, Nov 10) | medium | **planned** |
| Mission (block) | `map`, `mission` | generated 48x32 houses, seeds 0-2 (more if variance demands) | kitchen `m2_home`, `m2_rack`, houses s0-s2 | built |
| Task visibility | `task_visibility` | `discovered` fixed for paper runs (agents start knowing no task) | discovered (houses), posted (kitchens) | built (v2.0.00) |
| Spawn | `spawn.mode`, `min_separation` | `uniform`, 4 steps, seeded per episode seed (part of the seed's common random numbers) | uniform (houses), mission (kitchens) | built (v2.0.00) |
| Coverage decay | `exploration.search_ttl` | null for finite missions; finite only with continuous arrivals | null | built (v2.0.00) |
| Coverage sharing | `exploration.share_every` | 10 ticks; null as an ablation (no map merging) | 10 | built (v2.0.00) |
| Battery | `battery.*` | off (scaffold only; candidate factor later) | off | **planned** |
| Team size, task count | generator `agents`, task counts | fixed at 6 agents, 22 tasks (scaling is an open question) | 3 (kitchen), 6 (houses) | built |
| Dirty dishes | generator `dirty_clusters`, `cluster_tiles`, `stack_size` | fixed at 3 clusters, 1-3 tiles each, 1-5 dishes per tile; one wash task per cluster | as planned (houses) | built (v3.1.00) |
| Latency model | `latency.*` | fixed reference rates, plus `L = 0` and fixed-`L` ablations | defaults | built |
| Seed | `seed` | replicates; see section 4 | 0-39 | built |

The seed drives separate random streams for each purpose (spawn, channel, skills and drops, motion, churn). Two conditions run at the same seed therefore start from the same cells and face the same packet-loss and drop draws until their actions diverge.

## 3. Responses

Computed from the logs by `experiments/analyze.py`; its module docstring holds the exact definitions.

**Hero metrics** (one per figure):
1. **Completion within the tick budget.** Partial success (the share of tasks done) is reported next to it.
2. **Makespan** for finished runs, relative to the C0 oracle once it exists. Until then it is relative to the best known run, which is not an oracle.
3. **Wasted work:** duplicated-work ticks, contested tasks, and orphan delay after a crash.
4. **Cost per completed task:** tokens, messages, and LLM calls.

**Diagnostics:** voided tasks, splits (`split` events), drops, items broken, stale-belief and hallucination rates, blocked ticks, move conflicts, yields, and tasks outstanding over time (hidden tasks count as outstanding from tick 0).

**Discovery diagnostics** (v2.0.00, in the analyzer):
- discovery latency: mean ticks from a task's `exists` to its first `discovered`, plus the count never found;
- exploration ticks: agent-ticks in `explore` and `search`;
- team coverage over time;
- coverage messages, counted apart from coordination messages, so they do not inflate `messages_per_task`.

**LLM responses** (v5.1.00, in the analyzer):
- **False completion claims** (`false_completions`): a `report` (G) or `mark_done` (C2, C3) on a task that still needed work, because it was never done or someone worked on it after the claim. A claim that leaves a wash cycle to finish the task is true. This is RoCo's "false-done herding" (App. 12.1). C1 has no structured claim, so its false-done herding shows only in the dialogue text and is not counted.
- **Re-plan attempts:** `rejected_responses` (executor replies refused at the belief check or for not parsing; `parse_failures` counts the latter) and `plan_rejections` (C1 joint plans refused by the plan check). This mirrors RoCo's re-plan metric.
- **Ledger lag** (`ledger_lag_final`): the share of (agent, done task) pairs whose agent's final ledger copy does not show the task done. It measures divergence; there is none without a ledger.
- **Costs by kind:** `llm_calls_by_purpose` (`act`, `utility`, `dialogue`), `messages_by_kind` (consensus `cbba` and `ledger` traffic apart from claims and C1's `dialogue`), and `llm_wall_seconds` (a cost only, hard rule 1).

**Planned responses** (not yet in the analyzer): lease false positives (a lapsed lease whose owner was healthy but partitioned), and C1's false-done claims read from the dialogue text.

### Metric reference for the paper

Each run's `metrics.json` holds these fields (`experiments/analyze.py`, `run_metrics`). `print_summary` gives means with 95% bootstrap CIs per (environment, cell), and section 5 says how pairs are compared. "Per run" fields are summed over the episode. Rates are computed per run and then averaged over seeds; never pool counts across runs.

| Field | Formula (per run) | Source stream | Unit | Used for | Caveats |
|---|---|---|---|---|---|
| `within_budget` | every task done or voided before `max_ticks` | events, state | bool | Hero 1; H1 (against scale), H3 (against `p`, `r`, `k`) | Censoring: report the count alongside makespan |
| `partial_success` | done tasks / tasks (mission + found) | events | share | Hero 1, next to completion | Task counts grow with splits and unloads |
| `makespan` | last tick, finished runs only | state | ticks | Hero 2 | Unfinished runs have none |
| `makespan_vs_oracle` | makespan / C0 replay at the same (mission, seed) | events, `oracle_plan.json` | ratio | Hero 2; H2 (C3 vs C2), H1 | Needs a C0 run in the same folder |
| `duplicate_work_ticks` | ticks on a done task by agents other than its finisher | actions, events | agent-ticks | Hero 3; H1, H3 attribution | Finisher is the last tagged actor |
| `contested_tasks` | tasks worked on by more than one agent | actions | count | Hero 3 | Includes deliberate handovers |
| `orphan_delays` | crash to done, per crashed agent's task | events, actions | ticks | Churn figure | Censored crashes in `orphans_never_done` |
| `tokens_per_task` | (prompt + output tokens) / done tasks | llm_calls | tokens | Hero 4; Cost estimand | Cached replies still count (same prompt) |
| `messages_per_task` | non-coverage messages / done tasks | messages | messages | Hero 4 | Includes consensus traffic |
| `consensus_messages_per_task` | (`cbba` + `ledger` messages) / done tasks | messages | messages | Cost of consensus (C2, C3 vs C1, G) | 0 without a ledger |
| `llm_calls_by_purpose` | calls per `act`, `utility`, `dialogue` | llm_calls | calls | Cost table; C3's utility overhead vs C2; C1's talk overhead | Coded runs log `act` rows with 0 tokens |
| `false_completions` | claims (`report`, `mark_done`) on tasks never finished or worked on after the claim, one per agent-task | actions, events | count | RoCo's false-done herding; H3 (ledger vs none) | Not countable in C1 (claims are free text) |
| `false_completion_rate` | `false_completions` / distinct claims | actions | share | Same, normalized | None with no claim |
| `rejected_responses` | executor replies refused (belief check, unparseable, wrong task) | llm_calls | count | Re-plan attempts (RoCo) | Includes `parse_failures` |
| `replan_rate` | `rejected_responses` / `act` calls | llm_calls | share | Failure-mode table | 0 for coded runs |
| `plan_rejections` | C1 joint plans refused by the plan check | dialogues | count | C1 re-plans | C1 only |
| `ledger_lag_final` | share of (agent, done task) pairs whose final ledger copy is not `done` | ledgers, events | share | H3: replicas converge under loss and partition | None without a ledger; hard rule 10 applies across partitions |
| `stale_belief_rate`, `hallucination_rate` | outcome share among actions | actions | share | Failure-mode breakdown (section 5) | Only stale beliefs should rise with `p` |
| `discovery_latency_mean`, `undiscovered` | `exists` to first `discovered` | events | ticks, count | Discovery diagnostics | None in posted mode |
| `llm_wall_seconds` | summed wall seconds of LLM calls | llm_calls | seconds | Cost only (hard rule 1); run planning | Concurrent calls overlap; sleep inflates it; never a result |

## 4. Design structure

- **Baseline point:** `p = 0`, `r = ∞`, `k = 0`, `drop_prob = 0.025`, `m` = 22, `n` = 6, medium difficulty, on the generated houses, with discovered tasks, uniform spawns, and coverage sharing every 10 ticks.
- **One factor at a time for coverage.** Each sweep varies one factor and holds the rest at baseline (Liu et al. 2013 use this design). `experiments/plot.py` enforces it: each figure keeps only runs at baseline on the other axes.
- **Crossed only where a hypothesis claims an interaction:** condition × scale (`m`, difficulty, `n`), condition × `p`, condition × `r`, and `k` × memory. A full factorial is out of budget and would answer questions the paper does not ask.
- **Missions are blocks.** Results are reported per house and pooled with house as a block, never as if houses were replicates of one environment. If house difficulty varies widely, stratify by a difficulty index (DynaHMRC style: location spread, object count, collaboration need).
- **Common random numbers.** Every condition runs on the same (mission, seed) pairs, so differences are analysed per pair.
- **Denominator.** C0's replayed makespan at the same (mission, seed), reported as `makespan_vs_oracle`. Each C0 run also reports its proven bound and planned makespan, so a paper table can state how close the denominator is to optimal in the planning model.
- **Budgets and censoring.** Each mission's tick budget is 4x the slowest heuristic-plan replay over 3 uniform spawns, so no competing condition sets the yardstick. Houses s0/s1/s2 have budgets of 2100/2450/3500 ticks (v5.0.00). An unfinished run counts as a failure for completion and is left out of makespan, and the censored count is always reported next to makespan.
- **How runs are launched:** `python -m experiments.sweep configs/sweeps/<experiment>.yaml` runs every cell of an experiment at every seed, in parallel under one stamp. A spec lists episode configs (one per condition, `{env}` expanded over its environments), `first_seed` and `seeds`, `loss_levels`, `range_levels`, `churn`, and `workers`. C0 runs at baseline only. The spec is copied next to the runs, and a failed episode is reported without stopping the rest. `configs/sweeps/pilot_coded.yaml` is the coded pilot (C0 and G, 3 houses, 20 seeds).
- **Where runs go:** `python -m experiments.run_single <config> --run-name <exp>` writes to `runs/<exp>/<env>/<cell>/<YYYYMMDD-HHMMSS>/seed_<n>/`, with one stamp per launch. Analyze and plot one launch with `--stamp`; without it, repeats of a seed are pooled. Dev runs live apart in `runs/dev/<env>/` and are overwritten on each refresh. They are never results.
- **Not yet sweep axes:** mission scale (needs missions generated per level; list them as environments), memory kept vs wiped on rejoin (not built), and `drop_prob` (fixed by design).

## 5. Analysis plan

- **Messages have a deliberation cost.** Each delivered message makes its receiver re-decide after its current skill, paying `L` ticks. Curves against `p` may therefore not be monotonic, and a dip at low loss may be real, not a bug. Check re-decision counts before interpreting one.
- **Paired differences.** Condition A minus condition B per (mission, seed), with a 95% percentile bootstrap CI on the mean difference. The analyzer uses 2,000 resamples today; use 10,000 for the paper, as DynaHMRC does.
- **Normalized performance drop** for H3: `(score(perfect) - score(level)) / score(perfect)` per condition, as in AgentComm-Bench's protocol. This keeps a condition with a low ceiling from looking robust only because it had less to lose.
- **Crossovers** between condition curves over `r` (linear interpolation between the bracketing levels, bootstrap CI over seeds) are reported where they occur; there is no threshold hypothesis.
- **Interaction for H1:** the C1 minus C3 gap (and C1 minus C2) at each scale level, and the slope of that gap over `m`, difficulty, and `n`.
- **H2:** the paired C3 minus C2 difference in `makespan_vs_oracle` and completion, next to the token cost per completed task, at the baseline and at each scale level.
- **Failure-mode breakdown** from the outcome taxonomy (stale belief vs hallucination vs invalid call vs skill failure), per condition and level. Only stale beliefs should rise with `p`.
- **Seeds per cell:** set from the pilot variance of the noisiest hero metric, so that a planned effect size has a CI half-width below it. C0 and the coded-executor dev references are cheap and get more seeds (20-40).

## 6. Run budget (draft)

The September plan budgets about 400 LLM episodes on the cluster. The draft below assumes 3 houses × 3 seeds = 9 paired replicates per cell for the LLM conditions. Every paper condition except C0 uses the LLM executor, so only C0 and the coded dev references (coded G, R, and C2 when built) run 20+ seeds per cell at negligible cost.

| Sweep | LLM conditions | Levels beyond baseline | LLM episodes |
|---|---|---|---|
| Baseline (perfect comms) | C1, C2, C3, C4, G, L, R, C1+ledger, C3-noledger | 1 | 9 × 9 = 81 |
| H1: scale (`m` 11, 44; `n` 3, 12; difficulty easy, hard) | C1, C2, C3, G | 6 | 4 × 6 × 9 = 216 |
| H3: loss `p` | C1, C2, C3, C4 | 0.1, 0.3, 0.5, 0.7 | 4 × 4 × 9 = 144 |
| H3: ledger ablation | C1+ledger, C3-noledger | 0.3, 0.5 | 2 × 2 × 9 = 36 |
| H3: range `r` | C1, C2, C3 | 16, 12, 8, 6, 4 | 3 × 5 × 9 = 135 |
| H3: churn `k` × memory | C1, C2, C3, C4 | k = 1, 2 × kept, wiped | 4 × 4 × 9 = 144 |
| **Total** | | | **756** |

H2 (C3 vs C2) adds no rows: it reads the C2 and C3 cells of the other sweeps. The draft is far over the ~400 budget (v5.3.00 replaced the `rho` sweep with the larger scale sweep, and added C2 to the H3 sweeps). Large levels (`m` 44, `n` 12) also cost more per episode. Candidate trims: scale one factor only (tasks, or team size) and keep difficulty at three levels, drop C4 from the churn sweep, use 2 seeds per house for the ablation rows, cut the 0.1 loss level, or cut G from the scale sweep (coded G already shows duplicated work). Decide after the house pilot.

**Measured cost (house LLM pilot, laptop RTX 4070):** about 70 minutes of wall time per house episode, so about 400 episodes is about 470 GPU-hours. Paper sweeps therefore run on UMBC HPCF's `chip` cluster, one episode per GPU at a time, all on one GPU model (L40S, 40 cards; `design/cluster.md`); the cluster's minutes per episode, measured on `cluster_pilot_llm.yaml`, will size the final table.

Plus a frontier-model ceiling check on a few seeds, at perfect comms and at one high-loss level. Finalize this table after the M4 pilot measures episode cost and variance.

## 7. Runs executed

All runs below predate versioning; they ran on the code recorded as `[1.0.00]`.

| Date | Run | Setup | Headline |
|---|---|---|---|
| Oct 25-28 | Regression sweeps | `m2_home`, scripted and G, 40 seeds, `drop_prob` 0 / 0.025 / 0.2 | Both finish 40/40 at every drop rate. These are correctness checks, not results. |
| Oct 27 | Duplicated-work comparison | `m2_home`, 20 seeds, drop 0.025 | G wastes about 105 ticks per episode on duplicated work vs about 5 for scripted, at the same makespan. Stale-belief rate 0.109 vs 0.087. |
| Oct 28 | Headroom, kitchen | `pilot_headroom.yaml`, all 243 static assignments, drop 0, perfect comms | Best 292, median 320 (+10%), G 306 (+5%). Too small for allocation to matter. |
| Oct 28 | Headroom, houses | `pilot_headroom_houses.yaml`, 60 sampled assignments per house | Median +26 to +54% over the best sample. G is -19%, +6%, and -5% vs the best sample. 20-25 of 60 assignments unfinished per house. Allocation matters here. |
| Oct 28 | Plotting demo | `m2_home`, scripted and G, 6 seeds, `p` ∈ {0, 0.3, 0.6}, `r` ∈ {12, 8, 4}, one churn run (1 agent at tick 40, rejoin after 60) | G wastes about 105-280 duplicated ticks across levels. Scripted fails at `r` = 4 and 8 because its completion reports never arrive. Exercises the figures only. |

Runs on v2.0.00 (discovered tasks, uniform spawns, coverage sharing):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 4 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G finishes 120/120 (mean makespan 875 / 917 / 800 ticks). Scripted finishes 118/120; the 2 failures are an entry-table gridlock in s1. Correctness checks, not results. |
| Nov 4 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish (means 234-303 ticks). |

These numbers are not comparable with the earlier rows: discovery changes what every condition faces (MAJOR version).

Runs on v2.1.00 (C0 built, generator and movement fixes, regenerated houses):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 6 | C0 measurement | houses s0-s2, 5 seeds each, CP-SAT at deterministic limit 60 | Mean bound / planned / replay: s0 345/440/473, s1 451/451/477 (proven optimal), s2 337/463/512. Gap 25-31% on s0, 0% on s1, 36-39% on s2. G at drop 0 takes 1.39 / 1.62 / 1.40x the replay. |
| Nov 6 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G 118/120 (means 666 / 870 / 711); scripted 119/120 (means 743 / 1178 / 794). Correctness checks, not results. |
| Nov 6 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish (means 235-303 ticks). |

The generator fix moved washes and packs to their nearest appliance, so these houses differ from every earlier row. Earlier house numbers carry an artificial single-dishwasher bottleneck.

Runs on v3.0.00 (drops are a time penalty):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 7 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G 119/120 (means 674 / 866 / 737); scripted 120/120 (means 768 / 1196 / 805). Correctness checks, not results. |
| Nov 7 | Drop effect, houses | G, 40 seeds each, drop 0 vs 0.025 | 672 vs 674, 877 vs 866, 737 vs 737: flat at the paper's rate. G's one failure (s0 seed 34) occurs at both. |
| Nov 7 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish. Scripted 298 / 294 / 245, G 303 / 302 / 261. Still falls at 0.2, because a break voids a single-plate task and skips its wash. |

Runs on v3.1.00 (dirty-dish clusters, regenerated houses; not comparable with earlier house rows):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 7 | C0 measurement | houses s0-s2, 5 seeds each, CP-SAT at deterministic limit 60 | All 15 proven optimal. Mean planned / replay: 400/448, 453/489, 688/747. G at drop 0 takes 1.74 / 1.87 / 1.58x the replay. One long wash task sets the bound (open question 0a). |
| Nov 7 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G 120/120 (means 849 / 1059 / 1218); scripted 119/120 (752 / 1039 / 1086; s1 seed 24 fails). Correctness checks, not results. |
| Nov 7 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish. Scripted 298 / 294 / 248, G 303 / 303 / 264. |

First run through the sweep runner (launched just before the version bump, so its run.json files record 3.1.0; the simulation is the same in v3.2.00):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 8 | Coded pilot | `configs/sweeps/pilot_coded.yaml`: C0 and coded G, houses s0-s2, 20 seeds, `p` 0.1-0.7, `r` 16-4, churn `k = 1`; 720 episodes | No episode failed. G at baseline: 1.94 / 2.09 / 1.64x C0's replay, 20/20 finished. G's completion falls with loss and range on s2 (5/20 at `p = 0.7`, 1/20 at `r` = 4 or 6), and with churn everywhere (12-15/20). Makespan in those cells covers survivors only. Coded G, so not a paper result; its variance sets seeds per cell. |

Runs on v4.0.00 (per-dish wash goals, splits, pick cancels, executor and exploration fixes; not comparable with earlier rows):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 8 | C0 measurement | houses s0-s2, 5 seeds each, CP-SAT at deterministic limit 60 | 15/15 proven optimal. Mean planned / replay 400/448, 453/476, 688/761. No split in any replay. |
| Nov 8 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G 120/120 (877 / 1147 / 1227), about one split per run; scripted 120/120 (1120 / 1160 / 1435), slower because teammates no longer take over fixed assignments. Correctness checks, not results. |
| Nov 8 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish. Scripted 344 / 303 / 200, G 344 / 324 / 243: the put-back rule costs trips with one dishwasher. |
| Nov 8 | Coded pilot, rerun | `configs/sweeps/pilot_coded.yaml` on v4.0.00 | The coded pilot rerun (720 episodes, no failures) wrote 360 MB instead of 15 GB; G takes 1.99 / 2.39 / 1.64x C0's replay at baseline. |

Runs on v5.0.00 (washed dishes unloaded as found tasks; C0 plans unloads; not comparable with earlier rows):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 8 | C0 measurement | houses s0-s2, 5 seeds each, CP-SAT at deterministic limit 60 | 15/15 proven optimal. Mean planned / replay 445/503, 519/537, 806/865. Every unload in a replay was planned. |
| Nov 8 | Regression, houses | houses s0-s2, G and scripted, 40 seeds each, drop 0.025 | G 119/120 (1001 / 1460 / 1604; s1 seed 16 gridlocks after a four-way claim race on one unload); scripted 120/120 (1140 / 1318 / 1576). About one unload per run. |
| Nov 8 | Regression, kitchen | `m2_home`, posted mode, scripted and G, 40 seeds, drop 0 / 0.025 / 0.2 | All 240 finish. Scripted 407 / 383 / 252, G 449 / 417 / 294. |

Runs on v5.1.00 (LLM conditions built; coded results unchanged):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 8 | Regression, kitchen | `m2_home`, scripted and G (coded), 40 seeds, drop 0.025 | 382.75 / 417.05, identical to v5.0.00. Rerun after the `search` change: identical. |
| Nov 9 | Regression, houses | houses s0-s2, G and scripted (coded), 40 seeds, drop 0.025 | G 39/40 at 1001 / 1460 / 1606 (s1 seed 16 gridlock, as before); scripted 120/120 at 1140 / 1318 / 1576. Within 2 ticks of v5.0.00. |
| Nov 8-9 | Live kitchen pilots | `m2_home`, posted tasks, Qwen3-8B Q4_K_M on llama.cpp, seed 0, 900 ticks; G-llm, C2, C3, C1 on the final code (`runs/pilot_llm_c/`); earlier attempts in `runs/pilot_llm_a`, `pilot_llm_b` | No run finished. Tasks done: C3 7/8, G-llm 5/8, C2 4/7, C1 4/10. C2 and C3 duplicated no work and their ledgers converged; C1 duplicated 106 ticks. 283-572 LLM calls of 0.8-1.7k tokens each; 7-34 min of LLM wall time per episode. One seed each, so not results; failure modes are in `design/decisions.md`. |

Runs on v5.2.00 (kitchen model bake-off, one seed each, not results; `runs/final/`):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 10 | Model bake-off | `m2_home`, C3, seed 0, 900 ticks, same code; Qwen3-8B, Qwen3.5-4B, Qwen3.5-4B with thinking, Ministral 3 8B | Tasks done 7/10, 3/6, 1/6, 8/11. No server crash. Thinking costs about 1,000 output tokens (27 ticks) per decision. Pilot for choosing which models enter the house pilot. |

Runs on v5.3.01 (house LLM pilot, not results; `runs/pilot_llm/`, stamp `20261112-223120`):

| Date | Run | Setup | Headline |
|---|---|---|---|
| Nov 12-14 | House LLM pilot | houses s0-s2, discovered tasks, seeds 0-1, budgets 2100 / 2450 / 3500 ticks; G-llm, C2, C3, C1; Qwen3-8B Q4_K_M, `--ctx-size 16384` | 21 of 24 episodes ran; 3 stopped on server "context size exceeded" errors (G-llm s1 and s2 seed 0, C1 s2 seed 1). No episode finished all tasks. Mean share of tasks done: C3 0.84 (n=6), G-llm 0.80 (n=4), C2 0.79 (n=6), C1 0.46 (n=5). Paired C3 minus C2 by house and seed: +0.05 (4 higher, 1 lower, 1 tie). C1 wasted 820-1,430 ticks on duplicate work with 6-12 contested tasks; C2 and C3 mostly none. G-llm made 2-5 false completion claims per episode, C2 and C3 0-2. 0.8-3.9k LLM calls and 1.8-6.7M tokens per episode; 55-147 min of LLM wall time. |

## 8. Design techniques borrowed from related work

- **DynaHMRC (dynahmrc2026; read in full):**
  - a factorial grid of task family × difficulty tier × team composition × dynamic setting;
  - a composite difficulty index;
  - step caps per tier;
  - 3 object sets × 3 repeats per cell;
  - 95% bootstrap CIs with 10,000 resamples and paired significance tests;
  - component and backbone ablations;
  - trigger-timing and severity levels for dynamic events;
  - token cost tables, redundant and conflicting action counts, and a failure-mode taxonomy.
  - *Adopted:* paired bootstrap, ablations, a difficulty index for blocking, timing levels for churn, the failure taxonomy.
- **Liu et al. 2013 (CoBA; read in full; the bib entry still needs the user's OK):** one factor at a time around a defaults table (robots, task arrival rate, comm reliability, heterogeneity, acquaintance), 20 runs per setting, mean ± SD of completed and failed tasks, two auction baselines, tasks remaining over time, and a physical-robot check. *Adopted:* one-factor-at-a-time coverage around a baseline, tasks outstanding over time.
- **RoCo (mandi2024roco; read in full Nov 2):** 6 tasks varied along task decomposition, observation symmetry, and workspace overlap; 20 runs per task; success within a round budget, steps in successful runs, and re-plan attempts; a full-observation "Central Plan" LLM as oracle; "w/o History" and "w/o Feedback" ablations (feedback-free runs get twice the episode budget); a failure analysis (false-done herding, open-loop execution errors). *Adopted:* the C1 harness and protocol, re-plan attempts and false completion claims as responses, and history and feedback as candidate C1 ablations.
- **AgentComm-Bench (bansal2026agentcomm; from our notes):** sweeps six communication impairments and reports normalized performance drop. *Adopted:* normalized drop for H3.
- **Classical auction studies (otte2020auctions, choi2009cbba, ponda2010dynamic, chen2024scalable; from notes or recall, verify before citing their methods):** sweep message loss and range averaged over random placements; compare auction score against an optimal or centralized solution; vary team size to compare centralized and decentralized structures. *Adopted:* the C0 reference and the loss and range sweeps. Team-size scaling is an open question.

## 9. Open DOE questions

0. ~~**H1's measure now that `rho` is out of C0.**~~ Void: `rho` was removed Nov 10, 2026 (v5.3.00). H2's measure is `makespan_vs_oracle`, C3 vs C2.

0a. **One wash task dominates each house (v3.1.00).** Carrying is one dish per trip, so the longest wash task alone (many dishes, or a far surface) is 50-67% of C0's planned makespan, and it takes 1.4-1.7x as long with an arm as with a carrier. Proof of optimality now comes from this chain, not from a better search. Makespan may end up measuring mostly who gets the long job. Candidate fixes: carry stacks, or keep dirty dishes near the kitchens. See `decisions.md`, Nov 7.

1. Seeds per cell for the LLM conditions (9 paired replicates in the draft). Decide from M4 pilot variance. *Nov 9:* a kitchen LLM episode takes 7-34 min of LLM wall time on the laptop, and a house episode an estimated 1-4 h, so the 24-episode house pilot (`configs/sweeps/pilot_llm.yaml`) needs one to three days locally or a cluster run.
2. Whether team size and task count become scaling factors, or stay fixed.
3. Churn timing: one window, or early / mid / late levels.
4. Whether to add a difficulty index to the generator and stratify houses by it.
5. Whether `drop_prob` stays fixed at 0.025 for all paper runs. It adds time penalty and variance. At 0.025 makespan is flat in the houses; at 0.2 it still falls in the kitchen, because a voided single-item task removes its wash.
6. ~~The `rho` mechanism.~~ Void (removed Nov 10, 2026).
7. **Scale levels and the difficulty definition** (proposed in the factor table; also `decisions.md`, open question 8): confirm, and check that `n = 12` fits the hardware.
8. **Utility inputs and recompute rule** (`decisions.md`, Future work): today the utility call sees only the capability card and task text.

## DOE change log

- **Nov 16, 2026 (v5.4.02):** the cluster is UMBC HPCF `chip`; one experiment uses one GPU model (L40S by default), extending the comparability rule to GPU models. No factor, level, or response changes.
- **Nov 14, 2026 (v5.4.00):** sweeps can run on the GPU cluster as Slurm array shards under one stamp (`design/cluster.md`); specs gain optional `llm:` and `only:` keys. Comparability rule: a comparison never mixes laptop and cluster runs, so the house pilot is rerun on the cluster (`configs/sweeps/cluster_pilot_llm.yaml`). §6 records the measured laptop cost (about 70 min per house episode).

- **Nov 14, 2026 (v5.3.02):** house LLM pilot recorded in §7 (21 of 24 episodes; 3 stopped on server context overflow, to be rerun at `--ctx-size 32768`). No factor, level, or response changed.

- **Nov 12, 2026 (v5.3.01):** sweeps run seed by seed (every environment and cell at a seed before the next seed), so a long local sweep that stops early is still balanced across cells. The house LLM pilot (`configs/sweeps/pilot_llm.yaml`, Qwen3-8B, 24 episodes) is set to run in that order.

- **Nov 10, 2026 (v5.3.00):**
  - Hypotheses revised: H1 structure (scale), H2 reasoning (C3 vs C2), H3 resilience (loss, range, churn). The old connectivity-threshold H3 is dropped.
  - `rho` removed as a factor, a generator setting, a metrics field, and a figure; houses regenerated (maps, configs, budgets identical).
  - New planned factors: tasks `m`, team size `n`, difficulty (proposed levels); the run budget now totals 756 untrimmed episodes, over the ~400 budget.
  - Figures renamed: `fig_h2_loss` to `fig_h3_loss`; the H1 scale figure and H2 table are planned.

- **Nov 10, 2026 (v5.2.00):**
  - New config field `llm.thinking` (off for every paper condition so far); the reply schema now fixes each tool's argument count (LLM conditions only; coded results unchanged).
  - Model bake-off run in the kitchen (section 7). The model is not frozen yet.
  - Nov 9 LLM pilots predate the schema change and are not comparable with later LLM runs.

- **Nov 9, 2026 (v5.1.01):**
  - Metric reference table for the paper (section 3).
  - New rates: `false_completion_rate`, `replan_rate`, `consensus_messages_per_task`.
  - Model candidates and bake-off plan (`design/llm_models.md`). The bake-off is not run yet: host memory on the laptop blocked it. No results change.

- **Nov 8, 2026 (v5.1.00):**
  - The LLM conditions C1, C2, and C3, and G with the LLM executor, are built and run against one llama.cpp server. The `executor` factor is built, with new fixed `llm:`, `cbba:`, and `dialogue:` blocks.
  - New responses: false completion claims, re-plan attempts (rejected responses, parse failures, C1 plan rejections), ledger lag, LLM calls by purpose, and messages by kind.
  - Sweep cells name G with the LLM executor `greedy_llm`, and the generator writes `episode_{greedy_llm,numeric_cbba,llm_cbba,llm_only}.yaml` per house.
  - Coded results are unchanged (kitchen identical, houses within 2 ticks), so earlier v5.0.00 rows stay comparable. A `search` that has swept its round now ends with a decision point (every condition).
  - First live LLM runs: kitchen pilots of G-llm, C1, C2, C3 (section 7).

- **Nov 8, 2026 (v5.0.00):**
  - Washed dishes seen inside a dishwasher become unload tasks, so task counts grow further during a run.
  - C0 plans unloads between washes on a shared machine.
  - Budgets regenerated (2100 / 2450 / 3500).
  - Perception settings added to Key Definitions/Concepts.
  - Every earlier run is incomparable.
- **Nov 8, 2026 (v4.0.00):**
  - Wash goals are per dish, and a cycle started mid-load splits its task, so task counts can grow during a run (each split adds one task, opened at its `exists`).
  - Exploration now also looks at furniture beside seen floor, so exploration ticks rise for every condition.
  - Logs are gzipped and compact; read them with `sim.logger.read_stream`.
  - Every earlier run is incomparable.
- **Nov 8, 2026 (v3.2.00):**
  - Key Definitions/Concepts section added.
  - Experiments now launch from `configs/sweeps/*.yaml` through `experiments/sweep.py`, which closes the config-file gap. `rho` and churn memory are not sweep axes yet.
  - First run through it: the coded pilot.
- **Nov 7, 2026 (v3.1.00):**
  - Dirty dishes come in clusters, one wash task each, replacing `dish_batch_size`. Budgets re-measured; earlier house numbers are not comparable.
  - C0 has no search cost and no `rho`, and no posted-task variants are planned. New open items: H1's measure (question 0) and the dominant wash task (question 0a).
  - Run layout: named experiments keep their logs by stamp, and dev runs live apart under `runs/dev/`.
- **Nov 7, 2026 (v3.0.00):**
  - Drops became a time penalty: a pause for non-fragile items, a cleanup reflex for fragile ones. Only the broken item is lost, and the agent keeps its task.
  - The `cleanup` task is gone. New responses: `drops`, `items_broken`.
  - All earlier runs with drops are no longer comparable.
- **Nov 6, 2026 (v2.1.00):**
  - C0 built: the denominator is its replayed makespan per (mission, seed), with a proven bound and planned makespan reported alongside.
  - New responses: `makespan_vs_oracle`, `oracle_bound`, `oracle_planned`, `solver_gap`.
  - Budgets moved from 2x greedy to 4x the oracle replay.
  - Generator and movement fixes change every house result; earlier house numbers are not comparable.
- **Nov 4, 2026 (v2.0.00):**
  - Paper runs use discovered tasks (agents start knowing no task), seeded uniform spawns, and shared coverage with an optional decay.
  - New fixed settings: `task_visibility`, `spawn`, `search_ttl`, `share_every` (null sharing is a candidate ablation).
  - New diagnostics: discovery latency, exploration ticks, coverage over time, coverage messages.
  - Budgets are re-measured; the battery is scaffolded but off.
  - Earlier pilot numbers are no longer comparable.

- **Nov 2, 2026 (v1.0.02):**
  - Conditions are now defined by `design/conditions.md`, and the condition factor is split into allocator × utility × ledger × executor.
  - C1 is RoCo-style with no ledger; the H2 ablations are C1+ledger and C3-noledger.
  - G and R use the LLM executor in paper runs. The draft budget rises to 513 LLM episodes, over budget with trims listed.
  - Added two planned responses (false completion claims, re-plan attempts) and RoCo's design to the borrowed techniques.
- **Oct 28, 2026 (v1.0.01):** file created. Recorded the pilot-stage design, the estimands, the draft run budget, and the runs executed so far.
