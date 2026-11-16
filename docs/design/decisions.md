# Decisions log

Newest entries last within each section. Format: **decision**, then *why*. Update this file in the same commit as any design change. Dates before October 15 are approximate because they come from an earlier planning session.

## Settled

### Project scope and experiment design (Sept 2026, planning session)

- **Pivot from gated continual RL to decentralized LLM + auction coordination.** The new topic has a clearer gap and fits the semester.
- **Core contribution is leaderless semantic allocation.** Per-agent LLMs generate bundle-independent utilities `u_ij`; a consensus auction resolves allocation; a replicated ledger tracks execution.
  *Why bundle-independent:* CBBA's convergence and approximation guarantees need diminishing marginal gains, so the LLM must not make a task's value depend on the rest of the bundle.
- **The LLM has exactly two jobs:** utility generation (cached per agent-task pair) and execution through tool calls.
- **Ledger semantics.** Status only moves forward (unclaimed, claimed, in progress, done; "done always wins"). Entries carry owner, bid, version, lease expiry. Gossip on contact. Leases detect crashed owners.
- ~~**`rho` knob** (fraction of tasks with language-only constraints) is the main lever for H1, and guards against an "LLM just adds noise" outcome.~~ *Removed Nov 10 (v5.3.00):* H1 is now about mission scale; see the Nov 10 entry below. Older entries below still mention `rho` as it was then.
- **Heterogeneous team of about 6 agents**, 15-30 tasks per mission, precedence constraints.
- **Custom Python environment**, not an existing benchmark. *Why:* no existing environment combines LLM agents, heterogeneous allocation, message-level communication impairment, and churn (PARTNR has no communication model; AgentComm-Bench has no LLM agents).
- **Channel model** is `p_deliver = 1 / (1 + (d/d0)^alpha) * wall_penalty^walls_crossed`, bounded, never a raw `1/d^2`. Plus base loss `p` and optional latency.
- **Rendering is pixel-art, drawn in code or from CC0 assets.** No ripped game assets.
- **TraderBots is cited, not implemented.** *Why:* scope before Dec 17.
- **The HTML demo's greedy auction is not CBBA.** Full CBBA and MURDOCH must be implemented in Python.

### Conditions (resolved Oct 2026)

- **Five conditions, as in the abstract:** C0 oracle, C1 LLM-only, C2 numeric CBBA, C3 LLM+CBBA, C4 LLM+MURDOCH.
  *Why this supersedes the earlier C0-C5 table:* packet loss, range, and churn are now *sweeps applied to every condition*, not separate conditions. C2 stays essential: without it we cannot say the LLM helps beyond CBBA itself. ~~C1 also gets the ledger, so the comparison isolates the allocation mechanism.~~ *Superseded Nov 2:* C1 has no ledger (RoCo-style); the C1+ledger ablation isolates the mechanism instead.

### Simulator architecture (Oct 15 and Oct 22, 2026)

- **Not Gymnasium/Stable-Baselines3 as the core.** Those are for RL training; our agents are LLMs. *Add* a PettingZoo Parallel API wrapper for familiarity and reproducibility.
- **Simulated time, not wall-clock.** *Why:* results must not depend on hardware. *Supersedes* the earlier "100 ms real-time ticks with blocking calls" suggestion.
- **Event-driven (semi-Markov) agents.** The LLM is queried only at decision points; skills run over many ticks. Both agents start with a decision point at tick 0 (initial observation + tasks, one call).
- **While an LLM call is in flight, the sim is paused in wall-clock time but the sim clock does not advance.** Deliberation is charged in simulated time through a latency model; the deciding agent idles `L` ticks while teammates keep acting.
- **Latency model:** `L = ceil((prompt_tokens / R_prefill + output_tokens / R_decode) / seconds_per_tick)` at fixed reference rates. Proposed defaults: `R_prefill = 2000 tok/s`, `R_decode = 40 tok/s`, `seconds_per_tick = 1.0`. Run a sensitivity ablation with `L = 0` and fixed `L`. *Values are provisional; tune in pilot runs and then freeze.*
- **Consensus rounds and messages cost ticks too** (one tick per hop plus channel latency), so CBBA and LLM negotiation share one timeline. This gives H3's "replanning horizon" a precise definition in ticks.
- **Observation frozen at query time; action applied `L` ticks later** after checking preconditions against ground truth. Stale-plan failures are an intended effect, not a bug.
- **One model server, many agents** (llama.cpp server with parallel slots, or vLLM). *Correction:* an earlier note implied one model copy per agent; that is wrong. An 8B 4-bit model is roughly 4.5-5.5 GB of weights on the 8 GB laptop GPU, with room for several agents' prompt histories. Drop to a 4B model if memory is tight.
- **Perception is deterministic code** (range, view cone, line of sight, containment), feeding a per-agent belief graph with timestamps and negative evidence. The LLM never does perception. `hidden_attrs` (language-only facts) are never perceivable.
- **Two failure types are logged separately:** hallucination (acted on an object the agent never observed) and stale belief (acted on something that used to be there). Only the second should worsen as communication degrades.
- **Perception noise is an ablation, off by default.**
- **Logging:** append-only JSONL per episode; both `tick` and `wall_time` recorded on every row; metrics computed from logs afterwards; response cache keyed by prompt hash.
- **Compute plan:** laptop (RTX 4070, 8 GB) for development and pilots; UMBC cluster Slurm job arrays (one job per condition x seed) for the roughly 400-episode sweeps.
- **Ceiling check with a frontier API model** on a small subset (a few seeds at perfect communication and one high-loss setting), to pre-empt "your model was just too small."

### M1 sim core (Oct 24, 2026)

- **Per-agent speed.** BFS returns path length in *steps*. Travel time is `max(1, ceil(steps / speed))` ticks. LLM text shows steps.
  *Why:* lets the team be heterogeneous in speed while keeping one distance function (hard rule 8). Agents share one passability map; per-agent mobility (a drone flying over counters) was considered and deferred.
- **Scenarios are an ASCII map plus a YAML mission**, written by hand for now. A seeded mission generator (for `rho` and difficulty) comes with the sweeps.
- **Static furniture is known from the map at tick 0. Movable objects must be perceived.**
  *Why:* agents need fixture locations to navigate, and a household robot would have the floor plan. Without this, scripted and LLM agents would have to explore before they could name a dishwasher. Revisit if the exploration cost itself becomes a variable of interest.
- **Decision-point rules.** A policy returns a list of calls. Decisions happen at episode start, when the list is used up, on any failure, and on rejoin. A message triggers a decision immediately only when the agent is idle or in `wait`; otherwise it is read after the current skill. A response containing any call that fails the belief check is rejected whole, and the agent still pays `L`.
- **Failure taxonomy:** `hallucination`, `invalid_call`, `stale_belief`, `skill_failure`. Failures caused by the agent's own state (hands full, not holding the object) are `invalid_call`, not `stale_belief`.
  *Why:* the stale-belief rate should measure world drift only, which is what communication loss is predicted to worsen.
- **Communication range `r` is one swept value for the team**, not a capability-card field. *Why:* per-agent radios would confound the `r` sweep. Radio distance is Euclidean.
- **Perception runs at the start of each tick, before decisions.** A frozen prompt therefore shows what is in view at that tick.
- **M1 simplifications** (revisit if they matter): agents do not perceive each other (they do block each other since Oct 25), there is no payload or weight, objects nest only inside furniture, and "memory wiped" on rejoin is not implemented (open question 7).
- **Dependencies:** `pyyaml` at runtime (configs and missions are YAML); `pytest` and `ruff` for development. They are installed in a project-local `.venv`.

### Household task templates (Oct 25, 2026)

- **A task is an orchestrated series of primitives** (clean a plate, unpack the dishwasher, pack groceries, take out trash), instantiated from `configs/task_templates.yaml` by `kind`. *Why:* mimics a generalist policy given a subtask ("pack the dishwasher") after it wins the bid (PARTNR, Helix style); `repeat k` is unrolled at load time, and done stays a ground-truth goal check so orphaned tasks are resumable.
- **New primitives:** `start`, `lock`, `unlock`, `tie_bag`. **New objects:** trashcan with `capacity` (each item counts 1, perceivable `fill` gauge), dishwasher with `washing_time` (60 ticks, `open` refused mid-cycle).
- **Locks are world-level mutual exclusion, not part of the ledger.** Blocked calls fail as `stale_belief` (a lock the agent could not have known about). Released on `unlock` or holder crash. *Superseded later the same day by workspace reservations (next section).*
- *Superseded Nov 7 (drops are a time penalty):* **Fragile drop shatters; sturdy drop is re-picked.** Shatter voids the unfinished tasks that needed the object; `voided` counts as finished for `after` and episode end but is never logged `done`. *The `on_break` cleanup list was superseded by perception-triggered discovery (next section).*
- **Scripted test policy** now retries a failed step after a wait instead of restarting the task (restarting is not idempotent once locks and held objects exist).
- **Open:** conditional tasks ("trash only if more than half full") are judged by the LLM from the task text and fill gauge, so the scripted policy always runs them; `set_table`, `wipe`, `heat_meal`, `do_laundry` templates are not written yet.

### Fleet physics: workspaces, bodies, slots, drops, discovery (Oct 25, 2026)

Answers to the four open questions from the household-templates pass.

- **Locks reserve the direct workspace, not the whole object.** `lock(target)` reserves the target's cells within the agent's reach plus every free floor cell around them; `lock(cabinet_1[2])` reserves one slot. Other agents cannot work on reserved slots and route around reserved floor. An agent may hold several (dishwasher plus one end of a cupboard), so a second agent can still fill the far end of the cupboard.
- **Agents physically block each other.** Every agent occupies its cell (offline agents stay as obstacles); moves route around bodies and reserved cells, wait when there is no way around, and fail `stale_belief "path ... blocked by X"` after `BLOCKED_PATIENCE_TICKS` (6). Agents step in sorted id order, so they never swap through each other. Blocked ticks are logged per agent (congestion). Doorways are 2 cells wide in `home_medium.txt` so the default map is deadlock-free; `home_medium_tight.txt` keeps 1-wide doors for a tight-spaces experiment. *Tried and reverted:* letting a blocked agent close in along the static path to queue; agents then clustered at busy workspaces and blocked each other (27/40 seeds finished instead of 40/40).
- **Distance stays static (hard rule 8).** `PathPlanner.steps_to_any` without `avoid` (walls and furniture only) feeds bids, LLM text, and the oracle. Execution passes `avoid` (bodies, reservations) to detour. Detours cost ticks but change nobody's distance information.
- **Furniture spans cells; each cell is a slot with `slot_capacity`.** `place(item, cabinet_1)` uses the closest reachable slot with room and otherwise fails `cabinet_1[0], cabinet_1[1] full`; `cabinet_1[i]` names a slot in `move_to`, `place`, and `lock`. Per-slot fill is perceived (only when the inside is visible) and remembered after the door closes; the map gives location, not contents. Plates stack within a slot. Missions keep total cupboard slots equal to the number of dishes.
- **Drying-rack fallback.** If every cupboard the unpacking agent believes has room is full, the dish goes on the drying rack (unlimited) and the unpack task counts it handled (`in_kind cabinet|drying_rack`). A clean dish on the rack triggers `store_dish`, whose executor picks a cupboard believed to have room, else one it has not looked inside yet.
- *Handling superseded Nov 7; the knob stays:* **Drops: one knob `faults.drop_prob` (0.025 in `m2_home`),** rolled once per pick, place, and move while holding. The object lands on the agent's own tile (replaces "adjacent surface"). `BREAKABLE_TYPES = (plate, cup)` shatter; everything else is re-picked. Breakability is physical by type, so `hidden.fragile` stays a pure language-only constraint for `rho`. Per-skill `success` still models non-drop failures (a fumbled place keeps the object in hand). *Why 0.025:* the paper is about fleet coordination, not grasping; drops should be an occasional disturbance, not the main effect.
- **`start` hands the dishwasher's reservation to the running cycle**, which releases it when the wash ends, so any agent can then open and unload it.
- **Tasks are discovered by perception, from beliefs only.** Triggers: shards on the floor (`cleanup`), a dirty dish outside a dishwasher (`clean_plate`), food on an `entry` surface (`pack_groceries`, one task per item so CBBA can bundle), a clean dish on a drying rack (`store_dish`), a trashcan gauge above half (`take_out_trash`). Ids derive from the triggering object, so two finders create the same task. The finder logs `discovered` and broadcasts `task_found` over the lossy channel; teammates learn it from that message or their own perception. Discovered tasks go to the auction (M3). Until then the scripted stand-in lets an idle agent adopt a known discovered task.
- **Docks.** Idle scripted agents park next to `dock_<agent id>` so they do not stand in doorways. Docks also give the battery/charging idea a place to land later.
- **Scripted stand-in rules** (scaffolding only, replaced by the ledger and CBBA): report a voided task at once; release reservations whenever a task is left unfinished; drop a discovered task whose object a teammate is holding; treat "already open/closed" as the step being met.
- **Sweep result (scripted, `m2_home`, 40 seeds each):** all 40 finish at `drop_prob` 0.025 (a shatter in 7) and at 0.2 (a shatter in 30).
- **Open:** a lease/TTL safety net for reservations (an LLM agent that forgets to `unlock` will block teammates until it crashes); whether triggers should also fire from `told` beliefs once gossip is wired in.

### Baselines, metrics, and the headroom pilot (Oct 27, 2026)

Prompted by DynaHMRC's experimental design (success, partial success, steps, and comm cost against MCTS and CoELA, with bootstrap CIs) and by Liu et al. 2013 (AND/OR task trees).

- **New baselines and ablations** (CLAUDE.md, outline, abstract): **G** greedy without LLM or consensus (nearest unclaimed task by static steps, broadcast claim); **L** elected-leader LLM (one agent elected per mission assigns every task); **C1-noledger** and **C3-noledger** (*renamed Nov 2:* C1 itself has no ledger, so the C1 ablation is now **C1+ledger**); **R** random. The five conditions keep their names and order.
  *Why:* without G we cannot say consensus beats *any* coordination, and DynaHMRC's own MCTS baseline beat it on easier settings. L tests "leaderless beats elected leaders" head on and answers open question 5 (a mission leader is also a central orchestrator, which a partition or the leader's crash cuts off). The -noledger ablations are the only way to split H2's gain between consensus and ledger.
- **G is built** (`allocation/greedy.py`, `condition: greedy`). It reuses the scripted executor; only the task choice differs. With perfect comms and no drops it already duplicates work: two agents claim the same task in the same tick before hearing each other.
- **Episode configs name their condition** (`condition: scripted | greedy`), so a condition is a config file, not a flag.
- **Every action is tagged with its task** (`DecisionResponse.task_id`) and its start tick, so duplicated work is measurable from logs. Mission tasks are logged `posted` at tick 0, so the analyzer knows the full task set.
- **`claim` and `release` primitives** broadcast a task id and log `claimed` and `released` events. Only tasks chosen at run time are claimed. *Why:* a fixed assignment is agreed in advance, and every message costs its receivers a decision (and `L` ticks). Claiming every task made the perfect-comms smoke run slower than the lossy one.
- **Metrics** (`experiments/analyze.py`, from logs only): success (all done, none voided), partial success, makespan for finished runs, voided, duplicate-work ticks (ticks on a task another agent finished), contested tasks, orphan delay, stale-belief and hallucination rates, blocked ticks, messages, LLM calls, tokens; 95% percentile-bootstrap CIs over seeds, standard library only. *Caveat:* drops shorten makespan by voiding work, so makespan is always reported next to voided.
- **Robustness fixes found by the G sweep:** a believed object can stop existing (bagged trash), which is now an ordinary `stale_belief` instead of a crash. `goal_holds` treats a deleted object as an unmet goal. A coded agent only takes a task whose pick targets it believes in or that the task reveals by opening something, and leaves a task whose object it sees shattered. The agent now knows its own workspace reservations (`DecisionRequest.own_reservations`) and releases leftovers between tasks; queued unlocks were lost when a decision replaced the queue.
- **`clean_plate` fetches the plate before locking the dishwasher.** *Why:* holding the dishwasher's workspace while searching for the plate blocked every teammate.
- **Coded executors explore for unseen objects:** a pick of an object never seen becomes a walk to the nearest furniture not yet visited (static steps), until perception finds it. Without this, a fixed assignment of a task to an agent that had never seen the object retried a hallucinated pick forever.
- **Idle agents yield (body-level, all conditions).** A blocked mover nudges an idle teammate (nothing queued, or waiting) off its path or the cells next to its target. Busy agents never yield, so contested workspaces still block. Counted per agent as `yields`. *Why:* the kitchen is too cramped to place docks clear of every workspace; a parked agent blocked the only free cell next to the counter.
- **Sweep (40 seeds, `m2_home`):** scripted and G both finish 40/40 at `drop_prob` 0, 0.025, and 0.2.
- **Headroom pilot** (`configs/pilot_headroom.yaml`, drop 0, perfect comms, all 3^5 = 243 static assignments): best 295 ticks, median 331 (+12%), worst 543 (+84%); the hand assignment 305; G 306 (+4% vs best). There is some headroom, mostly between good and bad assignments, and G already sits near the best. *Read:* on this 22x8 map with 3 agents and 5 mission tasks, a smarter allocator cannot beat G by more than about 4% on makespan. G's cost shows elsewhere: over 20 seeds at drop 0.025 it wastes about 105 ticks per episode on duplicated work (scripted hand assignment: about 5) and has a higher stale-belief rate (0.109 vs 0.087), at the same makespan. So bigger generated missions (plan item 2) are needed before makespan can separate the allocators; duplicated work and stale beliefs already can.
- **Task hierarchy (from Liu et al.):** current tasks are AND-only sequences allocated whole; OR exists only as execution fallbacks (rack, unseen cupboard). Splitting AND blocks into allocatable subtasks is planned; OR stays at execution, since coupled constraints break CBBA's bundle-independence assumption.

### Movement, spawns, generated houses, hero metrics (Oct 28, 2026)

- **Simultaneous movement with a seeded 4-way-stop rule** (`sim/motion.py`). All movers move together in rounds, one round per cell of speed. Cells occupied at the start of a round are off-limits, so there is no following, no chains, and no swaps. When several movers propose the same cell, a seeded RNG (`seed:motion`) picks the winner, and the losers count a `move_conflict`.
  *Why:* agents used to move one at a time in id order, so the lower id always won ties (a bias in every congestion number). Over 200 seeds each side now wins 40-60%. Collisions stay impossible by construction.
- **Local replanning.** A mover follows the cached static shortest path and runs an obstacle-aware BFS only when its next cell is blocked. *Why:* recomputing an avoid-aware BFS every sub-step was 17 of 22 seconds per generated-house episode; this cut it to about 3 s with the same detour lengths in tests. It also matches robot local planners. It reintroduces "close in and queue behind the blocker", which with yielding and the head-on rule now finishes all 240 regression runs (it did not before those existed).
- **Patience counts progress, not motion.** A tick is stuck unless static steps to the target drop below the best so far. *Why:* two agents shuffling around each other in a doorway (found in greedy seed 32 at drop 0.2) moved every tick and never ran out of patience.
- **Head-on rule:** when two stuck movers stand on each other's static paths, a seeded coin flip picks one to step aside off every mover's path. This is the 4-way stop for a head-on meeting.
- **Spawn validation** (`validate_spawns` in `sim/scenario.py`): start cells must be floor, not furniture, distinct, and able to reach every piece of furniture. The check reuses the one static distance function (hard rule 8). *Extended Nov 4:* generated houses now spawn uniformly at random (seeded); see below.
- **Seeded house generator** (`experiments/generate_mission.py`, `configs/generator/house_48x32.yaml`). It builds 48x32 houses with 7 rooms (recursive splitting, 2-wide doors, an outside strip), two kitchens, and about 23 tasks: 2 unpack, 4 batched washes, 3 pack, 2 trash, 12 tidy. Docks sit next to each agent's start. Cupboard slots are at least the number of dishes. The tick budget is 2x the greedy makespan without drops. Same seed gives byte-identical files.
  *Why this size:* Liu et al. 2013 (Fig. 5) found differences between allocators negligible for small teams, and our kitchen pilot agreed. The house is sized so travel dominates and comm range can partition the team, while a 400-episode sweep stays fast in Python.
- **New templates:** `clean_dishes` (a batch per wash cycle, reserving the dishwasher only for close-and-start) and `tidy` (pure travel, no shared resource). Mission tasks may carry a `note` appended to the description (the `rho` text), and generated missions record `rho`.
- **Trigger coverage:** any known task about an object covers it, except follow-up kinds (`cleanup`, `store_dish`), which need a task of their own kind. *Why:* a dirty dish in a `clean_dishes` batch must not also spawn `clean_plate`, while shards from a voided wash must still spawn a cleanup.
- **Executor:** an object believed missing is treated like one never seen. Greedy does not take the task, an agent on it explores, and a failed "not here" pick on a runtime-chosen task releases it. *Why:* an agent that adopted a cleanup by radio walked to the shards' stale location forever.
- **Headroom pilot results** (drop 0, perfect comms):

  | Mission | Assignments | Best | Median | Worst | Mission's scripted | Greedy |
  |---|---|---|---|---|---|---|
  | Kitchen | all 243 | 292 | 320 (+10%) | 544 | 300 | 306 (+5%) |
  | House s0 | 60 sampled | 848 | 1158 (+37%) | 1379 | 838 | 685 (-19%) |
  | House s1 | 60 sampled | 895 | 1374 (+54%) | 1873 | 1475 | 947 (+6%) |
  | House s2 | 60 sampled | 966 | 1216 (+26%) | 1769 | 1287 | 915 (-5%) |

  - In each house, 20-25 of the 60 sampled assignments did not finish within budget.
  - The "best of 60 random" is a weak reference; greedy reassigns as it goes and beats it in two houses. Gap-to-oracle numbers need C0.
- **Hero metrics** (analyzer): completion within the tick budget, makespan (relative to the oracle once C0 exists), wasted work, cost per completed task. Also added: tasks outstanding over time, messages and tokens per task, move conflicts, yields, `rho`, comm range, crash ticks.
- **Hero figures** (`experiments/plot.py`):
  1. H2: completion and wasted work against `p`, as two panels.
  2. H3: completion against `r`, with crossovers marked.
  3. H1: makespan relative to the best known run against `rho`.
  4. Churn: tasks outstanding over time with crash ticks marked.
  5. Cost: tokens against completion.

  Each figure is written as PNG, PDF, and a CSV table twin, and holds the other sweep axes at baseline. *Why the last rule:* pooling runs from another sweep put all the range runs into the `p = 0` point.
  Colors use the validated categorical palette with a fixed slot per condition; no-ledger ablations reuse the base hue, dashed.
- **New dependency: matplotlib**, in an optional `[analysis]` extra. *Why:* paper figures only; the sim and tests never import it, and `test_plot.py` skips without it.
- **Open:** sampled static assignments are a weak headroom reference until C0 exists; the head-on rule picks a side at random rather than by who is closer to the goal.

### Condition definitions: layer model and RoCo-style C1 (Nov 2, 2026)

Canonical text: `conditions.md`. Prompted by the need to keep conditions from bleeding into each other and to map them onto real VLA/VLM S1-S2 robots.

- **Layer model.** Perception (belief graph), S1 skills (primitives), and task recipes (templates) are fixed for every condition. A condition is one choice each of S2 executor (`coded | llm`), allocator, utility source, and shared state (ledger or none).
  *Why:* G and `scripted` used the coded executor while C1-C4 use the LLM executor, so G vs C2 differed in two layers at once.
- **C1 is RoCo-style with no ledger** (`mandi2024roco`, read in full Nov 2). Who-does-what lives only in transcripts and the compacted history. There are no structured `claim`/`release`/`report` tools. The H2 ablations become C1+ledger and C3-noledger, a 2x2 of {dialogue, auction} x {ledger, none}.
  *Why:* this is the LLM-only system the literature actually builds (RoCo, CoELA), and the 2x2 still splits consensus from ledger.
- **C1 dialogue is lockstep per communication component** (the user's rule):
  - **Rounds:** a decision point opens a round among the online agents connected to the trigger. Turns are sequential, each costing `L` plus a hop. Replies end with PROCEED or EXECUTE (a joint plan, allowed after everyone has spoken). Limits are `M` turns and `K` re-plans.
  - **Delivery:** every message crosses the lossy channel per receiver, so partitioned or crashed agents and dropped packets never reach a member's view. `k` partitions run `k` rounds with `k` transcripts.
  - **Execution:** busy members keep executing while they talk (S1 runs while S2 deliberates).
  - *Why not fully lockstep:* RoCo's Algorithm 1 steps all robots together, which here would idle fast agents through 60-tick tasks. The dialogue stays lockstep; execution does not.
- **No notes tool for any condition.** Memory is history plus compaction only. *Why:* faithful to RoCo, and C1 vs C3 must not also differ in memory.
- **C3/C4 LLM scope:** read the ledger, output `u_ij`, and report done, failed, or release. Code bids, resolves, and enforces monotone status. *Why:* free ledger edits or direct bids would put allocation in the LLM and void CBBA's assumptions.
- **G and R use the LLM executor in paper runs** (coded versions stay as dev references). **`scripted` is dev only**, never in the paper.
- **C0 = best known plan**, from a heuristic solver with ground truth. It plans assignment and order (paths from the shared BFS), is computed per (mission, seed) at baseline with drop 0, and is not swept. Budgets move to `k x C0 makespan` once it exists. *Corrected Nov 5:* per (mission, seed), since spawns differ by seed. *Built Nov 6:* see the C0 entry below.
- **New planned metrics:** false completion claims (RoCo's false-done herding) and re-plan attempts.
- **Open:** the values of `M` and `K`; whether the coded vs llm executor contrast enters the run budget.

### Discovery-only tasks, shared exploration, uniform spawns, battery scaffold (Nov 4, 2026)

- **Agents start knowing no task, only the floor plan** (`task_visibility: discovered` for generated houses and paper runs).
  - **Reveal:** a mission task becomes known to an agent when it sees one of the task's reveal objects. These are `reveal_on`, or by default its movable goal objects. The first finder logs `discovered`, announces it with `task_found`, and keeps working if busy, so tasks are found mid-task.
  - **No duplicates:** ids stay the mission ids, and triggers never duplicate a mission task.
  - **Posted mode** remains for the hand-written kitchens and tests.
  - *Why:* the user's decision. It makes exploration, information sharing, and stale knowledge part of every condition's job, as for a real household fleet.
  - *Rejected:* writing a new trigger per task kind. Reusing the mission as ground truth keeps batching and the language notes unchanged.
- **Exploration is nearest-frontier, the standard baseline** (Yamauchi 1997; Burgard et al. 2005 for the coordinated variant; **not in `references.bib`, verify before citing**).
  - **Two S1 skills:** `explore` (team coverage, for finding tasks) and `search` (own coverage, for finding an object).
  - **Sharing:** coverage is shared every `share_every` ticks as `coverage` messages over the lossy channel, merged by latest sighting. It belongs to the perception layer, the same for all conditions, and never prompts a decision.
  - **Decay:** cells expire after `search_ttl` ticks when it is set (null for finite missions), for regenerating tasks later.
  - *Why two maps:* coverage says where a teammate looked, not what it saw. With one shared map, an agent needing an object a teammate had seen stopped searching and waited forever (house s1, seed 0). Belief gossip, planned with the ledger, would make `search` rarely needed.
  - *Why `search` restarts:* objects move. A dish gets carried off, or lands on floor seen earlier. *Refined Nov 9 (v5.1.00):* a search that has swept its round now ends with a decision point, and the next search starts the new round. Two C2 agents searching for a plate inside the dishwasher had swept for 550 ticks, since ledger and CBBA messages never interrupt a skill.
- **`task_found` is not given to C1** (structured id). C1 learns a teammate's find from dialogue.
- **Uniform spawns** (`spawn.mode: uniform`, `min_separation: 4` steps) come from a seeded stream per episode seed, the same for every condition. Docks stay fixed for parking and charging. The generator's budget is 2x the **slowest** greedy makespan over 3 spawns.
- **Coded executor fixes**, found by the house sweeps:
  - re-broadcast reports no longer advance the next task (a pre-existing bug that skipped a `close`);
  - an item already where the task wants it is skipped, not taken back out;
  - an agent holding a task's item does not leave that task when a teammate reports it done.

  The last two were G claim races turning into permanent failures.
- **Battery model and scheduled disconnect:**
  - designed in `simulator.md`;
  - the config (`battery:`) is parsed, and `enabled: true` is rejected until it is built;
  - scheduled disconnect is the announced counterpart of crash churn.
- **Continuous arrivals** are planned only (Poisson rate, as in Liu et al. 2013).
- **Sweep result** (v2.0.00, generated houses, discovered tasks, uniform spawns, drop 0.025, 40 seeds each):
  - **G:** 120/120 finished.
  - **Scripted:** 118/120. The two failures are an entry-table gridlock in house s1 under the fixed round-robin assignment; scripted is dev only.
  - **Kitchen in posted mode:** scripted and G finish 40/40 at drop 0, 0.025, and 0.2.
- **Cost:** a house episode takes about 10 s of wall time, up from about 3 s, mostly exploration bookkeeping.
- **Open:**
  - belief gossip timing;
  - whether the discovered-mode budget should move to C0;
  - optimizing exploration if the sweeps need it.

### C0 oracle: heuristic plus CP-SAT, replayed in the sim (Nov 6, 2026)

- **C0 is the best known plan with a proven bound in its planning model.** A seeded heuristic (insertion, relocate, swap) seeds OR-Tools CP-SAT. The denominator for every condition is the plan's **replayed** makespan per (mission, seed). Each run logs the bound B, planned value V, and replay R to `oracle_plan.json`.
  *Why CP-SAT:* the user chose the option most defensible to reviewers. A heuristic alone cannot answer "how do you know your oracle is good?"
- **New dependency: `ortools`, in an optional `[solver]` extra.** *Why:* the only route to a proven bound. The sim and the heuristic never import it.
- **Reproducibility:** limits are deterministic time and search counts, with interleaved search, so results do not depend on wall-clock speed. Checked: repeat solves are identical on houses s0 and s2, and 1 vs 8 workers are identical on s1.
- **The plan carries release ticks.** Agent task lists do not fix the order on shared furniture, and re-timing them lost the solver's order (one plan went from 418 to 605). An agent does not begin a task before its release tick.
- **Model corrections, found by comparing replays with plans:**
  - A task completes when its last goal-making step ends, not after the walk back to unlock. Replays were beating "proven optimal" plans.
  - Loading trips hold the dishwasher, since it cannot be opened mid-cycle. Replays were up to 118 ticks late.
  - Result: every replay now lands at or above its plan. Trips are exclusive against each other too, which is slightly stricter than the sim, where two agents can load at once.
- **Redundant capacity constraint per furniture** (first possible hold + total hold - overhang <= makespan). It cut the gaps from about 44% to 25% on s0 and about 71% to 39% on s2.
- **Results (deterministic limit 60, 5 seeds per house):**

  | House | Mean bound B | Mean planned V | Gap (V-B)/B | Mean replay R | R over V | G at drop 0 | G / R |
  |---|---|---|---|---|---|---|---|
  | s0 | 345 | 440 | 25-31% | 473 | +8% | 656 | 1.39 |
  | s1 | 451 | 451 | 0% (OPTIMAL) | 477 | +6% | 773 | 1.62 |
  | s2 | 337 | 463 | 36-39% | 512 | +11% | 716 | 1.40 |

  Four times the solver time (240) moved the gap only about 2 points and did not improve the plan, so the limit stays at 60. Solves took about 10 s on s1 and about 50 s on s0 and s2. Bounds on s0 and s2 are loose; the optimum could lie anywhere between B and V.
- **Budgets moved off greedy:** `k x` the slowest heuristic-plan replay over 3 spawns, with `k = 4`, giving 1950/1950/2100 ticks. *Why 4:* G at drop 0.025 takes about 1.5x the replay on average, and `k = 4` leaves room for its slow seeds without censoring a competitor by design.
- **Generator bug fixed (present since Oct 28):** `_nearest` measured from a furniture cell, which has no distance, so it silently returned the first option. Every wash went to `dishwasher_1` and every pack to `pantry_1`. All earlier house numbers carry that artificial bottleneck.
- **Two movement livelocks fixed (all conditions):**
  - A detour that first moves away from the target stepped straight back onto the static path. A detour now continues until the static distance improves.
  - Two speed-2 movers detouring in step undid each other's moves every tick, with no blocker to name. Movers stuck for 2 ticks now sit out each round on a seeded coin flip.
- **Caveats for reading gaps:**
  - The gap to C0 measures allocation *and* information: C0 knows every task and object, while paper conditions discover them.
  - Serial use of shared furniture (wash cycles in particular) sets a floor no allocator can beat. `dish_batch_size: 2` means 4 washes on one machine in house s0; raising it would make allocation matter more.
  - `rho` constraints are not in the planning model yet.
- **Regression (final code, 40 seeds, drop 0.025):** G finishes 118/120 (mean 666 / 870 / 711), scripted 119/120. G's failures are coordination failures: a false completion (s2 seed 24) and a claim race that ends with bodies blocked at a trashcan (s0 seed 34, not fully traced). Kitchen in posted mode: all 240 finish.
- **Correction (Nov 5):** the single-seed "loss is slower" test failure came from delivered messages costing re-decisions (`L` ticks each), not movement luck, as first stated. Curves against `p` may therefore not be monotonic.
- **Open, all resolved Nov 7:** `dish_batch_size` gave way to dirty-dish clusters; `rho` stays out of C0; the CP-SAT citations are confirmed.

### Drops are a time penalty (Nov 7, 2026)

*Supersedes the Oct 25 rules on re-picking, shatter voiding, and discovered cleanup.*

- **The purpose of drops is realistic penalty and variance, not a coordination subsystem** (the user's decision). The old rules voided a whole batch on one break and left the other dishes unwashed, so faults shortened missions.
- **Non-fragile:** a 2-tick pause. The item stays in hand, with no failure and no decision, so no LLM cost.
- **Fragile:** a `clean_up` reflex in every condition, with no LLM call: 3 ticks to pick up the shards, a walk to the nearest trashcan, 1 tick to dispose, then back to the same task. *Why a reflex:* the task must not go up for grabs, and every condition must pay the same cost (hard rule 7).
- **Only the broken item is lost.** A task with no item left is voided. The discovered `cleanup` task and its template are gone.
- **Crash while holding something fragile:** the shards stay on the floor as debris with no task, since nobody is there to run the reflex.
- **Triggers skip broken objects.** An agent that has not seen a dish break may still believe in it, and a task for it could never finish. This relies on ground truth in the same way as the perfect-dedupe premise; it only suppresses a task and gives the agent no information.
- **Result (40 seeds each, mean makespan at drop 0 vs 0.025):**
  - **Houses, G:** 672 vs 674, 877 vs 866, 737 vs 737. Flat at the paper's drop rate, which is the goal.
  - **Kitchen, scripted / G:** 298 vs 294, and 303 vs 302.
- **Not fully solved at high rates:** at drop 0.2 the kitchen still falls (scripted 245, G 261). Its tasks wash one plate each, so a break voids the task and skips its 60-tick wash. A generated cluster that loses every dish does the same. Paper runs use 0.025. Read makespan with `voided`, and if `drop_prob` is ever swept, expect this effect.
- **Regression (drop 0.025):** G 119/120, scripted 120/120. G's failure (s0 seed 34) also happens at drop 0, so drops do not cause it.

### Dirty-dish clusters, C0's scope, reflex refinements, run layout (Nov 7, 2026)

*Supersedes `dish_batch_size` from Oct 28.*

- **Dirty dishes come in clusters** (the user's rule). The generator places `dirty_clusters` (3) clusters of `cluster_tiles` (1-3) adjacent surface tiles, each tile a stack of `stack_size` (1-5) dishes. One cluster is one `clean_dishes` task: all its dishes are loaded, then washed together in one cycle. Clusters may not touch, because adjacent means same task.
  - *Why:* a batch of 2 split dishes by an arbitrary number and forced serial 60-tick cycles, a floor no allocator could beat, though a dishwasher has no capacity limit.
- **C0 has no search cost and no `rho`** (the user's decisions). It is the perfect case, and C1-C4 should get close without matching it, so a condition's search cost belongs in its gap. No posted-task variants are planned. With `rho` out, H1's makespan gap could stay flat or widen; DOE open question 0 proposes the violation rate as H1's primary measure. H1's wording stays.
- **The reflex takes the agent back** (the user's refinements). `nearest_trashcan` reads ground truth, so no search or scene-graph bug can arise from a fragile drop. `return_to_task` then walks back to the drop tile before any decision. A non-fragile drop is a same-tile drop and re-pick by the same gripper; timing is unchanged.
- **Result, C0 (5 seeds per house, CP-SAT at deterministic limit 60):** all 15 runs are proven optimal. Mean planned / replay: 400/448, 453/489, 688/747. G at drop 0 takes 1.74 / 1.87 / 1.58x the replay.
- **Do not read "optimal" as an easier search.** One chain now sets the makespan. Carrying is one dish per trip, so the longest wash task (many dishes, or a far surface) takes 50-67% of the planned makespan even for a fast carrier, and 1.4-1.7x as long for an arm (s2: 402 vs 676 ticks). The plan gives these jobs to the two carriers. The solver can prove optimality because the bound meets this chain. Makespan may therefore measure mostly who gets the long job. Stack carrying, or keeping dirty dishes near the kitchens, would ease this; neither is built (DOE open question 0a).
- **Dev figures and experiment layout.** `experiments/dev_refresh.py` reruns a small sweep per environment from `configs/dev/<env>.yaml` into `runs/dev/<env>/{logs,figures}`, deleting only that folder, so fixed file names always show the latest code. Named experiments go to `runs/<exp>/<env>/<cell>/<YYYYMMDD-HHMMSS>/seed_<n>/` through `run_single --run-name`, and never overwrite each other. The stamp is wall-clock, used only as a folder name (hard rule 1).
- **Citations:** `perron2023cpsat` and `perron2026cpsat` are in `references.bib`, not yet cited. The paper is a two-page invited-talk abstract, so cite it for what CP-SAT is, not for performance.
- **Regression (drop 0.025, 40 seeds):** G 120/120 (means 849 / 1059 / 1218); scripted 119/120 (752 / 1039 / 1086; the failure is s1 seed 24, not traced). Scripted now beats G on all three houses, the reverse of v3.0.00, probably because its fixed assignment happens to suit the long washes; it is dev only, so this is not a result. Kitchen: 240/240, scripted 298 / 294 / 248 and G 303 / 303 / 264 at drop 0 / 0.025 / 0.2.

### Sweep runner, a traced failure, standby, log size (Nov 8, 2026)

- **One command runs an experiment** (`experiments/sweep.py`, specs in `configs/sweeps/`). It builds cells as the dev refresh does (baseline, then one factor at a time; C0 at baseline only), runs them in parallel under one stamp, and writes metrics and a per-(environment, cell) summary. *Why:* the working agreement wants experiment settings in config files, and the scratch scripts were not. A failed episode is logged and skipped so it cannot lose a long sweep; the exit code still reports it.
- **Scripted s1 seed 24 traced:** a duplicated multi-item task in the coded executor, not a sim bug.
  - `carrier_e` owns `t_wash_dishwasher_2_1` by the fixed assignment, which is never claimed. `carrier_f`, idle, adopts the same task at tick 373, because the stand-in's adoption rule only checks for a claim.
  - `carrier_f` always targets the dish `carrier_e` is carrying. The "never take a loaded item back out" check runs when the calls are chosen, and the dish is in a gripper then, not in the dishwasher. When the queued `pick` runs, the dish is in the open dishwasher, and the ground-truth precondition (reachable, not held) passes. So `carrier_f` takes out each dish just loaded and puts it back.
  - At tick 515 it is holding `cup_25` when `carrier_e` closes and starts the machine. `carrier_e` then reports the task done, a false completion, since the ground-truth goal fails while `cup_25` is outside.
  - `carrier_f` puts `cup_25` back in at tick 595 and, having heard the report, leaves. Nobody runs another cycle, and the episode hits the budget.
  - *Fixed in v4.0.00:* see the next entry.
- **The 32-minute refresh was Modern Standby.** The System log shows Kernel-Power 506 (entering) at 02:38:55, 30 s after the refresh started, and 507 (exiting) at 03:10:18. Every process was frozen in between. Ticks are unaffected (hard rule 1), but a long sweep would stall, so sweeps and dev refreshes now hold a Windows "system required" power request. It blocks idle sleep only; closing the lid still sleeps.
- **What the logs hold, and where the bytes go** (house run, greedy, every-tick snapshots, 77 MB): `beliefs` 70 MB, of which full per-agent belief snapshots (one per agent per state snapshot) are 99.8%; `state` 6 MB (full world snapshots); `llm_calls` 3 MB (full prompts); `messages` 0.6 MB; `actions` 0.2 MB; `events` 0.04 MB. At the default snapshot every 10 ticks, a house run is 7-17 MB. Proposed, not built: log each belief change with its content instead of periodic full belief snapshots, write a small per-tick agent-pose row with sparse full-state keyframes, and gzip the streams. That is roughly 1 MB per house run before compression, with nothing hard rule 9 needs lost.

### Shared dishwashers, pick cancels, compact logs (Nov 8, 2026, v4.0.00)

The user's rules, from the s1 seed 24 trace:
- **Wash goals are per dish, `washed` in any dishwasher**, instead of "in this machine and its cycle done". *Why:* an agent facing a busy machine must be able to use another one.
- **A cycle started mid-load splits the task** (a simulator rule in every condition, the user's choice). The loaded dishes finish with that cycle; the rest become a new task, announced as found, so the team's allocator (CBBA, MURDOCH, or dialogue in C1) picks it up. A dish in hand goes back to its pile by a no-LLM reflex. *Why:* it is nobody's fault, and a clean break gives the ledger a discrete task instead of a half-done one.
- **No idling at a busy dishwasher** (an executor rule, the user's choice; LLM agents decide for themselves). The countdown is perceivable, so a coded agent knows when a cycle ends without looking again. With nothing loaded, it moves the wash to another dishwasher believed free; otherwise it puts the dish back, gives the task up, and does not take a wash on while its machine is busy.
- **A queued pick is cancelled once perception shows the item moved** (the user's refinement), and the pick re-checks ground truth. *Why:* the race the trace showed, walking to a dish a teammate then loads, is closed at sight rather than on arrival.
- **Coded executor fixes:** no adopting a teammate's fixed assignment (scripted), no targeting an item a teammate holds, no call for a door already believed in the wanted state (it was 1,045 failed `open` calls in 60 G pilot runs). A task given up comes off the agent's list, so it can be taken again once the obstacle clears.

Found while verifying, and fixed:
- **Stale "holding" beliefs.** Shards dropped into a closed trashcan left the agent believing it still held them, so it never searched (hands believed full). Agents now know their own hands (`sync_hands`, proprioception).
- **Search could stall for good.** The coverage map counted only floor; a floor cell seen at the edge of view range hid the shelf beyond it, so an agent's map could be complete while a dish it needed sat unseen. A floor cell next to unseen furniture now stays a frontier. This changes exploration for every condition.
- **C0's replay drifted into splits.** Its planning model let another task's cycle fit between one wash's loading trips. A wash now holds its machine from its first loading trip to the end of its cycle, and the replay waits for the wash planned before it on the same machine, and waits out a busy machine rather than rerouting.
- **Kept at the ledger's level of truth:** a coded agent reads its current task back from the shared task list each decision, so it sees a split or a break.
- **Coded executor races found by the regressions:**
  - a single-dish wash whose dish broke has no goals left, and was restarted forever; a task whose goals mention nothing it picks up now counts as over;
  - a pick that failed because its dish moved sent the agent searching the house; an adopted task is now handed back (and comes off its list, so it can be retaken once the dish is seen);
  - a pick in progress when the split happened finished for the old task, leaving the dish in hand for good; a split now stops running and queued picks of the split-off dishes;
  - a queued put-back could be cut short by news; the executor now asks for owed put-backs again until the dish is down.

Logs (the user's choice of all three proposals): gzipped streams, belief changes carrying the new belief (with full graphs only at the start and end), a per-tick `poses` stream, and full state keyframes every 50 ticks. An unchanged sighting refreshes `last_seen` without a row; every prompt shows it.

- **Results (houses s0/s1/s2, budgets unchanged at 1900/2150/3050):**
  - **C0** (5 seeds per house, CP-SAT): 15/15 proven optimal, mean planned / replay 400/448, 453/476, 688/761, and no split in any replay.
  - **G, drop 0.025, 40 seeds:** 120/120, mean 877 / 1147 / 1227 (v3.1.00: 849 / 1059 / 1218). The new costs are visible: about one split per run, held dishes carried back, and agents walking to the other kitchen's dishwasher. Duplicated wash work remains (claim races, G's nature); failed `open` calls fell from about 17 to 0.3 per run.
  - **Scripted, drop 0.025, 40 seeds:** 120/120, mean 1120 / 1160 / 1435 (v3.1.00: 752 / 1039 / 1086). Slower by design: teammates no longer take over each other's fixed assignments, so the most loaded agent sets the finish. Its v3.1.00 speed came from duplicated work.
  - **Kitchen** (`m2_home`, one dishwasher, 40 seeds, drop 0 / 0.025 / 0.2): all 240 finish. Scripted 344 / 303 / 200, G 344 / 324 / 243 (v3.1.00: 298 / 294 / 248 and 303 / 303 / 264). With a single dishwasher, the put-back rule costs trips that waiting would not. A countdown threshold (wait when the cycle ends sooner than a round trip) would recover this; not built, the user's rule is literal for now.
  - **Logs:** 0.34-0.59 MB per house run (from 7-17 MB). The coded pilot rerun (720 episodes, no failures) wrote 360 MB instead of 15 GB; G takes 1.99 / 2.39 / 1.64x C0's replay at baseline.

### Washed-dish unloads, waiting at a busy dishwasher, C0 plans unloads (Nov 8, 2026, v5.0.00)

- **Washed dishes become an unload task when seen** (the user's rule). Before, they stayed in the machine and were washed again. An agent that opens a dishwasher and sees clean dishes announces `unload_<dishwasher>_<first dish>` (a follow-up kind, so the wash that covered the dish does not hide it). It then minds its own load.
- **The unload keeps its lock** (the user's choice over overlapping load and unload). A loader that arrives during an unload waits until the dishes are out, which is accepted idling in every condition. An unload not yet begun cannot stop a cycle; then the dishes are washed again and the unload waits for the cycle.
- **Wait if the cycle ends soon** (the user's request). The coded executor waits out a busy machine when its countdown ends sooner than the cheapest way out (walk to another free dishwasher, or round trip to put the dish back), and also when it has nothing else to do.
  - *Correction:* the v4.0.00 kitchen slowdown was first blamed on the put-back rule. The put-back itself is the cost, but the "soon" comparison rarely fires: a round trip looks cheaper than most waits. What helps is "nothing else to do": a put-back buys time only for other work. Measured, the "soon" part alone is neutral (within 2% on houses, none in the kitchen); "nothing else to do" saves about 3% in the kitchen.
- **C0 plans unloads** (the user's choice). Between consecutive washes on a shared machine, an unload of the earlier batch by any agent gates the next wash's first hold. Every order of those washes is solved and the best kept; the bound is the minimum over orders. The unload's model walks to the dishwasher, not to the dishes' starting pile. Inserting tasks predecessors-first keeps the heuristic from fixing two washes' order the wrong way round before their unload exists. The replay's sightings find only planned unloads.
- **Old pilot logs deleted** (approved): the v3.2.00 pilot's 15 GB stamp.

- **Results** (budgets regenerated to 2100 / 2450 / 3500, since C0 now plans unloads):
  - **C0** (5 seeds per house, CP-SAT): 15/15 proven optimal, mean planned / replay 445/503, 519/537, 806/865. One planned unload per house, every unload in a replay a planned one, no splits.
  - **G, drop 0.025, 40 seeds:** 119/120, mean 1001 / 1460 / 1604, about one unload per run. G takes 1.99 / 2.72 / 1.85x C0's replay. Its failure (s1 seed 16) is a claim race: four idle agents claimed the same unload on one tick, and five bodies jammed around a corner dishwasher until the budget ran out.
  - **Scripted, drop 0.025, 40 seeds:** 120/120, mean 1140 / 1318 / 1576.
  - **Kitchen, 40 seeds, drop 0 / 0.025 / 0.2:** all 240 finish. Scripted 407 / 383 / 252, G 449 / 417 / 294: two unloads and some cupboard overflow (store tasks) are new work. C0 plans the kitchen in 268 ticks and replays it in 310.

### LLM layer, ledger, CBBA, and C1-C3 built (Nov 8, 2026, v5.1.00)

The plan the user approved ("integrate LLMs and implement C1-C3"), with their choices: llama.cpp with Qwen3-8B (WSL2 only if Windows lacked something; it did not), CBBA score = suitability x travel discount, `rho` paused, deterministic history compaction.

- **Serving (hard rule 6):** one `llama-server` (release b12211, native Windows CUDA build), `Qwen3-8B-Q4_K_M`, 6 parallel slots, each agent pinned to its own slot so its prompt prefix stays cached. Setup and flags: `llm_setup.md`.
  - *Memory:* per-slot KV buffers (36,864, then 30,720 tokens) ran out of CUDA memory mid-episode on the 8 GB card. One unified 16k buffer (`--kv-unified`) uses 5.9 GB.
  - *Then a third crash with VRAM flat:* all three crashes came while heavy CPU jobs ran beside the server, with its host prompt cache at the 8 GB default. The likely cause is host memory pressure under the Windows WDDM driver (not proven). The host cache is now capped at 2 GB (`--cache-ram 2048`), and heavy jobs no longer run beside live pilots. No crash since.
  - *Sleep:* `run_single` holds no keep-awake request, and one pilot slept for four hours mid-episode. Wall time is a cost only, so results are unaffected, but `llm_wall_seconds` for that run is meaningless.
- **Client** (`llm/client.py`): the OpenAI-compatible chat API over stdlib `urllib`, so no new dependency. Settings:
  - **Replies:** JSON-schema-constrained, Qwen3 thinking off (`enable_thinking: false`), temperature 0, seed 0.
  - **Cache** (`llm/cache.py`): every reply is stored by sha256 of the model hash and the request body. Reruns replay identically, and `replay_only` re-scores a run without a GPU.
  - **Provenance:** the model file's sha256 goes in `run.json`.
- **One seam (hard rule 7):** `LLMPolicy` = an allocator plus the LLM executor. The allocators are G, CBBA (C2, C3), and dialogue (C1). The coded policies are untouched, and the coded kitchen regression is unchanged (scripted 382.75, G 417.05 over 40 seeds, as in v5.0.00), so this is a MINOR version.
- **Loop hooks (sim core):**
  - **Coordination step.** Each tick every LLM policy runs `on_tick` in code. Messages of kind `ledger`, `cbba`, and `dialogue` are read there and never make a decision point, like `coverage`. A change of assignment does (hard rule 2).
  - **Concurrent decisions.** Decisions due on one tick are sent concurrently and applied in agent order, so thread timing never changes an outcome.
  - **Several calls per decision.** A decision may make several calls (C3's utility query, then the action), and their latencies add. Each call is an `llm_calls` row with a `purpose` (`act`, `utility`, `dialogue`).
- **Executor prompt** (`llm/prompts.py`, `configs/prompts/`):
  - **System prompt:** role, rules, and the condition's tools.
  - **Observation:** the belief graph, capability card, current task with its recipe, the allocator's lines (bundle and ledger rows, or the last plan), and the compacted history. History keeps the last `history_keep` (8) actions verbatim, plus one code-written line counting the earlier ones, failures, and reports.
  - **Reply:** `{thought <= 200 chars, calls <= 3}`. A reply that does not parse is an `invalid_call` that costs `L`.
  - **Body-dependent tools:** a tool the agent's body cannot use is not offered (no `scan` without a scanner).
  - **History shows `ok` or `FAILED` plus the reason**, not the analysis label. The model read "stale_belief" as "my belief is wrong" and tried to unlock what it had failed to lock.
- **Executor-layer rules from the live pilots** (every LLM condition; `conditions.md` §1). Each gives the LLM executor what the coded executor already knows or does:
  - **Locks belong to their task:** code unlocks, every decision, any workspace the current task's recipe does not lock.
  - **Voided tasks are flagged:** a task whose items have all shattered is shown as over.
  - **Task tools name the agent's own task:** a pilot agent reported a task called "none".
  - **G retakes a released task** after 120 ticks.
  - **G's prompt lists the claims and reports it has heard.**
  - *Why not leave these to the LLM:* each failure was the LLM missing information or a guarantee the coded executor has. Leaving them in would confound coded vs LLM (and G vs CBBA) with information, not reasoning.
- **Ledger (M2, `sim/ledger.py`):** rows `{status, owner, bid, version, lease_expiry}`.
  - **Merge:** done is absorbing; otherwise higher status, then version, then bid, then the lower agent id wins. The merge is commutative, associative, and idempotent (tested).
  - **Gossip:** a broadcast on change, plus anti-entropy every `gossip_every` (10) ticks.
  - **Leases:** 60 ticks, renewed every 20. A teammate's lapsed lease reopens the task.
  - **Logging:** merged rows are not re-broadcast (no echo), but they are logged, so the `ledgers` stream holds every change to every copy.
- **CBBA (M3, `allocation/cbba.py`):**
  - **Bid:** `c_ij = s_ij * 0.97^steps_ij`, independent of the rest of the bundle. Bundles hold up to 3 tasks.
  - **Consensus:** Choi et al.'s update table, including the outbid cascade (an outbid task drops every task added after it).
  - **Commitment:** the head task's bid becomes `COMMITTED_BID` once work starts, so nobody outbids work under way. A task given up is not re-bid by that agent for `give_up_ticks` (120).
  - **Suitability:** C2's `s_ij` is 1 for a capable agent (hands, scanner), else 0. C3's is one batched LLM utility call over newly known tasks, scores 0-100 divided by 100, cached per (agent, task); a score the reply omits counts 0.5.
- **The ledger tools are `mark_done` and `give_up`.** `conditions.md` listed "report done, report failed, release"; `give_up` covers both failed and release. *Why:* one name per ledger operation, kept apart from G's `report`/`release` messages, which carry no ledger.
- **C1 (`allocation/dialogue.py`)**, following `conditions.md` with these values and refinements:
  - **Rounds:** members are the online agents within `comm_range` of the trigger, transitively. The trigger speaks first, then the others by id. Limits are `M` = 2 x members turns and `K` = 2 re-plans. Messages are at most 400 characters.
  - **Code checks:** members, known tasks, and that everyone spoke before EXECUTE. Capability and hands are not checked yet.
  - **Adoption:** only when the EXECUTE message reached the member. A WAIT asks again after `retry_ticks` (30).
  - **Without a task:** an agent holds still (`wait 5`, with no LLM call) until a plan arrives, unless it holds a workspace, which it must still unlock.
  - **`talk_to_team`:** the agent's tool to ask for a round when it has finished or is stuck. It stands in for `report`/`release`, which C1 lacks.
  - **No `task_found`:** these messages are silenced in C1 by the channel (`Network(silenced=...)`).
- **Test double** (`tests/fake_llm.py`): answers each act call through the coded executor, gives every utility 100, and in dialogue proposes open tasks round-robin. It reads ground truth, which only a test double may do. With it, the whole pipeline runs without a GPU. Kitchen, seed 0: coded G 312, C2 389, C3 351, C1 439 ticks, all finished.
- **New responses:** LLM calls by purpose, parse failures, rejected responses (re-plans), C1 plan rejections, false completion claims, final ledger lag, and messages by kind (`experiments/analyze.py`).
  - *Correction while building:* a "claim before ground-truth done" rule counted coded G's report at a wash start as false, though the cycle finishes the task. A claim is now false only if its task still needed work.
- **Open:** prompt iteration from the live pilots; calibrating the latency rates (open question 2); wall time per house episode; whether C1's plan check should cover capability.

- **Live kitchen pilots** (Qwen3-8B, `m2_home`, posted tasks, seed 0, 900-tick budget, final v5.1.00 code; `runs/pilot_llm_c/`). One seed each, so these are not results. They show that every condition runs live and what fails.

  | Condition | Tasks done | LLM calls | Tokens per call | Rejected replies | False completions | Duplicated ticks | Final ledger lag |
  |---|---|---|---|---|---|---|---|
  | G-llm | 5/8 | 369 | 1,642 | 28 | 2 | 47 | none (no ledger) |
  | C2 | 4/7 | 329 | 1,690 | 15 | 1 | 0 | 0.0 |
  | C3 | 7/8 | 283 (11 utility) | 1,586 | 8 | 0 | 0 | 0.0 |
  | C1 | 4/10 | 572 (114 dialogue) | 827 | 16, plus 1 plan | 0 | 106 | none (no ledger) |

  - **No run finished within 900 ticks;** coded G finishes this kitchen in about 417. Task counts differ because unloads are found during the run.
  - **Consensus shows up even here.** C2 and C3 duplicated no work, and every ledger copy ended in agreement. C1's rounds gave the same plate to two arms (106 duplicated ticks), the failure H2 predicts. C1 held 26 rounds and 21 EXECUTE plans.
  - **What the model gets wrong:**
    - locking or opening before walking next to the furniture;
    - unlocking what a teammate holds;
    - searching for objects inside closed containers;
    - releasing or reporting when blocked.

    The prompt rules cut these but did not remove them.
  - **Cost:** 7-34 minutes of LLM wall time per kitchen episode. A house has twice the agents and 2-4x the ticks, so expect roughly 1-4 hours per house episode on the laptop. The 24-episode `pilot_llm.yaml` would take one to three days, which argues for the cluster.
  - **Fixed along the way:**
    - Server memory: see the serving notes above.
    - Executor-layer rules: see above.
    - A search that has swept its round now ends (the Nov 4 entry).
    - The `ledgers` stream now logs merged rows.
    - The history no longer shows outcome labels.
  - **Coded regression after these fixes:** kitchen 383 / 417, as before. Houses: G 1001 / 1460 / 1606 (the s1 seed 16 gridlock remains) and scripted 1140 / 1318 / 1576, within 2 ticks of v5.0.00.

### Model candidates, a blocked bake-off, metrics for the paper (Nov 9, 2026, v5.1.01)

- **Which model is still open** (open question 3). `llm_models.md` lists 17 candidates; Apache-2.0 GGUFs were checked for Qwen3.5-4B, Qwen3.5-9B, and Ministral 3 8B. The bake-off pits Qwen3.5-4B (efficiency) and Ministral 3 8B (a second family at Qwen3-8B's size) against the Qwen3-8B pilots, in the kitchen with C3 and G-llm.
  - *Why speed is not the criterion:* ticks come from tokens at fixed rates (hard rule 3), so a faster model only saves wall time. The model's behavior is what changes results, so one model is frozen for all conditions.
  - *Why tool-calling scores are only a proxy:* replies are grammar-constrained JSON, not native tool calls.
- **The bake-off did not run.** Every server, including Qwen3.5-4B at 3.1 GB of VRAM, crashed with CUDA "out of memory". The laptop had 3.6 GB of RAM free after 11 days of uptime, with `dwm.exe` holding 5 GB of shared GPU memory, so the cause is host memory, not VRAM (`llm_setup.md`). Next step: restart Windows and rerun (procedure in `llm_models.md`). The server now runs with no host prompt cache (`--cache-ram 0`).
- **Paper-facing rates in the analyzer:** `false_completion_rate` (per distinct claim), `replan_rate` (per act call), and `consensus_messages_per_task`. `DOE.md` §3 now has a metric reference table (formula, source stream, unit, use, caveats), and `outline.md` points to it.

### Bake-off, thinking probe, grammar-enforced arity (Nov 10, 2026, v5.2.00)

- **The restart fixed the crashes.** After the Windows restart (19.5 GB free, `dwm.exe` at 0.25 GB shared GPU memory), ten live kitchen episodes across three models ran with no out-of-memory error. The shared DWM memory still crept to 0.57 GB over about ten hours. *Also found:* a long-running probe without a keep-awake request slept for hours (the same Modern Standby as Nov 8); every live script now holds one.
- **Bake-off result** (`llm_models.md`, C3, kitchen, seed 0): Ministral 3 8B 8/11 tasks, Qwen3-8B 7/10, Qwen3.5-4B 3/6, Qwen3.5-4B with thinking 1/6. Model choice (open question 3) stays open: one seed in one kitchen cannot decide it. Ministral 3 8B and Qwen3-8B go to the house pilot; Qwen3.5-4B is the fast fallback.
- **Reasoning (thinking) mode is not used.** It costs about 1,000 output tokens per decision, so latency `L` rises from about 3 to 27 ticks (hard rule 3) and the team makes a third as many decisions. It works under the JSON grammar and cut stale-belief errors from 0.26 to 0.02, but it finished fewer tasks. `llm.thinking` (default false) stays as a config switch for a future cluster ablation.
- **Reply schema fixes arity** (every LLM condition). `anyOf` over one object per tool, each with exactly its number of arguments and whole ticks for `wait`. Qwen3.5-4B had 159 of 434 replies refused for an empty or whitespace-filled `args`. *Why in the grammar, not the prompt:* the prompt example and a clearer error message did not help; the model repeated the call. A refused call's message also quotes the tool's usage line.
- **Same-code rule for comparisons.** The pilots of Nov 9 (Qwen3-8B) predate the schema change, so only `runs/final/` compares models.

### Hypotheses revised, `rho` removed, scale sweep proposed (Nov 10, 2026, v5.3.00)

- **New hypotheses (exactly three; no threshold hypothesis).**
  - **H1 (structure):** as the number of tasks `m`, task difficulty, and team size `n` grow, LLM-only teams degrade through duplicated, orphaned, and conflicting work, while auction-based teams do not.
  - **H2 (reasoning):** LLM utilities achieve higher utility relative to the oracle than numeric utilities, at a measurable token cost.
  - **H3 (resilience):** under packet loss `p`, range `r`, and churn `k`, LLM-utility auctions degrade less and recover faster than numeric auctions, beyond what the replicated ledger alone provides (the ledger ablation tests this). Resilient or emergent behavior from LLM reasoning is hoped for, and only claimed if the data show it.
  - *Why:* the study's question became "does structure keep LLM teams coordinated as missions scale, does LLM context beat a hand-designed score, and does it hold under degraded links". The packet-loss sweep stays as a swept variable with no separate threshold claim, so the old H3 (connectivity threshold, `r*`) is dropped.
- **`rho` and language-only constraints are removed from the project.** The LLM's value is context-aware utilities, not reading constraints that exist only in text. Removed: the generator's `rho` setting and its notes and hidden attributes, `language_only` and `note` on tasks, the `rho` field in `metrics.json`, and `fig_h1_rho`. The three houses were regenerated: maps, episode configs, and tick budgets are byte-identical, and only the `rho` notes, hidden attributes, and the `rho` key left the mission files. The generic `hidden_attrs` and `scan` mechanism stays. DOE open question 0 (H1's measure under `rho`) is void.
- **Figures renamed to the new hypotheses.** `fig_h2_loss` is now `fig_h3_loss`; `fig_h3_range` keeps its name (both are H3, resilience). The H1 scale figure and the H2 table join with the scale sweep.
- **Mission-scale sweep, proposed levels (Draft 1; adjust for hardware).** One factor at a time around the baseline (the DOE's rule), with the house area scaled with `m` to hold task density:

  | Factor | Levels | Generator knobs |
  |---|---|---|
  | Tasks `m` | about 11, **22**, about 44 (exact counts measured after generation) | `tidy_items`, `dirty_clusters`, `groceries_per_entry`, `scraps_per_trashcan`; house 32x24, **48x32**, 64x48; `rooms` 5, **7**, 10 |
  | Team size `n` | 3, **6**, 12 | `agents` list (mix kept: about a third two-handed, a third one-handed arms, a third fast carriers, one scanner) |
  | Difficulty | easy, **medium**, hard | easy: `cluster_tiles` [1, 1], `stack_size` [1, 2]; medium: [1, 3] and [1, 5]; hard: [3, 4] and [3, 5] |

  - **Difficulty means** shared-appliance coupling and per-task handling load: how long the wash tasks are and how long each holds an exclusive dishwasher. It keeps `m` fixed. *Risk:* DOE open question 0a (one long wash task sets the makespan) gets worse on hard, so read hard together with that question.
  - **Hardware limit:** `--parallel 6` gives one server slot per agent, so `n = 12` needs 12 slots (cluster, or a smaller context). `m = 44` and 12 agents also raise every episode's decision count.

### House LLM pilot; sweeps move to the GPU cluster (Nov 14, 2026, v5.3.02 and v5.4.00)

- **House pilot** (Qwen3-8B, laptop, `DOE.md` §7): 21 of 24 episodes ran in 28 hours, about 70 minutes each. 3 stopped when the server's shared 16k context overflowed (house prompts reach 4.4k tokens; the server now runs at 32k). The laptop also leaked about 1 GB of RAM per hour through zombie processes (`llm_setup.md`).
- **Paper sweeps run on the university GPU cluster** (`cluster.md`). *Why:* about 400 LLM episodes at 70 minutes is about 20 days of laptop time, with the paper due Dec 17.
  - **The container holds the runtime, not the code:** llama.cpp's official CUDA server image at b12211 (the laptop's build) plus Python and PyYAML. The code is a git checkout mounted at `/work`, so code fixes need no rebuild and `run.json` keeps recording the commit.
  - **One Slurm array task runs one GPU, one server, and one shard of a sweep.** Each episode pins its agents to slots 0-5, so episodes cannot share a server. Shards share one stamp, so `analyze --stamp` reads them as one launch.
  - **The same model file** (sha256 checked on download) keeps `run.json` hashes and response cache keys equal across machines.
- **Comparability rule:** simulated time never depends on the GPU (hard rules 1 and 3), but replies can differ across GPUs and llama.cpp builds even at temperature 0. A comparison never mixes laptop and cluster runs. Paper runs all come from the cluster, and the house pilot is rerun there, which also measures how far the laptop's 21 episodes move on different hardware.
- **Sweep specs gained** an optional `llm:` block (replaces fields of every episode's `llm`, such as `model_path: /models/...`) and an optional `only:` list (rerun named episodes). The runner takes `--shard`, `--of`, `--stamp`, and `--endpoint`; these are infrastructure, not experiment settings, and never reach a prompt or a cache key.

### Cluster target is UMBC HPCF `chip`; one GPU model per experiment (Nov 16, 2026, v5.4.02)

- **The cluster runner targets `chip`** (`cluster.md`; facts from the HPCF FAQ, read Nov 16). GPUs are a separate Slurm cluster (`-M chip-gpu`), the account is `pi_<group>`, and the GPU model is chosen with `--constraint`. Home is 500 MB, so the checkout, image, model, caches, and `runs/` live on the group's research volume. Modules work only on compute nodes, so setup runs inside a short GPU job.
- **One GPU model per experiment, L40S by default.** *Why:* replies can differ between GPU models (the Nov 14 comparability rule), so mixing the RTX 6000, RTX 8000, L40S, and H100 nodes inside a comparison would add hardware noise to paired differences. L40S: 40 cards (10 nodes x 4), 48 GB, and native kernels in llama.cpp's CUDA image (Ada, sm_89); the Turing cards would JIT-compile kernels.
- **Logs come back with every launch.** A dependent collect job runs after the last shard (success or not) and writes `summary.txt`: Slurm state per shard, requeues, failed episodes with a ready `only:` rerun line, server error counts, GPU and build per shard. A per-minute GPU sample (`gpu_shard<i>.csv`) gives the cost section utilization and memory. Preempted shards are requeued and rerun under the same stamp; finished calls replay from the response cache.
- **Not in the HPCF docs, recorded by `setup.sh` in `cluster_info.txt` on first use:** chip-gpu QOS limits, which partitions hold the L40S nodes, and the installed driver (a sample shows 575 / CUDA 12.9; the image needs CUDA 12.8+).

## Future work

Deferred on purpose. Nothing here is built unless marked.

- **Battery and charging (on hold until the MVP phase ends).** The model is designed (`simulator.md`, `battery:` config) and the loader rejects `enabled: true`. When built it adds a dynamic (agents return to their dock to charge) and an input for utilities.
- **Mission-scale sweep (planned, not built).** `experiments/sweep.py` has only `loss_levels`, `range_levels`, and `churn`. To build: scale axes in the sweep spec, generator support for the levels above (check that the easy and hard knobs generate; per-level houses are listed as environments, as `rho` would have been), `metrics.json` fields for `m`, `n`, and difficulty, and the H1 figure and H2 table in `experiments/plot.py`. The paper's Approach and Experiments are written as if it exists.
- **Richer utility inputs (decision needed).** Today the utility call sees only the capability card and the task text. The harness already holds the scene graph, held items, reservations, the action history, and the agent's bundle and ledger rows, so the call could take them too. Two consequences to decide first: (1) a score then depends on state, so "scored once per (agent, task)" must become a recompute rule (a task's scene changed, the agent's bundle or position changed, a fixed period); (2) CBBA's convergence argument needs each score to be bundle-independent at bidding time, so the bundle or the other tasks' bids must not enter the prompt, or the score must be frozen per auction round. Battery joins once it exists. Until decided, the paper describes the utility inputs as built.
- **Unbuilt conditions, planned for the end of November:** C4 (LLM+MURDOCH, with its bid mapping, open question 4), L (elected leader), R (random), C1+ledger, C3-noledger, and wiped-memory churn (open question 7). C4 has no numeric-MURDOCH counterpart in the plan; consider one so the utility effect can be isolated inside the second auction family.
- **Seeds, model, and latency:** final model (open question 3), seeds per cell, and latency calibration (open question 2) wait for the house pilot.

### Paper and course logistics

- **Paper is in AAAI style** (`aaai2027`) with natbib author-year citations. `[submission]` is removed so the instructor can see the author; restore it only if the class requires anonymity.
- **Related Work cites 24 papers**, grouped into five subsections with one `\paragraph` per paper (2-3 sentences each).
- **Figure 1 is the architecture diagram** (`fig_architecture.pdf`), a `figure*` spanning both columns.

## Open questions

1. **External-validity environment.** The September plan said "one AI2-THOR subset"; the October discussion favors an **Overcooked-AI port** (cheaper; precedent in LLM-Coordination). Pick one, or drop to a stretch goal.
2. **Latency parameters** (`R_prefill`, `R_decode`, `seconds_per_tick`): confirm in pilot runs. *Measured Nov 8 (Qwen3-8B Q4_K_M, RTX 4070 laptop, 118 calls):* prefill about 1,700 tok/s on the uncached part of each prompt, decode about 37 tok/s. The provisional 2,000 / 40 are within 20%; keep them. The model charges the whole prompt at the prefill rate though the server reuses cached prefixes, so it errs on the expensive side. `seconds_per_tick` stays 1.0, a modelling choice.
3. **Which 7-8B model** (Qwen3-8B, Llama 3.x 8B, other) and quantization. Decide in the week-7 pilot, then freeze. *Nov 9:* candidates and the bake-off plan are in `llm_models.md`. *Nov 10:* the bake-off ran after the restart (see the Nov 10 entry); no model is frozen yet.
4. **MURDOCH details for LLM+MURDOCH:** how LLM utilities map to single-item first-price bids, and how re-auction on failure interacts with the ledger.
5. ~~**Centralized LLM orchestrator** (SMART-LLM/COHERENT style) as a stretch baseline: yes or no?~~ Resolved Oct 27: covered by baseline L (elected-leader LLM).
6. **Lease timeout length** and how to report lease false positives under partitions.
7. **Dropout memory variants:** exactly what "memory wiped" resets (belief graph, ledger, action history, or all).
8. **Mission-scale levels and the difficulty definition** (Nov 10 proposal above): confirm the grid, and whether `n = 12` fits the hardware.
9. **Utility inputs and the recompute rule** (see Future work).

## Paper to-dos

- ~~Read DynaHMRC; confirm "elected leaders"; add its authors.~~ Done Oct 27 (experiments section read; ablation prose still unread).
- Soften the abstract's "provide conflict-free assignment" to "among connected agents" (see hard rule 10 in `CLAUDE.md`).
- Compress the abstract to a single paragraph of 150-200 words if targeting a venue; the four-paragraph version is for the class write-up.
- Consider adding HLA, DPT-Agent, LLM-Coordination, and the 2026 degraded-communication allocation benchmark (arXiv:2609.13711); see the end of `docs/paper/related_work_notes.md`.
- Convert `outline.md` into an Outline section of `main.tex` if the assignment PDF must contain it.
- Confirm DOIs and pages for entries marked *From recall* in `related_work_notes.md`.

## Superseded

- *Oct 15:* "100 ms ticks, agents block on LLM calls." Replaced by simulated time and the event-driven latency model.
- *Oct 15:* "Run 2-3 agents comfortably on the laptop, serialize the rest." Replaced by the one-server design; 6 agents is plausible with 4-8k tokens of context each.
- *Sept:* Six-condition table C0-C5. Replaced by the five conditions plus sweeps.
