# Conditions: canonical definitions

This is the one place that defines every condition, baseline, ablation, and dev-only policy. CLAUDE.md, `DOE.md`, the outline, and the paper summarize it, and they must not contradict it. Decided Nov 2, 2026 (see `decisions.md`). Update this file first when a condition changes.

## 1. The layer model

Every robot team is described by the same seven layers. Three are **fixed for every condition**, so no condition gets better eyes, better hands, or better task know-how. A condition is exactly one choice for each of the other four. If two conditions differ in more than the layer a comparison is meant to isolate, that comparison is confounded.

| Layer | Job | Real-world analog | In the sim | Varies? |
|---|---|---|---|---|
| Perception | Sensors to a scene graph, plus multi-robot map merging | VLM perception, scene-graph mapping (`gu2024conceptgraphs` inspired), SLAM map merging | Deterministic perception into a per-agent belief graph (`plate_2 in cabinet_1[2]`), hard rule 5; coverage maps shared as `coverage` messages over the lossy channel (where teammates looked, not what they saw) | **No** |
| S1 skills | Run one motor skill over many ticks | VLA / visuomotor policy. Illustrative, not cited: Figure Helix's S1, NVIDIA GR00T's fast system. | Primitives (`move_to`, `explore`, `search`, `pick`, `place`, `open`, …) run by the sim, with fumbles and drops; `explore`/`search` pick their own frontier cell by code; a shattered dish triggers the `clean_up` and `return_to_task` reflex, and a dish in hand when its wash splits goes back by `return_item`, all with no LLM call; a queued pick is cancelled once its item is seen elsewhere | **No** |
| Task know-how | How "unpack the dishwasher" breaks into skills | What a generalist policy was trained on | Task templates (`configs/task_templates.yaml`), given to every executor as a recipe | **No** |
| S2 executor | At decision points: choose the next skill, handle failures, replan | Onboard VLM/LLM planner (Helix's S2) | `coded` (the scripted executor in `allocation/scripted.py`) or `llm` (`allocation/llm_policy.py`, one prompt history per agent on a shared server) | coded \| llm |
| Allocator | Who does which task | Fleet dispatcher, market, negotiation | Oracle solver, round-robin, nearest-task, CBBA, MURDOCH, dialogue, elected leader, random | **Main variable** |
| Utility source | How well an agent suits a task | Cost model vs semantic judgment | Static BFS steps (hard rule 8), or an LLM utility `u_ij` | steps \| llm |
| Shared state | Who owns what, what is done | Fleet database / blackboard | Replicated ledger (`simulator.md` §9), or none | ledger \| none |

The *ledger* is a structured, replicated task board: `task_id -> {status, owner, bid, version, lease_expiry}`, with monotone status, synced by gossip and merged by deterministic code. It is **not** the action history. The action history is the narrative of an agent's own skill calls and their results, compacted after `N` entries. Every LLM condition gets an action history; only some get a ledger.

**Common to every LLM agent:**
- the belief graph as text, with the distances to its two exploration frontiers;
- its capability card;
- the descriptions of the tasks it knows: in discovered mode (paper runs), only tasks it revealed or learned of; in posted mode, all mission tasks;
- its own compacted action history;
- failed-call feedback (a rejected response still costs `L`);
- the primitives, including `explore` and `search`;
- its coverage maps, merged from teammates' coverage messages.

**No condition gets a free-text notes tool.** Long-horizon memory is history plus compaction only, the same mechanism for all.

**Executor-layer rules shared by every LLM condition** (`allocation/llm_policy.py`, v5.1.00). They give the LLM executor what the coded executor already knows or does:
- **Locks belong to their task.** At every decision, code unlocks any workspace the agent holds that its current task's recipe does not lock. The coded executor always unlocks; an LLM that forgot once stranded a teammate's clean-up reflex for 800 ticks. The rule is applied every decision, so an unlock lost with a rejected reply is issued again.
- **Task tools name the agent's own task.** `report`, `release`, `mark_done`, and `give_up` on any other id (or with no task) reject the reply, as the coded executor only ever reports its own task.
- **G retakes a released task** after `RELEASE_RETAKE_TICKS` (120). The coded executor never drops a mission task for good, and three LLM agents that each released the groceries once had orphaned them.
- **Voided tasks are flagged.** When every item a task moves has shattered, the prompt says the task is over and names the condition's finishing tool (`report`, `mark_done`, or `talk_to_team`). The coded executor reads the same thing from the shared task list.
- **Failures read as `ok` or `FAILED` plus the reason.** The outcome category (`stale_belief`, ...) is an analysis label and is not shown.

**How agents learn of a task:** by their own sight of its reveal objects (all conditions), or by a teammate's `task_found` announcement. `task_found` carries a structured task id, so it goes to the ledger conditions, C1+ledger, and G, but **not to C1**. C1 agents learn a teammate's find only from the teammate's free-text dialogue.

**What an agent does with a task it finds** depends on the condition. Finding never interrupts a busy agent; what the find turns into differs:

| Condition | A found task becomes |
|---|---|
| C0 | Nothing to do: C0 knows every task from tick 0 (a split's new task is adopted by the coded executor) |
| C1 | A note in that agent's LLM context; it may mention it in dialogue |
| C1+ledger | A ledger row, plus a note in context; dialogue still allocates it |
| C2, C3, C4 | An unclaimed ledger row, gossiped and added to the auction (CBBA or MURDOCH) |
| G | A `task_found` broadcast and an entry in the finder's known tasks; nearest free agent takes it |
| L | A report to the elected leader (planned; message format open) |
| R | An entry in the finder's known tasks (planned; how random allocation handles late tasks is open) |

**Washed dishes seen inside an open dishwasher** produce an unload task that its finder announces like any found task (`simulator.md` §9).

**A wash split by someone else's cycle start** (a simulator rule, `simulator.md` §9) produces a new task that its finders announce like any found task, so each condition handles it as the table says.

**The coded executor** (G, `scripted`, C0's replay; LLM agents decide these things themselves) never adopts a task in a teammate's fixed assignment or targets an item a teammate holds, skips an open or close the door is already believed in, and does not idle at a busy dishwasher. A wash whose machine is believed mid-cycle (by its countdown) with nothing loaded waits it out when the cycle ends sooner than the cheapest way out (the walk to another free dishwasher, or the round trip to put the dish back), or when the agent has nothing else to do; otherwise it moves to another dishwasher believed free, or puts the dish back, gives the task up, and does not take a wash on while its machine is busy. C0's replay waits instead, since its plan already holds the machine.

*As built (v5.1.00):* G (coded or LLM executor) broadcasts `task_found`. C2 and C3 add a found task to the ledger and the auction. C1's channel silences `task_found`, so a find reaches teammates only through dialogue. C4, L, R, and the ablations are not built.

## 2. Condition cards

### C0: Oracle (reference)
- **Executor:** coded, at no deliberation cost (zero tokens, `L = 0`). **Allocator:** a central solver: a seeded heuristic (insertion plus local search) whose plan seeds OR-Tools CP-SAT (`allocation/oracle.py`, `allocation/oracle_cpsat.py`). **Utilities:** ground truth. **Shared state:** not needed; the plan is fixed at tick 0.
- **Knows:** every object and every task from tick 0 (`omniscient_beliefs`), the one documented exception to hard rule 5.
- **Unloads:** between consecutive washes on a shared dishwasher, C0 plans an unload of the earlier batch before the next is loaded, trying every order of those washes (`simulator.md` §11).
- **Plans assignment, order, and start times only.** The planning model uses primitive ticks and static BFS steps (hard rule 8), precedence, and exclusive furniture, including a dishwasher through its wash. Paths come from the shared BFS, and collisions are resolved at execution by the motion rules. Multi-robot path planning is out of scope.
- **Three numbers per (mission, seed):** the proven lower bound in the model, the planned makespan, and the replayed makespan. The proof covers the model, not the sim, so C0 is the **best known plan**, not "the optimum". The denominator for every other condition is the **replayed** makespan, so all conditions are measured under the same physics. C1-C4 should get close without matching it. A ratio slightly below 1 is still possible, through congestion in C0's replay, and is reported as it is.
- **What the gap to C0 measures:** allocation *and* information, on purpose. C0 is the perfect case: it has no search cost, and no posted-task variants of other conditions are planned (the user's decision, Nov 7). A condition's search cost is part of its gap. Read the gap with discovery latency and exploration ticks.
- **Scope:** computed per (mission, seed), since spawns differ by seed, at baseline (perfect comms, no churn, drop 0, posted tasks); not swept. Each mission's tick budget is `k x` the slowest heuristic-plan replay over a few spawns, so no competing condition sets the yardstick.
- **Reproducibility:** search is limited by counts and CP-SAT's deterministic time, using interleaved search, never a wall clock.
- **Analog:** an offline operations-research solver with perfect information. Not achievable in practice; a yardstick.

### C1: LLM-only (RoCo-style dialogue, no ledger)
- **Executor:** llm. **Allocator:** natural-language dialogue. **Utilities:** implicit in the dialogue. **Shared state:** none.
- **Template:** RoCo (`mandi2024roco`). Prompt sections follow RoCo's: task context, round history (here, the compacted action history plus the transcripts the agent received), capability, communication instruction, current observation (the belief graph plus this round's received dialogue), and plan feedback.
- **Tools:** the primitives, plus the round protocol below. **No** `claim`/`release`/`report`/`task_found` (they carry structured task ids) and no ledger reads. Who-does-what lives only in the transcript and the history.
- **Round protocol (lockstep dialogue per communication component):**
  1. **Trigger:** an agent's decision point opens a round, unless its component already has a round running, in which case it waits for that round's plan.
  2. **Members:** the online agents connected to the trigger at round start (comm range, walls, online). Partitioned and crashed agents are not members.
  3. **Turns:** sequential, in member-id order starting with the trigger. Each turn is one LLM call costing `L` ticks plus a one-tick hop. Every message goes through the lossy channel to each other member separately, so a member's view of the transcript holds only the messages that reached it.
  4. **Ending a turn:** each reply ends with `PROCEED`, or with `EXECUTE` and a joint plan (each member gets a task id, `WAIT`, or `CONTINUE`). `EXECUTE` is allowed only after every member has spoken this round. At most `M` turns per round.
  5. **Validation:** a joint plan is checked in order: parse, then capability and hands constraints, then the belief check (hallucination). A failure returns the reason to the members as plan feedback and starts another exchange, up to `K` re-plans. After that the round ends with no plan.
  6. **Adoption:** a member adopts the plan only if the `EXECUTE` message reached it.
  7. **Execution is not lockstep.** Members in the middle of a skill keep executing while their LLM talks (S1 runs while S2 deliberates) and adopt the plan when the skill ends. The trigger idles until the plan arrives.
  8. **Partitions:** `k` partitions mean `k` independent concurrent rounds. Each has its own canonical transcript (`dialogues` stream), and per-receiver deliveries are in `messages`.
- **No-bleed rules:** no structured task table of any kind; no notes tool. A plan is advice that each receiver adopts or not. Nothing enforces agreement.
- **Expected failure modes** (RoCo App. 12.1 shows the first): false-done herding (one agent claims a task is done and the others repeat it), duplicated work after lost messages, and forgetting under compaction.
- **Analog:** RoCo, and CoELA-style dialogue teams (`zhang2024coela`).
- **As built** (`allocation/dialogue.py`, `condition: llm_only`, `dialogue:` block):
  - **Limits:** `M` = `turns_per_member` (2) x members and `K` = `replans` (2). Messages are at most `message_chars` (400).
  - **Code checks:** members, known tasks, and that everyone spoke before EXECUTE. Capability and hands are not checked yet.
  - **Asking for a round:** an agent with no task asks for one, and so does an agent that calls `talk_to_team`, the C1 tool for "I finished" or "I am stuck". After a WAIT it asks again after `retry_ticks` (30).
  - **Without a task:** the agent holds still with no LLM call, unless it holds a workspace it must still unlock.

### C2: Numeric CBBA
- **Executor:** llm. **Allocator:** CBBA (`choi2009cbba`). **Utilities:** static BFS travel cost, computed by code. **Shared state:** ledger.
- **What the LLM does:** receives its bundle ("t3 then t7") and executes it through primitives. It reads the ledger and reports through the status tools, `mark_done` (done) and `give_up` (failed or released). It never sets utilities or bids.
- **As built** (`allocation/cbba.py`, `condition: numeric_cbba`, `cbba:` block):
  - **Bid:** `c_ij = s_ij * discount^steps_ij`, with `s_ij` = 1 for a capable agent (hands, scanner) and 0 otherwise. `discount` is 0.97, and bundles hold up to `bundle_size` (3) tasks.
  - **Consensus:** Choi et al.'s update table on `cbba` messages. The head task is committed once work on it starts.
  - **Ledger:** the bundle is mirrored as claims with leases (`lease_ticks` 60), and both are gossiped on change and every `gossip_every` (10) ticks.
  - **Giving up:** a task given up is not re-bid by that agent for `give_up_ticks` (120).
- **No-bleed rule:** C2 differs from C3 **only** in the utility source (steps vs an LLM score); that difference is H2.
- **Analog:** a classical cost-based allocator assigning jobs to robots whose onboard VLM/VLA executes them. CBBA comes from decentralized UAV task assignment.

### C3: LLM+CBBA
- **Executor:** llm. **Allocator:** CBBA. **Utilities:** LLM `u_ij`, one per (agent, task), independent of the rest of the bundle, cached. **Shared state:** ledger.
- **LLM tools:** read the ledger; output `u_ij`; report done (`mark_done`), or failed or released (`give_up`).
- **As built** (`condition: llm_cbba`): identical to C2 except `s_ij`. When an agent learns of tasks it has not scored, its next decision first makes one batched utility call (`purpose: utility`, `allocation/utility.py`). The call returns 0-100 per task, divided by 100, and is cached per (agent, task) for the episode. A task the reply leaves out scores 0.5. The call costs `L` like any other. **CBBA code** builds bids, runs consensus, and enforces monotone status. The LLM never edits ledger rows or places bids itself.
- **Why the restriction:** CBBA's convergence and conflict-freedom (among connected agents only, hard rule 10) assume bundle-independent utilities and a monotone ledger. An LLM that edits rows or bids freely is doing the allocation, and C3 would collapse into C1+ledger.
- **Analog:** no deployed system we know of; this combination is the contribution. The nearest work pairs an LLM with classical planning (`kannan2024smartllm`, `zhang2025lammap`).

### C4: LLM+MURDOCH
- **Same as C3 in every layer except the allocator:** single-item auctions (`gerkey2002sold`). Whoever posts or discovers a task auctions it, collects bids for a fixed window, and awards it. The winner renews ownership, and a task is re-auctioned on failure.
- **Tests:** a second auction family, and sensitivity to round trips (each award needs a bid and a reply). How LLM utilities map to bids is open question 4 in `decisions.md`.

### G: Greedy nearest-task (baseline)
- **Executor:** llm in paper runs; coded for dev and pilots. **Allocator:** whenever free, take the nearest task by static steps that this agent knows of and has not heard claimed or done, broadcast a claim, and do it. **Utilities:** steps. **Shared state:** none (claims are messages, not a ledger).
- **Knows:** in discovered mode (paper runs), only the tasks it revealed itself or heard of through `task_found`. In posted mode, every mission task from tick 0. With no task it can take, it explores.
- **Isolates:** any cheap coordination vs consensus. Duplicated work comes from claim races and lost claims.
- **Analog:** nearest-available dispatch in simple robot fleets: the closest free robot takes the job and announces it.
- **Code:** `allocation/greedy.py`: `GreedyPolicy` (coded executor), and `GreedyAllocator`, which `LLMPolicy` wraps with the LLM executor (`executor: llm`). In the LLM version code issues the claim, and the LLM reports and releases with the `report` and `release` tools. Its prompt lists the claims and reports the agent has heard (`CLAIMED BY TEAMMATES`, `REPORTED DONE`), which G's code tracks for the coded executor too.

### L: Elected-leader LLM (baseline)
- **Executor:** llm. **Allocator:** one agent, elected per mission, assigns every task in natural language. **Utilities:** the leader's judgment. **Shared state:** none.
- **Isolates:** leaderless vs leader. A partition or the leader's crash cuts the team off from allocation.
- **Analog:** DynaHMRC's elected leader (`dynahmrc2026`), and central orchestrators (`kannan2024smartllm`, `liu2025coherent`). RoCo's "Central Plan" baseline is a full-observation version of the same idea.

### R: Random (sanity floor)
- **Executor:** llm in paper runs; coded for dev. **Allocator:** each task goes uniformly at random to an agent capable of it, fixed at tick 0. **Shared state:** none.

### Ablations: C1+ledger and C3-noledger
A 2x2 for H3, separating the allocation mechanism from the shared state:

| | No ledger | Ledger |
|---|---|---|
| **Dialogue (C1 protocol)** | **C1** | **C1+ledger**: C1 plus ledger reads and the status tools |
| **Auction (CBBA)** | **C3-noledger**: CBBA consensus only, no replicated execution state | **C3** |

### `scripted`: dev only, never in the paper
- **Executor:** coded. **Allocator:** the mission's fixed `scripted_assignment`. In generated houses this is round-robin: task *i* goes to agent *i* mod *n* (`experiments/generate_mission.py`). **Shared state:** none (claim and report messages).
- **Used for:** regression tests, smoke runs, and the headroom pilot's hand-assignment row. Not a baseline, and distinct from R.

## 3. Comparison map

| Comparison | Differs only in | Serves |
|---|---|---|
| C2 vs C3 | Utility source (steps vs LLM) | H2; also H3 (both have the ledger) |
| C3 vs C4 | Auction family | Generality of H2 and H3 |
| C1 vs C1+ledger, C3-noledger vs C3 | Shared state | H3 attribution (ledger) |
| C1 vs C3-noledger, C1+ledger vs C3 | Allocator (dialogue vs consensus) | H1, H3 attribution (consensus) |
| G vs C2 (both LLM executor) | Allocator (nearest-task vs CBBA) | "Does consensus beat any coordination?" |
| L vs C1 | Leader vs leaderless dialogue | Leaderless claim |
| Any condition, coded vs llm executor | Executor | "Does LLM reasoning help recovery?" (optional; not in the run budget) |
| Every condition vs C0 | Gap to the best known plan | All hypotheses |
