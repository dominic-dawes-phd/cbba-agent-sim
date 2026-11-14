# Paper Outline

**Working title:** Leaderless Semantic Allocation: Evaluating Consensus-Auction Agent Teaming Under Communication Degradation and Agent Churn

---

## 1. Introduction

LLM-driven robot teams look good in papers, but almost all of them are tested with perfect communication. Real deployments don't work that way: radios drop packets, robots wander out of range, and teammates crash and come back. When that happens, LLM teams can claim the same task twice, abandon a task a dead teammate was holding, or act on an outdated picture of what's already done.

- **The gap:** classical decentralized allocators (CBBA, MURDOCH) are conflict-free among connected agents and hold up under changing networks, but they score tasks with hand-designed numeric utilities over a few features, and richer context needs more hand engineering. LLM teams reason over rich context but lean on central orchestrators, elected leaders, or reliable messaging, and their errors compound with team and mission size.
- **Research questions:** if each robot's LLM only supplies *utilities* and a consensus auction does the actual allocation: does the auction keep LLM teams coordinated as missions scale (H1); do LLM utilities beat numeric ones (H2); does the team stay coordinated, and recover faster, when communication degrades and agents churn (H3)? And what does that extra structure cost?
- **Contributions (planned):**
  - A leaderless architecture: per-agent LLM utilities + consensus auction + a replicated execution ledger.
  - A lightweight multi-robot simulator with a message-level communication model and agent dropout/rejoin.
  - A controlled comparison of coordination conditions across mission-scale, packet-loss, range, and churn sweeps.
  - A ledger ablation that separates what consensus buys from what the shared state buys.

## 2. Related Work

Already drafted as mini-paragraphs (24 papers). For the final paper this becomes connected prose in five groups:

- Classical decentralized allocation (Contract Net, MURDOCH, TraderBots, CBBA, ACBBA)
- Allocation under degraded communication (Ponda et al., Otte et al., event-driven CBBA)
- LLM-driven multi-robot coordination (RoCo, SMART-LLM, COHERENT, LaMMA-P, LLaMAR, CoELA, DynaHMRC)
- Task tracking and replicated state (PARTNR, gossip, CRDTs, leases, ReAct)
- Evaluation under communication impairment (AgentComm-Bench)

Closing point: nobody combines per-agent LLM utilities with a leaderless auction and an execution ledger, then stress-tests it against an oracle under swept loss, range, and churn.

## 3. Problem Setup

- A team of agents *A* = {a₁ … aₙ} with different capabilities (hands, payload, speed, comms range, success probability).
- A mission of tasks *T* = {τ₁ … τₘ}, each a sequence of primitive steps with precedence constraints (open the dishwasher before loading it).
- **Communication** is unreliable: messages can be dropped, delayed, or blocked by distance and walls.
- **Goal:** maximize completed-task utility while avoiding duplicate work, orphaned tasks, and infeasible assignments.

## 4. Method

**LLM job #1: utilities.** Each agent's LLM reads a task's text plus the agent's own capability card and returns a utility u_ij (richer inputs, such as the scene graph and action history, are future work). Each score is made once per agent and task and cached, and doesn't depend on what else is in the agent's bundle, which keeps CBBA's convergence argument intact among connected agents.

**Allocation.** A consensus auction (CBBA, or MURDOCH as a second family) uses those utilities to assign tasks. No leader and no central planner.

**LLM job #2: execution.** Each agent works through its bundle by calling action primitives as tools: `move_to`, `pick`, `place`, `open`, `close`, `scan`, `handoff`, `report`. Failures come back to the LLM as short text ("pick failed: plate slipped, now on table").

**Replicated ledger.** Every agent keeps its own copy of the task board:

- Each entry: status, owner, bid, version/timestamp, lease expiry.
- Status only moves forward: unclaimed → claimed → in progress → done ("done always wins" on merge).
- Copies sync by gossip whenever agents can hear each other.
- If an owner's lease expires (it crashed or went silent), teammates can reclaim the task.

**Memory compaction.** The narrative action history is summarized after N entries to keep prompts short. The ledger itself is never compacted.

## 5. Simulator

There's no existing environment that has LLM agents, heterogeneous allocation, message-level comms impairment, *and* churn, so I'm building a lightweight one. It follows the same recipe as PARTNR, LLaMAR, and CoELA, so it shouldn't look unusual to reviewers.

- **World:** 2D household/kitchen grid with rooms, walls, furniture, and objects (Overcooked-style; MindAgent's CuisineWorld and LLM-Coordination's Overcooked-AI are precedent).
- **Objects:** position, relation (on/in), rigid or deformable, number of hands needed, and attributes only a scan reveals. Simple drop rule: dropped near a table lands on the table, otherwise on the floor.
- **Agents:** capability card (hands, movement speed, field of view, scanner, per-skill success probability) and gripper state. Communication range is one swept value for the whole team.
- **Observations:** symbolic, a text scene graph of what the agent can currently see, plus remembered and teammate-reported objects with their staleness. Distances are given as BFS step counts. No pixels.
- **Time:** discrete simulated ticks, not wall-clock. The world waits while LLMs think, so results don't depend on GPU speed. Each LLM call costs the calling agent idle ticks computed from its token counts at a fixed reference throughput (a "thinking cost"), so LLM-heavy conditions don't get free deliberation. Teammates keep acting meanwhile. Travel takes `ceil(steps / speed)` ticks.
- **Communication model:** delivery probability falls off with distance and each wall crossed, plus a base packet-loss rate p and optional latency.
- **Churn:** k agents crash at a random step and rejoin after Δ steps, either keeping or losing their memory.
- **Interface:** PettingZoo Parallel API (reset/step), so it's familiar and easy to reproduce.
- **Rendering:** a pixel-art viewer for debugging and figures, fully separate from headless experiment runs.
- **Logging:** per-step state and ledger snapshots, every message sent/dropped, every LLM call (prompt, output, tokens), every task event.

**Compute:**

- One quantized open model (e.g., Qwen3-8B, 4-bit) served once via llama.cpp or vLLM; each agent is just a separate conversation.
- Development and pilots on my laptop (RTX 4070, 8 GB).
- Full sweeps on the UMBC cluster as Slurm job arrays (one job per condition × seed).

## 6. Experiments

**Conditions**

| ID | Condition | Allocation | LLM role |
|---|---|---|---|
| C0 | Oracle | Central solver with ground-truth feasibility: the best known plan, with a proven bound in its planning model, replayed in the sim | none |
| C1 | LLM-only | RoCo-style dialogue rounds within each communication component; no ledger | negotiates + executes |
| C2 | Numeric CBBA | CBBA with travel-cost utilities | executes only |
| C3 | LLM+CBBA | CBBA with LLM utilities | utilities + executes |
| C4 | LLM+MURDOCH | Single-item auctions with LLM utilities | utilities + executes |

**Baselines and ablations**

| ID | Condition | What it isolates |
|---|---|---|
| G | Greedy: nearest unclaimed task plus a broadcast claim, no consensus (LLM executor, like C1-C4) | Any coordination vs consensus |
| L | Elected-leader LLM: one agent elected per mission assigns every task | Leaderless vs elected leader (DynaHMRC-style); doubles as a central orchestrator, which a partition or the leader's crash cuts off |
| C1+ledger, C3-noledger | C1 with the replicated ledger; C3 without it | With C1 and C3, a 2x2 that shows whether H3's robustness comes from consensus or from the ledger |
| R | Random allocation (LLM executor) | Sanity floor |

Every condition is one choice of S2 executor, allocator, utility source, and shared state, over identical perception, S1 skills, and task recipes (`docs/design/conditions.md`).

**Sweeps** (each condition runs at perfect comms first, then):

- Mission scale (one factor at a time around the baseline; levels proposed in `decisions.md`, Nov 10): tasks m ∈ {11, 22, 44} with the house area scaled to hold density (32x24, 48x32, 64x48), team size n ∈ {3, 6, 12}, difficulty ∈ {easy, medium, hard} (dish-cluster size and stack size, which set shared-appliance coupling and per-task load)
- Packet loss p ∈ {0, 0.1, 0.3, 0.5, 0.7}
- Comms range r ∈ {∞, 16, 12, 8, 6, 4} cells (small enough to cause partitions)
- Dropout k ∈ {0, 1, 2} agents, memory kept vs. wiped on rejoin

**Setup:** baseline 6 heterogeneous agents and 22 tasks per mission, multiple seeds per cell, run count to be set after the house pilot (`DOE.md`, section 6).

**Metrics** (all computed from logs by `experiments/analyze.py`; means with 95% bootstrap CIs over seeds)

- Utility relative to the oracle, measured as `makespan_vs_oracle` (a run's makespan over C0's replayed makespan per mission and seed; each oracle run also reports its proven bound and planned makespan)
- Success rate (every task done) and partial success (share of tasks done), as in DynaHMRC
- Makespan in ticks, shown next to voided tasks (a task whose only item shattered is voided, which removes its work); drops otherwise cost time: a pause, or a cleanup reflex
- Duplicated work (ticks spent on a task another agent finished) and contested tasks; orphan delay (crash to completion of the crashed agent's task); infeasible assignments
- Stale-belief and hallucination rates; blocked ticks (congestion)
- Recovery time after dropout
- Ledger divergence between agents (`ledger_lag_final`)
- Messages sent and dropped, LLM calls by purpose, and tokens; consensus messages per task (the price of CBBA)
- False completion claims (a task claimed done that still needed work; RoCo's "false-done herding") and their rate per claim; re-plan attempts (rejected LLM replies) and their rate per decision; C1 plan rejections

Field names, formulas, source logs, and caveats for every metric: `docs/DOE.md`, section 3, "Metric reference for the paper".

**Ablations and checks**

- Headroom pilot before any comparison: best vs median static assignment on each mission (`experiments/headroom.py`); a mission where they barely differ cannot separate allocators
- Compaction on vs. off
- Thinking cost set to zero (sensitivity check)
- Lease timeout length
- **Ceiling check:** a small subset re-run with a frontier API model, so "your model was just too small" isn't an easy objection
- **Stretch:** port a few tasks to Overcooked-AI for external validity

**Hero metrics** (the headline numbers, all from logs)

1. Completion within the tick budget (success and partial success).
2. Makespan relative to the oracle.
3. Wasted work (ticks on tasks another agent finished) and orphaned tasks.
4. Messages and tokens per completed task.

**Hero figures** (`experiments/plot.py`; PNG, PDF, and a CSV table twin each; other sweep axes held at baseline)

- **H1 (planned with the scale sweep):** completion, makespan relative to the oracle, and wasted work against `m`, difficulty, and `n`, one line per condition.
- **H2 (planned):** table of paired C3-minus-C2 differences in `makespan_vs_oracle` and success, with tokens per completed task.
- **H3:** completion and wasted work against packet loss `p` (`fig_h3_loss`), two panels, one line per condition with 95% CI bands. Liu et al. 2013 Fig. 7 style; DynaHMRC reports bars with bootstrap CIs.
- **H3:** completion against communication range `r` (`fig_h3_range`), crossovers between conditions marked where they occur (reported, not a hypothesis).
- **Churn (H3):** tasks outstanding over time with the crash tick marked; the slope after the crash shows recovery (Liu et al. 2013 Fig. 12 style).
- **Cost:** tokens per episode against completion, one labeled point per condition (DynaHMRC Tab. 26 analogue).

**Missions:** seeded 48x32 houses with 7 rooms, 6 agents, and 22 tasks (`experiments/generate_mission.py`). The 22x8 kitchen showed only 10% headroom between the median and best static assignment; the generated houses show 26-54%.

**How experiments map to hypotheses**

- **H1 (scale sweeps):** LLM-only teams (C1) should degrade through duplicates and orphans as `m`, difficulty, and `n` grow, while auction teams (C2, C3) do not. G sets the duplicate-work level of coordination without consensus.
- **H2 (C3 vs C2):** LLM utilities should close more of the gap to the oracle than numeric ones, at a measurable token cost.
- **H3 (loss, range, churn sweeps):** LLM-utility auctions should degrade less and recover faster than numeric auctions, beyond what the ledger alone gives (C3-noledger and C1+ledger split the gain between consensus and ledger). Resilient or emergent behavior is claimed only if the data show it.

## 7. Expected Results

- Growing duplicate and orphan rates for LLM-only teams with mission scale, and flat rates for auction teams (H1).
- A token-cost-versus-gap trade-off between C2 and C3 (H2).
- Separation between conditions in degradation slope and recovery time under loss, range, and churn (H3). If LLM-only turns out more robust than expected, that's a finding, not a failure.

## 8. Discussion and Limitations

- Symbolic perception only; no vision models.
- Simulated communication, not real radios.
- Small quantized model; partly addressed by the ceiling check.
- CBBA's conflict-free guarantee only holds among connected agents, which is exactly what partitions break. This is the point of the experiment, but it should be stated plainly.
- TraderBots and centralized LLM orchestrators are cited, not run.

## 9. Conclusion

What a consensus auction buys LLM teams, when the LLM is worth its cost, and where the whole approach stops working.

---

## Figures

**Figure 1 (include in this submission):** system architecture. Three agents, each with an LLM box producing utilities → shared CBBA consensus layer → per-agent ledger copies connected by a lossy radio link → action primitives into the world. A screenshot of the pixel-art simulator is a good second option.

**Planned figures for the final paper:**

- Simulator screenshot with ledger panel and radio links
- Outcomes vs. mission scale (m, difficulty, n), one line per condition (H1)
- Paired C3-minus-C2 gap in makespan relative to the oracle, with token cost (H2)
- Duplicate and orphaned work vs. packet loss and range (H3)
- Recovery time after dropout, by condition (H3)
- Cost panel: tokens and messages per completed task

## Timeline (to Dec 17)

- **Weeks 1–2:** world, primitives, comms channel, oracle
- **Weeks 3–4:** numeric CBBA, MURDOCH, ledger + gossip, unit tests
- **Weeks 5–6:** LLM agents and LLM-only baseline
- **Week 7:** pilot runs, tune scale levels and difficulty, freeze config
- **Weeks 8–10:** main sweeps, ablations, ceiling check
- **Weeks 11–12:** writing
