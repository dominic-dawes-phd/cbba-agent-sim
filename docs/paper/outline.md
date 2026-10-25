# Paper Outline

**Working title:** Leaderless Semantic Allocation: Measuring What Consensus Auctions Buy LLM-Driven Robot Teams Under Communication Degradation and Agent Churn

---

## 1. Introduction

LLM-driven robot teams look good in papers, but almost all of them are tested with perfect communication. Real deployments don't work that way: radios drop packets, robots wander out of range, and teammates crash and come back. When that happens, LLM teams can claim the same task twice, abandon a task a dead teammate was holding, or act on an outdated picture of what's already done.

- **The gap:** classical decentralized allocators (CBBA, MURDOCH) are conflict-free among connected agents and hold up under changing networks, but they can't read constraints that only exist in language ("this crate has glassware"). LLM teams can read those constraints but lean on central orchestrators, elected leaders, or reliable messaging.
- **Research question:** if each robot's LLM only supplies *utilities* and a consensus auction does the actual allocation, does the team stay coordinated when communication degrades and agents churn? And what does that extra structure cost?
- **Contributions (planned):**
  - A leaderless architecture: per-agent LLM utilities + consensus auction + a replicated execution ledger.
  - A lightweight multi-robot simulator with a message-level communication model and agent dropout/rejoin.
  - A controlled comparison of five coordination conditions across packet loss, range, and churn sweeps.
  - A knob (ρ, the share of tasks with language-only constraints) that shows when the LLM actually earns its cost.

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
- Some tasks carry **language-only constraints**: facts that live in the task description and not in any numeric field (fragile, heavy, needs a keycard from another agent). ρ is the fraction of tasks that have one.
- **Communication** is unreliable: messages can be dropped, delayed, or blocked by distance and walls.
- **Goal:** maximize completed-task utility while avoiding duplicate work, orphaned tasks, and infeasible assignments.

## 4. Method

**LLM job #1: utilities.** Each agent's LLM reads a task's text plus the agent's own capability card and returns a utility u_ij. Utilities are cached and don't depend on what else is in the agent's bundle, which keeps CBBA's convergence guarantee intact.

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
- **Objects:** position, relation (on/in), rigid or deformable, number of hands needed, and language-only attributes. Simple drop rule: dropped near a table lands on the table, otherwise on the floor.
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
| C0 | Oracle | Central solver with ground-truth feasibility | none |
| C1 | LLM-only | Natural-language negotiation (gets the ledger too) | negotiates + executes |
| C2 | Numeric CBBA | CBBA with travel-cost utilities | executes only |
| C3 | LLM+CBBA | CBBA with LLM utilities | utilities + executes |
| C4 | LLM+MURDOCH | Single-item auctions with LLM utilities | utilities + executes |

**Sweeps** (each condition runs at perfect comms first, then):

- Packet loss p ∈ {0, 0.1, 0.3, 0.5, 0.7}
- Comms range r (small enough to cause partitions)
- Dropout k ∈ {0, 1, 2} agents, memory kept vs. wiped on rejoin
- ρ ∈ {0, 0.25, 0.5, 0.75} share of language-only tasks

**Setup:** ~6 heterogeneous agents, 15–30 tasks per mission, multiple seeds per cell, roughly 400 episodes total.

**Metrics**

- Utility relative to the oracle
- Task completion rate and makespan
- Duplicate, orphaned, and infeasible assignment rates
- Recovery time after dropout
- Ledger divergence between agents
- Messages sent, LLM calls, and tokens

**Ablations and checks**

- Compaction on vs. off
- Thinking cost set to zero (sensitivity check)
- Lease timeout length
- **Ceiling check:** a small subset re-run with a frontier API model, so "your model was just too small" isn't an easy objection
- **Stretch:** port a few tasks to Overcooked-AI for external validity

**How experiments map to hypotheses**

- **H1 (ρ sweep):** LLM+CBBA's advantage over numeric CBBA should grow with ρ; at ρ = 0 they should tie, with numeric CBBA far cheaper.
- **H2 (loss and range sweeps):** LLM-only should fall apart through duplicates and orphans, while auction teams degrade gradually. The robustness comes from consensus + ledger, not the LLM.
- **H3 (low connectivity):** below some connectivity level, consensus can't converge within the replanning horizon, and LLM-only might actually win.

## 7. Expected Results

- A crossover plot for H1 instead of a simple win/lose: the LLM helps only when there's language-only information to exploit.
- Clear separation in duplicate/orphan rates under loss (H2).
- An estimated connectivity threshold (H3). If LLM-only turns out more robust than expected, that's a finding, not a failure.

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
- Utility/oracle vs. ρ, one line per condition (H1)
- Duplicate and orphaned task rate vs. packet loss (H2)
- Recovery time after dropout, by condition
- Convergence success vs. connectivity, showing the threshold (H3)
- Cost panel: tokens and messages per completed task

## Timeline (to Dec 17)

- **Weeks 1–2:** world, primitives, comms channel, oracle
- **Weeks 3–4:** numeric CBBA, MURDOCH, ledger + gossip, unit tests
- **Weeks 5–6:** LLM agents and LLM-only baseline
- **Week 7:** pilot runs, tune ρ and difficulty, freeze config
- **Weeks 8–10:** main sweeps, ablations, ceiling check
- **Weeks 11–12:** writing
