# Decisions log

Newest entries last within each section. Format: **decision**, then *why*. Update this file in the same commit as any design change. Dates before October 15 are approximate because they come from an earlier planning session.

## Settled

### Project scope and experiment design (Sept 2026, planning session)

- **Pivot from gated continual RL to decentralized LLM + auction coordination.** The new topic has a clearer gap and fits the semester.
- **Core contribution is leaderless semantic allocation.** Per-agent LLMs generate bundle-independent utilities `u_ij`; a consensus auction resolves allocation; a replicated ledger tracks execution.
  *Why bundle-independent:* CBBA's convergence and approximation guarantees need diminishing marginal gains, so the LLM must not make a task's value depend on the rest of the bundle.
- **The LLM has exactly two jobs:** utility generation (cached per agent-task pair) and execution through tool calls.
- **Ledger semantics.** Status only moves forward (unclaimed, claimed, in progress, done; "done always wins"). Entries carry owner, bid, version, lease expiry. Gossip on contact. Leases detect crashed owners.
- **`rho` knob** (fraction of tasks with language-only constraints) is the main lever for H1, and guards against an "LLM just adds noise" outcome.
- **Heterogeneous team of about 6 agents**, 15-30 tasks per mission, precedence constraints.
- **Custom Python environment**, not an existing benchmark. *Why:* no existing environment combines LLM agents, heterogeneous allocation, message-level communication impairment, and churn (PARTNR has no communication model; AgentComm-Bench has no LLM agents).
- **Channel model** is `p_deliver = 1 / (1 + (d/d0)^alpha) * wall_penalty^walls_crossed`, bounded, never a raw `1/d^2`. Plus base loss `p` and optional latency.
- **Rendering is pixel-art, drawn in code or from CC0 assets.** No ripped game assets.
- **TraderBots is cited, not implemented.** *Why:* scope before Dec 17.
- **The HTML demo's greedy auction is not CBBA.** Full CBBA and MURDOCH must be implemented in Python.

### Conditions (resolved Oct 2026)

- **Five conditions, as in the abstract:** C0 oracle, C1 LLM-only, C2 numeric CBBA, C3 LLM+CBBA, C4 LLM+MURDOCH.
  *Why this supersedes the earlier C0-C5 table:* packet loss, range, and churn are now *sweeps applied to every condition*, not separate conditions. C2 stays essential: without it we cannot say the LLM helps beyond CBBA itself. C1 also gets the ledger, so the comparison isolates the allocation mechanism.

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

### Paper and course logistics

- **Paper is in AAAI style** (`aaai2027`) with natbib author-year citations. `[submission]` is removed so the instructor can see the author; restore it only if the class requires anonymity.
- **Related Work cites 24 papers**, grouped into five subsections with one `\paragraph` per paper (2-3 sentences each).
- **Figure 1 is the architecture diagram** (`fig_architecture.pdf`), a `figure*` spanning both columns.

## Open questions

1. **External-validity environment.** The September plan said "one AI2-THOR subset"; the October discussion favors an **Overcooked-AI port** (cheaper; precedent in LLM-Coordination). Pick one, or drop to a stretch goal.
2. **Latency parameters** (`R_prefill`, `R_decode`, `seconds_per_tick`): confirm in pilot runs.
3. **Which 7-8B model** (Qwen3-8B, Llama 3.x 8B, other) and quantization. Decide in the week-7 pilot, then freeze.
4. **MURDOCH details for LLM+MURDOCH:** how LLM utilities map to single-item first-price bids, and how re-auction on failure interacts with the ledger.
5. **Centralized LLM orchestrator** (SMART-LLM/COHERENT style) as a stretch baseline: yes or no?
6. **Lease timeout length** and how to report lease false positives under partitions.
7. **Dropout memory variants:** exactly what "memory wiped" resets (belief graph, ledger, action history, or all).

## Paper to-dos

- Read DynaHMRC (arXiv:2606.14882) in full; confirm the "reintroduce elected leaders" claim before final related-work prose.
- Add the DynaHMRC author list to `references.bib`.
- Soften the abstract's "provide conflict-free assignment" to "among connected agents" (see hard rule 10 in `CLAUDE.md`).
- Compress the abstract to a single paragraph of 150-200 words if targeting a venue; the four-paragraph version is for the class write-up.
- Consider adding HLA, DPT-Agent, LLM-Coordination, and the 2026 degraded-communication allocation benchmark (arXiv:2609.13711); see the end of `docs/paper/related_work_notes.md`.
- Convert `outline.md` into an Outline section of `main.tex` if the assignment PDF must contain it.
- Confirm DOIs and pages for entries marked *From recall* in `related_work_notes.md`.

## Superseded

- *Oct 15:* "100 ms ticks, agents block on LLM calls." Replaced by simulated time and the event-driven latency model.
- *Oct 15:* "Run 2-3 agents comfortably on the laptop, serialize the rest." Replaced by the one-server design; 6 agents is plausible with 4-8k tokens of context each.
- *Sept:* Six-condition table C0-C5. Replaced by the five conditions plus sweeps.
