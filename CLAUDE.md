# CLAUDE.md: Leaderless Semantic Allocation (CMSC 601 project)

Read this file first every session. Details live in `docs/`; do not duplicate them here.

## What this project is

Course project for CMSC 601 (UMBC), due **Dec 17, 2026**. Author: Dominic Dawes.

Paper title: *Leaderless Semantic Allocation: Measuring What Consensus Auctions Buy LLM-Driven Robot Teams Under Communication Degradation and Agent Churn.*

**Research question.** If each robot's LLM only supplies task *utilities*, and a leaderless consensus auction (CBBA) does the allocation, does the team stay coordinated when communication degrades and agents drop out and rejoin? What does that added structure cost compared with unconstrained LLM coordination?

**Architecture in one paragraph.** Each agent's LLM turns a task's natural-language description plus the agent's capability card into a bundle-independent utility `u_ij` (cached). A consensus auction (CBBA, and MURDOCH as a second family) resolves the allocation with no leader. Each agent executes its bundle by LLM tool calls to action primitives. Allocation and execution state live in a replicated per-agent **ledger** (monotone task status, lease-based ownership, gossip sync). Narrative action history is compacted after N entries; the ledger is never compacted.

## The five conditions (keep these names and this order)

| ID | Condition | Allocation | LLM role |
|---|---|---|---|
| C0 | Oracle | Central solver with ground-truth feasibility | none |
| C1 | LLM-only | Natural-language negotiation (also gets the ledger) | negotiates + executes |
| C2 | Numeric CBBA | CBBA with travel-cost utilities | executes only |
| C3 | LLM+CBBA | CBBA with LLM utilities | utilities + executes |
| C4 | LLM+MURDOCH | Single-item auctions with LLM utilities | utilities + executes |

Each runs under perfect communication, then sweeps: packet loss `p`, communication range `r`, `k`-agent dropout/rejoin (memory kept vs wiped), and `rho` (fraction of tasks with language-only constraints).

## Hypotheses (do not reword without updating the abstract in `docs/paper/main.tex`)

- **H1.** LLM-generated utilities close the gap to the oracle in proportion to `rho`. Where constraints are purely geometric (`rho = 0`), numeric CBBA matches LLM+CBBA at a fraction of the cost.
- **H2.** Robustness and reasoning come apart. LLM-only teams degrade sharply as `p` rises and `r` shrinks, through duplicated and orphaned tasks. Auction teams degrade gracefully, so the gain is attributable to consensus and the replicated ledger, not the LLM.
- **H3.** There is a connectivity threshold below which consensus fails to converge within the replanning horizon, and below which greedy LLM-only coordination outperforms the auction.

## Where things are

| Path | What |
|---|---|
| `docs/paper/main.tex`, `references.bib` | Paper source (AAAI style). Mirrors Overleaf. |
| `docs/paper/outline.md` | The paper outline |
| `docs/paper/related_work_notes.md` | The 24 cited papers, with overlap/difference notes |
| `docs/design/simulator.md` | Simulator design: clocks, perception, channel, ledger, logging |
| `docs/design/decisions.md` | Dated log of decisions and open questions. **Check before changing a design choice.** |

| `sim/` | Simulator core (M1, LLM-free): world, BFS pathing, perception, beliefs, channel, agent/Policy interface, primitives, loop, logger |
| `allocation/` | Coordination logic per condition (only `scripted.py` test scaffolding so far) |
| `llm/` | Latency model (client and prompts come in M4) |
| `configs/` | Episode configs, `maps/` (ASCII), `missions/` (YAML) |
| `experiments/` | `run_single.py` |
| `tests/` | pytest unit and episode tests |

The full layout, including what is still planned, is at the end of `docs/design/simulator.md`.

## Hard rules (these protect the validity of the results)

1. **All reported times are simulated ticks, never wall-clock.** Wall-clock seconds and tokens are logged as costs only. No result may depend on GPU speed.
2. **LLM calls happen only at decision points** (skill finished or failed, message arrived, task changed, episode start). Skills run over many ticks without an LLM call.
3. **Deliberation costs simulated time** through the latency model in `simulator.md` (tokens mapped to ticks at a fixed reference throughput, not measured speed).
4. **Observations are frozen when a query is issued; actions apply `L` ticks later** after re-checking preconditions against ground truth.
5. **Perception is deterministic code, not the LLM.** Agents act on a belief graph that can go stale. Ground truth is never given to an agent.
6. **One model server, many agents.** Each agent is a separate prompt history on the same loaded model. Never load one model copy per agent.
7. **All conditions share one agent interface and one codebase.** Only the coordination logic differs between conditions.
8. **Distances come from one BFS path-length function** used by the oracle, numeric CBBA, and the text shown to LLMs, so no condition gets better distance information.
9. **Log everything needed to replay a run** (events, every LLM call with its full prompt, messages incl. drops, ledgers, belief changes) and compute metrics from the logs afterwards. Record the seed, config, and model file hash in `run.json`.
10. **CBBA's conflict-free guarantee holds only among connected agents.** Never write code, comments, or paper text that claims it holds across partitions.
11. The `robot-team-sandbox.html` demo (if present) uses a *greedy* auction, not CBBA. It illustrates dynamics only and is not evidence for the paper.

## Working agreements

- **Paper first, code second.** When a design choice changes, update `docs/design/decisions.md` and the matching paper section in the same commit.
- Propose a plan before writing code for the simulator core, the latency model, the ledger, or CBBA.
- Experiment settings go in config files (one per condition/sweep) in the repo, not command-line magic.
- Do not add dependencies without saying why.
- Never invent citations. If a reference is needed that is not in `references.bib`, say so and let the user verify it.

## Code style (the author's standards)

- Keep code legible and non-convoluted. No proxy variables, no excessive logging, no excessive helper functions, no lambdas.
- Handle each `except` branch fully in place (log, then continue) instead of storing a failure message and branching on it later. Avoid nested try/except.
- Helper functions only when necessary, and named cleanly and intuitively.
- Use specific type hints; do not use `Any` where a concrete type is possible.
- No nested closures; never define nested classes.
- Do not pass or return unused dict fields just for the sake of it.
- Declare constants and global parameters in the proper place (a `config` module or config files), not inline.
- Google Python style; Ruff-clean; newline at end of every file; make sure all modules are imported.
- Python 3.11+, dataclasses for plain data.

## Commands

Use the project venv (`python -m venv .venv`, then `.venv/Scripts/python -m pip install -e .[dev]`).

- Tests: `.venv/Scripts/python -m pytest`
- Lint: `.venv/Scripts/ruff check .` and `.venv/Scripts/ruff format --check .`
- One episode: `.venv/Scripts/python -m experiments.run_single configs/m1_smoke.yaml` (logs go to `runs/<name>/seed_<seed>/`)
