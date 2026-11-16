# CLAUDE.md: Leaderless Semantic Allocation (CMSC 601 project)

Read this file first every session. Details live in `docs/`; do not duplicate them here.

## What this project is

Course project for CMSC 601 (UMBC), due **Dec 17, 2026**. Author: Dominic Dawes.

Paper title: *Leaderless Semantic Allocation: Evaluating Consensus-Auction Agent Teaming Under Communication Degradation and Agent Churn.*

**Research questions.** If each robot's LLM only supplies task *utilities*, and a leaderless consensus auction (CBBA) does the allocation: (1) does the auction keep LLM teams coordinated as missions scale (tasks `m`, difficulty, team size `n`); (2) do LLM utilities beat hand-designed numeric ones; (3) does the team stay coordinated, and recover faster than numeric auctions, when communication degrades and agents drop out and rejoin? What does that added structure cost compared with unconstrained LLM coordination?

**Architecture in one paragraph.** Each agent's LLM turns a task's natural-language description plus the agent's capability card into a bundle-independent utility `u_ij` (scored once per agent and task, cached; richer inputs are future work, see `docs/design/decisions.md`). A consensus auction (CBBA, and MURDOCH as a second family) resolves the allocation with no leader. Each agent executes its bundle by LLM tool calls to action primitives. Allocation and execution state live in a replicated per-agent **ledger** (monotone task status, lease-based ownership, gossip sync). Narrative action history is compacted after N entries; the ledger is never compacted.

## The five conditions (keep these names and this order)

Canonical definitions (layer model, allowed LLM tools, no-bleed rules, C1 round protocol): `docs/design/conditions.md`.

| ID | Condition | Allocation | Ledger | Executor | LLM role |
|---|---|---|---|---|---|
| C0 | Oracle | Central solver with ground-truth feasibility | n/a | coded | none |
| C1 | LLM-only | RoCo-style lockstep dialogue per comm component | none | llm | negotiates + executes |
| C2 | Numeric CBBA | CBBA with travel-cost utilities | yes | llm | executes only |
| C3 | LLM+CBBA | CBBA with LLM utilities (LLM gives `u_ij` + status reports; code bids) | yes | llm | utilities + executes |
| C4 | LLM+MURDOCH | Single-item auctions with LLM utilities | yes | llm | utilities + executes |

Each runs under perfect communication, then sweeps: mission scale (tasks `m`, task difficulty, team size `n`; planned, not built), packet loss `p`, communication range `r`, and `k`-agent dropout/rejoin (memory kept vs wiped).

Baselines and ablations (added Oct 27, 2026; see `docs/design/decisions.md`):

| ID | Condition | What it isolates |
|---|---|---|
| G | Greedy: nearest unclaimed task, broadcast claim, no consensus (LLM executor in paper runs) | Any coordination vs consensus |
| L | Elected-leader LLM: one agent elected per mission assigns all tasks | Leaderless vs elected leader (DynaHMRC-style; also stands in for a central orchestrator) |
| C1+ledger, C3-noledger | C1 with the ledger; C3 without it (2x2 with C1 and C3) | Consensus vs ledger, for H3's attribution |
| R | Random allocation (LLM executor in paper runs) | Sanity floor |

Built so far: `scripted` (dev only, never in the paper: fixed or round-robin assignment), `G`, and `C0` (best known plan with a proven bound in its planning model, replayed in the sim), all with the coded executor. With the LLM executor on one llama.cpp server (`docs/design/llm_setup.md`): G (`greedy`, `executor: llm`), C1 (`llm_only`), C2 (`numeric_cbba`), and C3 (`llm_cbba`), with the replicated ledger (`sim/ledger.py`) and CBBA. C4, L, R, and the ablations are not built. C0 is the one documented exception to hard rule 5: it starts with ground truth. An episode config's `condition:` names which one runs. Paper runs (generated houses) use `task_visibility: discovered` (agents start knowing no task) and seeded uniform spawns; the hand-written kitchens keep posted tasks and mission spawns.

## Hypotheses (do not reword without updating the abstract in `docs/paper/main.tex`)

- **H1 (structure).** As `m`, task difficulty, and `n` grow, LLM-only teams degrade through duplicated, orphaned, and conflicting work, while auction-based teams do not.
- **H2 (reasoning).** LLM utilities achieve higher utility relative to the oracle (`makespan_vs_oracle`) than numeric utilities (C3 vs C2), at a measurable token cost.
- **H3 (resilience).** Under `p`, `r`, and churn, LLM-utility auctions degrade less and recover faster than numeric auctions, beyond what the replicated ledger alone provides (the ledger ablation tests this). Resilient or emergent behavior is hoped for and claimed only if the data show it. There is no threshold hypothesis; the packet-loss sweep is a swept variable.

`rho` (language-only constraints) was removed Nov 10, 2026; do not reintroduce it.

## Where things are

| Path | What |
|---|---|
| `docs/paper/main.tex`, `references.bib` | Paper source (AAAI style). Mirrors Overleaf. |
| `docs/paper/outline.md` | The paper outline |
| `docs/paper/related_work_notes.md` | The 24 cited papers, with overlap/difference notes |
| `docs/design/simulator.md` | Simulator design: clocks, perception, channel, ledger, logging |
| `docs/design/decisions.md` | Dated log of decisions and open questions. **Check before changing a design choice.** |
| `docs/design/llm_setup.md`, `llm_models.md` | Model server setup, memory checks, GPU monitoring; model candidates and the bake-off |
| `docs/design/cluster.md` | Runbook for sweeps on UMBC HPCF `chip` (Apptainer + Slurm, L40S), without Claude Code; a collect job writes `summary.txt` and the tarball |
| `docs/design/conditions.md` | Canonical condition definitions: layer model, allowed LLM tools, no-bleed rules, C1 round protocol, comparison map |
| `docs/CHANGELOG.md` | Versioned log of what changed (`MAJOR.MINOR.PATCH`; the version is recorded in every `run.json`) |
| `docs/DOE.md` | Design of experiments: estimands, factors and levels, responses, analysis plan, run budget, runs executed |
| `sim/` | Simulator core (LLM-free): world, BFS pathing, perception, beliefs, channel, agent/Policy interface (incl. the per-tick `on_tick` coordination hook), primitives, loop, logger, motion, triggers, exploration (coverage maps, frontier `explore`/`search`, coverage sharing), `ledger.py` (the replicated ledger) |
| `allocation/` | Coordination logic per condition: `scripted.py` (test scaffolding and the shared scripted executor), `greedy.py` (baseline G), `oracle.py` and `oracle_cpsat.py` (C0), `llm_policy.py` (the LLM executor wrapped around an allocator), `cbba.py` (C2, C3), `utility.py` (C3's utility call), `dialogue.py` (C1) |
| `llm/` | Latency model, OpenAI-compatible client (`client.py`), response cache, prompt building; prompt texts and tool descriptions are in `configs/prompts/` |
| `configs/` | Episode configs, `maps/` (ASCII), `missions/` (YAML) |
| `experiments/` | `run_single.py`, `analyze.py` (metrics from logs, bootstrap CIs), `headroom.py` (does allocation matter on a mission?), `generate_mission.py` (seeded 48x32 houses), `plot.py` (hero figures), `plot_episode.py` (per-episode views), `dev_refresh.py` (dev figures), `sweep.py` (a whole experiment from one spec), `console.py` (HiveSim banner, status checks, progress lines, summaries; display only) |
| `configs/dev/` | One dev figure spec per environment: its episode configs, seeds, and sweep levels |
| `configs/sweeps/` | One spec per experiment: episode configs (with `{env}`), environments, seeds, sweep levels, workers |
| `configs/generator/`, `configs/generated/` | Generator settings; generated houses (map, mission, greedy and scripted episode configs) |
| `tests/` | pytest unit and episode tests |
| `cluster/`, `Dockerfile`, `.github/workflows/container.yml` | Cluster scripts (`cluster.env` is the only file to edit), the runtime image (llama.cpp CUDA server + Python; the code is a bind-mounted git checkout), and its GHCR build |
| `scripts/` | Helper scripts: `check.sh` (tests + lint), `fetch_model.sh` (weights with sha256 check; shared with the cluster), `start_server.sh` (laptop llama-server), `preflight.ps1` (memory gate) |
| `README.md` | The hands-on guide for running everything by hand (setup, runs, cluster, cheat sheet, troubleshooting). Keep it in step when commands change. |

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
- **Changelog after every implementation pass** (auto mode or user-confirmed): add a `docs/CHANGELOG.md` entry of 3-6 bullets, bump the version per the convention at the top of that file, and set the same version in `pyproject.toml` (PEP 440 form, so `1.0.01` is written `1.0.1`).
- **Keep `docs/DOE.md` current.** In the same pass, update it whenever factors, levels, responses, conditions, the analysis plan, or executed runs change, and add a line to its DOE change log.
- **Refresh the `m1_smoke` and `m2_home` dev figures after each pass that touches `sim/`, `allocation/`, or `experiments/`,** and look at them. They are checks, never results.
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
- Tests + lint in one go: `bash scripts/check.sh`. Before a long LLM run: `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1`; start the server with `bash scripts/start_server.sh`; weights via `bash scripts/fetch_model.sh qwen3-8b ~/llm`.
- Lint: `.venv/Scripts/ruff check .` and `.venv/Scripts/ruff format --check .`
- One episode: `.venv/Scripts/python -m experiments.run_single configs/m1_smoke.yaml` (logs go to `runs/<name>/seed_<seed>/`, replacing the last run)
- One episode to keep: add `--run-name <exp>` (logs go to `runs/<exp>/<env>/<cell>/<YYYYMMDD-HHMMSS>/seed_<seed>/`; `analyze` and `plot` take `--stamp` to select one launch)
- A whole experiment: `.venv/Scripts/python -m experiments.sweep configs/sweeps/pilot_coded.yaml` (every cell and seed in parallel under one stamp; logs in `runs/<experiment>/<env>/<cell>/<stamp>/seed_<n>/`, the spec copied to `runs/<experiment>/sweep_<stamp>.yaml`, summary printed at the end)
- Dev figures: `.venv/Scripts/python -m experiments.dev_refresh configs/dev/m2_home.yaml`, or `--all` (needs the `[analysis]` extra, and `[solver]` for the houses; writes `runs/dev/<env>/figures/` with fixed names)
- Metrics: `.venv/Scripts/python -m experiments.analyze runs/<name>` (writes `metrics.json` per run, prints means with 95% CIs)
- Headroom pilot: `.venv/Scripts/python -m experiments.headroom configs/pilot_headroom.yaml` (generated houses: `configs/pilot_headroom_houses.yaml`)
- Generate houses: `.venv/Scripts/python -m experiments.generate_mission configs/generator/house_48x32.yaml`
- Oracle run: `.venv/Scripts/python -m experiments.run_single configs/generated/house_48x32_s0/episode_oracle.yaml` (needs the `[solver]` extra for CP-SAT: `pip install -e .[dev,solver]`)
- LLM run: start `llama-server` as in `docs/design/llm_setup.md`, then `.venv/Scripts/python -m experiments.run_single configs/m2_home_llm_cbba.yaml` (also `m2_home_numeric_cbba`, `m2_home_llm_only`, `m2_home_greedy_llm`; houses: `episode_{greedy_llm,numeric_cbba,llm_cbba,llm_only}.yaml`). Replies are cached in `runs/llm_cache`. Tests never need the server (`tests/fake_llm.py`).
- Hero figures: `.venv/Scripts/python -m experiments.plot runs/<sweep> figures/<sweep>` (needs the `[analysis]` extra: `pip install -e .[dev,analysis]`)
- Cluster sweep (on the cluster, see `docs/design/cluster.md`): `bash cluster/submit.sh configs/sweeps/cluster_pilot_llm.yaml 8`, and the collect job writes `runs/<experiment>/cluster_<stamp>/summary.txt` and the tarball (by hand: `bash cluster/collect.sh <experiment> <stamp> <array job id>`); on the laptop, extract the tarball in the repo and analyze with `--stamp`. Shards of one launch share its stamp (`experiments.sweep --shard I --of N --stamp S --endpoint URL`).
- Format only Python files (`ruff format` on `*.py`); on `.` it rewrites the code blocks in `docs/*.md`.
