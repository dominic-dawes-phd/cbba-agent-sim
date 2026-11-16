# cbba-agent-sim

Simulator and experiments for the CMSC 601 (UMBC) project *Leaderless Semantic Allocation: Evaluating Consensus-Auction Agent Teaming Under Communication Degradation and Agent Churn*. A team of household robots shares work on a grid map. Each robot's LLM scores tasks, and a leaderless auction (CBBA) decides who does what. The project compares that against LLM-only teams, numeric auctions, greedy claiming, and an oracle, under packet loss, limited radio range, and robots dropping out.

This README is the hands-on guide: how to set up, run, and fix things without an AI assistant. The design lives in `docs/`, and `CLAUDE.md` holds the project's rules and summary.

## Directory map

```text
cbba-agent-sim/
├── README.md                 # this guide
├── CLAUDE.md                 # project summary, hard rules, conditions table (read for the "why")
├── pyproject.toml            # package, version, dependencies and extras, ruff and pytest settings
├── Dockerfile                # cluster runtime image: llama.cpp CUDA server + Python (no code inside)
├── .github/workflows/        # container.yml: GitHub builds the image and publishes it to GHCR
├── sim/                      # the simulator core, no LLM in here
│   ├── world.py, pathing.py  #   grid map, objects, BFS distances (the one distance function)
│   ├── perception.py         #   what each robot can see (deterministic code)
│   ├── beliefs.py            #   each robot's possibly stale picture of the world
│   ├── channel.py            #   radio: packet loss, range, walls
│   ├── primitives.py         #   actions robots can take (move_to, pick, place, ...)
│   ├── loop.py               #   the tick loop that runs one episode
│   ├── ledger.py             #   replicated task ledger shared by gossip
│   ├── logger.py             #   writes every log stream of a run
│   └── config.py             #   episode config dataclasses and constants
├── allocation/               # who does which task: one module per condition
│   ├── scripted.py           #   dev-only fixed assignment (+ the shared coded executor)
│   ├── greedy.py             #   G: nearest unclaimed task
│   ├── oracle.py, oracle_cpsat.py  # C0: central solver with ground truth
│   ├── cbba.py               #   C2 and C3: the consensus auction
│   ├── utility.py            #   C3: asks the LLM for task utilities
│   ├── dialogue.py           #   C1: LLM-only team dialogue
│   └── llm_policy.py         #   the LLM executor wrapped around any allocator
├── llm/                      # talking to the model server
│   ├── client.py             #   OpenAI-compatible HTTP client, retries, response cache use
│   ├── cache.py              #   response cache (one JSON file per reply)
│   ├── prompts.py            #   prompt building and reply parsing
│   └── latency.py            #   tokens -> simulated ticks (never wall-clock)
├── experiments/              # command-line tools (run with python -m experiments.<name>)
│   ├── run_single.py         #   run one episode from a config
│   ├── sweep.py              #   run a whole experiment from a sweep spec
│   ├── console.py            #   HiveSim banner, status checks, progress lines, summaries
│   ├── analyze.py            #   metrics from logs, means with 95% CIs
│   ├── plot.py               #   paper ("hero") figures
│   ├── dev_refresh.py        #   quick check figures for the dev environments
│   ├── generate_mission.py   #   builds the seeded 48x32 houses
│   └── headroom.py           #   does allocation matter on a mission?
├── configs/                  # every experiment setting lives in a file here
│   ├── m1_smoke.yaml         #   tiny kitchen: the fastest episode
│   ├── m2_home*.yaml         #   the medium kitchen, one file per condition / model
│   ├── sweeps/               #   one spec per experiment (which configs, seeds, sweep levels)
│   ├── dev/                  #   dev figure specs (m1_smoke, m2_home, a house)
│   ├── generated/            #   the three generated houses (map, mission, episode configs)
│   ├── generator/            #   settings the houses were generated from
│   ├── maps/, missions/      #   hand-written ASCII maps and task lists
│   └── prompts/              #   LLM system prompts and tool descriptions
├── tests/                    # pytest tests (none needs a GPU or a server)
├── scripts/                  # helper scripts for the laptop (and shared with the cluster)
│   ├── check.sh              #   tests + lint in one command
│   ├── fetch_model.sh        #   download model weights, verify sha256
│   ├── start_server.sh       #   start the local llama.cpp server
│   └── preflight.ps1         #   memory check before long runs (Windows)
├── cluster/                  # running sweeps on UMBC HPCF (chip), or any Slurm cluster
│   ├── cluster.env           #   YOUR cluster settings: the only file to edit (set GROUP)
│   ├── setup.sh              #   one-time, inside a GPU job: image, model, cluster_info.txt
│   ├── submit.sh             #   submit a sweep as a Slurm array + a collect job
│   ├── sweep.sbatch, run_shard.sh  # what each array task runs (server + shard + GPU log)
│   ├── collect.sh            #   after the last shard: summary.txt, then pack
│   └── pack_results.sh       #   tar one launch for the trip back to the laptop
├── docs/
│   ├── CHANGELOG.md          #   what changed in each version
│   ├── DOE.md                #   design of experiments, runs executed so far
│   ├── design/               #   simulator.md, conditions.md, decisions.md, llm_setup.md,
│   │                         #   llm_models.md, cluster.md (runbooks and rationale)
│   └── paper/                #   main.tex, references.bib, outline, related-work notes
└── runs/                     # NOT in git: every run's logs, metrics, figures, the LLM cache
```

## Setup (once)

You need Python 3.11 or newer (`python --version`) and Git. On Windows, use the **Git Bash** terminal for the commands below.

```bash
git clone https://github.com/dominic-dawes-phd/cbba-agent-sim.git
cd cbba-agent-sim
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Linux / macOS
python -m pip install -e ".[dev,analysis,solver]"
bash scripts/check.sh              # tests + lint, about 3 minutes; ends with "all checks passed"
```

- **`-e`** installs the repo in editable mode, so code changes take effect without reinstalling.
- **Extras:** `dev` = pytest and ruff; `analysis` = matplotlib (figures); `solver` = OR-Tools (the oracle C0 and house dev figures).
- **Activate the venv in every new terminal** (`source .venv/Scripts/activate`). After that, `python` means the venv's Python. Without activating, write `.venv/Scripts/python` (Windows) or `.venv/bin/python` (Linux) instead of `python`.

## First runs (no GPU, no model needed)

These use the coded executor (no LLM), so they are fast.

```bash
python -m experiments.run_single configs/m1_smoke.yaml       # one episode, seconds -> runs/m1_smoke/seed_0/
python -m experiments.analyze runs/m1_smoke                  # metrics.json + a summary table
python -m experiments.sweep configs/sweeps/smoke.yaml        # a 6-episode sweep, seconds -> runs/smoke/
python -m experiments.dev_refresh --all                      # check figures, a few minutes -> runs/dev/<env>/figures/
```

The full coded pilot (`configs/sweeps/pilot_coded.yaml`) is about 1,400 episodes on 8 processes and takes hours.

**What a run writes** (`runs/<...>/seed_<n>/`): `run.json` (config, seed, git commit, model hash, version), `metrics.json` (after `analyze`), and one gzipped JSON-lines file per log stream. The streams are `events`, `actions`, `messages`, `beliefs`, `ledgers`, `llm_calls`, `dialogues`, `poses`, and `state`.

- Without `--run-name`, `run_single` overwrites the last run of that config.
- With `--run-name <exp>`, it keeps every launch under `runs/<exp>/<env>/<cell>/<YYYYMMDD-HHMMSS>/`.

## LLM runs on the laptop

LLM conditions (G with the LLM executor, C1, C2, C3) need a model server. The laptop setup is llama.cpp build b12211 on the RTX 4070, with the model in `C:\Users\domin\llm` (`~/llm` in Git Bash).

1. **Install llama.cpp (once):**

   ```bash
   mkdir -p ~/llm && cd ~/llm
   curl -L -o llama.zip https://github.com/ggml-org/llama.cpp/releases/download/b12211/llama-b12211-bin-win-cuda-12.4-x64.zip
   curl -L -o cudart.zip https://github.com/ggml-org/llama.cpp/releases/download/b12211/cudart-llama-bin-win-cuda-12.4-x64.zip
   mkdir -p llama.cpp && cd llama.cpp && unzip -o ../llama.zip && unzip -o ../cudart.zip && cd -
   ```

2. **Download the model weights (once, about 5 GB, checksum verified):**

   ```bash
   bash scripts/fetch_model.sh qwen3-8b ~/llm         # also: qwen3.5-4b, ministral3-8b
   ```

3. **Check memory before a long run (Windows):**

   ```bash
   powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1
   ```

   Fix every FAIL line before a run of more than an hour; each one prints its fix.

4. **Start the server** in its own terminal and leave it open (Ctrl+C stops it):

   ```bash
   bash scripts/start_server.sh                       # http://127.0.0.1:8080
   curl http://127.0.0.1:8080/health                  # from another terminal: {"status":"ok"}
   ```

5. **Run** (in a second terminal, with the venv active):

   ```bash
   python -m experiments.run_single configs/m2_home_llm_cbba.yaml --run-name try_c3    # ~15 min
   python -m experiments.sweep configs/sweeps/pilot_llm.yaml                           # ~1 day!
   python -m experiments.analyze runs/pilot_llm --stamp <YYYYMMDD-HHMMSS>
   python -m experiments.plot runs/pilot_llm figures/pilot_llm --stamp <YYYYMMDD-HHMMSS>
   ```

- Use `experiments.sweep` for anything long: it keeps Windows from sleeping (`run_single` does not).
- **Replies are cached in `runs/llm_cache`,** so rerunning an identical episode replays them instantly.
- **The laptop config files point at `C:/Users/domin/llm/...`.** If your model lives elsewhere, edit `llm.model_path` in the config, or add an `llm:` block to a sweep spec.

## Container and cluster (long sweeps)

A house episode takes about 70 minutes on the laptop, so the paper's sweeps run on UMBC HPCF's `chip` cluster (L40S GPUs). Full runbook, including the account request: `docs/design/cluster.md`.

- **The image** (`Dockerfile`) contains only the runtime: llama.cpp's CUDA server (same build as the laptop), Python 3.12, and PyYAML. GitHub builds it (`.github/workflows/container.yml`) and publishes it as `ghcr.io/dominic-dawes-phd/cbba-agent-sim:b12211`.
- **The code is not in the image.** The cluster uses a `git clone` of this repo, mounted into the container, so after a code change you only `git pull` there.
- **Apptainer** (the cluster's container tool) turns the image into one file (`.sif`) and runs it with the GPU. One Slurm array task = one GPU + one model server + a share of the episodes.

```bash
# once: log in, clone onto the research volume (home is only 500 MB)
ssh <you>@chip.rs.umbc.edu
cd /umbc/rs/<group>/users/<you>
git clone https://github.com/dominic-dawes-phd/cbba-agent-sim.git && cd cbba-agent-sim
nano cluster/cluster.env                                    # GROUP (and MAIL_USER if wanted)
bash cluster/setup.sh          # on the login node it only prints the srun line for a GPU job;
                               # run that, cd back here, rerun setup.sh, send back cluster_info.txt

# every time:
git pull
bash cluster/submit.sh configs/sweeps/cluster_smoke_llm.yaml 1   # smoke test, 1 GPU
bash cluster/submit.sh configs/sweeps/cluster_pilot_llm.yaml 8   # the pilot, 8 GPUs
squeue -M chip-gpu --me                                      # is it running?
cat runs/<experiment>/cluster_<stamp>/summary.txt            # written by the collect job at the end

# back on the laptop, in the repo folder (the exact scp line is at the end of collect.out):
scp <you>@chip.rs.umbc.edu:<repo on chip>/runs/<experiment>_<stamp>.tar.gz runs/
tar -xzf runs/<experiment>_<stamp>.tar.gz
python -m experiments.analyze runs/<experiment> --stamp <stamp>
```

## What the terminal shows (and checking back with tmux)

`run_single` and `sweep` print the HiveSim banner, then:

1. **Status checks**, one `[ OK ]`, `[WARN]`, or `[FAIL]` line each: Python version, git commit (a warning if there are uncommitted changes), free disk under `runs/`, and for LLM runs the GPU (`nvidia-smi`), the model file, the response cache, and the server (`/health`, then its slots and context size). A `FAIL` stops before anything runs, with the fix on the same line (exit code 2).
2. **What runs**: condition (C0-C3, G), mission, tasks, agents, seed, tick budget, channel, churn, model and server, and where the logs go. A sweep shows its cells and shard.
3. **Progress lines**, timestamped, so a glance tells you whether it is alive:
   - every minute during an episode, and at the first and last tick of an LLM episode: `<env>/<cell>/seed_<n> | tick 812/2100 | tasks 9/22 done | online 6/6 | LLM 431 calls (0 cached), 1.14M tokens | 18m02s wall`;
   - in a sweep, one line per finished episode: `[3/24] finished ... at tick 1904 | 3h31m so far, about 24h left`.
4. **A summary**: how it ended, tasks done, the ratio to the oracle (if one ran), LLM calls and tokens, messages, wall time. A sweep adds the finished/failed count and its per-cell table.

Colors appear only in a terminal (set `NO_COLOR=1` to turn them off); files and Slurm logs get plain text. Wall time is shown as a cost only; every result is in simulated ticks.

To leave a long run going and check back later (Linux or WSL; on the cluster, read `slurm_<i>.out` instead):

```bash
tmux new -s hive                                      # a session that survives closing the terminal
python -m experiments.sweep configs/sweeps/pilot_llm.yaml    # (| tee a file to keep a copy; colors then go)
# detach: Ctrl-b then d    come back: tmux attach -t hive    scroll back: Ctrl-b then [    list: tmux ls
```

## Command cheat sheet

Run from the repo root with the venv active.

| Task | Command | Output |
|---|---|---|
| Tests + lint | `bash scripts/check.sh` | terminal |
| Tests only / one file | `python -m pytest -q` / `python -m pytest tests/test_llm.py -q` | terminal |
| Lint | `ruff check .` and `ruff format --check allocation experiments llm sim tests` | terminal |
| Auto-format code | `ruff format allocation experiments llm sim tests` (never on `.`: it rewrites `docs/*.md`) | edits files |
| One episode (scratch) | `python -m experiments.run_single <config.yaml>` | `runs/<name>/seed_<n>/` |
| One episode (kept) | `python -m experiments.run_single <config.yaml> --run-name <exp>` | `runs/<exp>/<env>/<cell>/<stamp>/seed_<n>/` |
| Whole experiment | `python -m experiments.sweep configs/sweeps/<spec>.yaml` | `runs/<experiment>/...`, summary at the end |
| One shard of a sweep | `python -m experiments.sweep <spec> --shard 0 --of 4 --stamp <stamp>` | same, merged by the shared stamp |
| Rerun failed episodes | copy the spec, add `only: [<env>/<cell>/seed_<n>, ...]`, run `sweep` on the copy | new stamp |
| Metrics | `python -m experiments.analyze runs/<exp> [--stamp <stamp>]` | `metrics.json` per run, table |
| Paper figures | `python -m experiments.plot runs/<exp> figures/<exp> [--stamp <stamp>]` | `figures/<exp>/` |
| Dev check figures | `python -m experiments.dev_refresh configs/dev/m2_home.yaml` or `--all` | `runs/dev/<env>/figures/` |
| Generate the houses | `python -m experiments.generate_mission configs/generator/house_48x32.yaml` | `configs/generated/` |
| Oracle on a house | `python -m experiments.run_single configs/generated/house_48x32_s0/episode_oracle.yaml` | `runs/...` |
| Headroom pilot | `python -m experiments.headroom configs/pilot_headroom.yaml` | terminal |
| Peek at a log stream | `zcat runs/m1_smoke/seed_0/events.jsonl.gz \| head` | terminal |
| Load a stream in Python | `python -c "from pathlib import Path; from sim.logger import read_stream; print(read_stream(Path('runs/m1_smoke/seed_0'), 'events')[:3])"` | terminal |
| Memory preflight | `powershell -ExecutionPolicy Bypass -File scripts/preflight.ps1` | PASS / FAIL |
| Start the model server | `bash scripts/start_server.sh` | runs until Ctrl+C |
| GPU use, live | `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1` | terminal |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sim'` (or `yaml`) | venv not active, or not installed | `source .venv/Scripts/activate`, then `python -m pip install -e ".[dev,analysis,solver]"`; run from the repo root |
| `No module named 'matplotlib'` / `'ortools'` | extra not installed | reinstall with `".[dev,analysis,solver]"` |
| `python` opens the Microsoft Store or is the wrong version | Windows app alias | use `py -3.12 -m venv .venv`, then the venv's Python |
| `Connection refused` or `LLMServerError: server unreachable` | the model server is not running, or another port | start `bash scripts/start_server.sh`; check `curl http://127.0.0.1:8080/health` |
| `LLMServerError: server answered 500: ... Context size has been exceeded` | six agents' prompts overflow the server's context | raise `--ctx-size` in `scripts/start_server.sh` (it is 32768) |
| Server log says `CUDA error: out of memory`, or the server dies mid-run | usually host RAM, not VRAM (Windows counts GPU memory against RAM) | run the preflight; Restart Windows (Start > Power > Restart, not Shut down); disable VS Code's Docker extensions |
| RAM keeps dropping during long runs | zombie processes (about 90 KB each), mostly from Docker extension polling | preflight check 3 and 4; disable Docker, Docker DX, Container Tools, Dev Containers |
| A long run stalls for hours | the laptop went to sleep | run via `experiments.sweep` (keeps Windows awake); keep the lid open and plugged in |
| Sweep ends with `N episodes failed: ...` and exit code 1 | those episodes raised; the error is in the log above the list | fix the cause, rerun just those with `only:` in a spec copy |
| `CacheMissError` | `llm.replay_only: true` but the reply was never cached | set `replay_only: false` and run with the server up |
| `$'\r': command not found` running a script | Windows line endings got into a shell script | `sed -i 's/\r$//' <script>`; `.gitattributes` keeps them LF in git |
| `bash: cluster/submit.sh: Permission denied` | the execute bit is missing | run it as `bash cluster/submit.sh ...` |
| Cluster: `apptainer pull` says `denied` / 401 | the GHCR package is private | GitHub > your profile > Packages > cbba-agent-sim > Package settings > Public |
| Cluster: `apptainer not found` from `setup.sh` | you are on the login node (modules work only on compute nodes) | run the `srun ... --pty bash -l` line it prints, then `setup.sh` again |
| Cluster: `Disk quota exceeded`, or login fails | the 500 MB home is full | `du -sh ~/.* ~/*`; move big folders to `/umbc/rs/<group>/users/<you>` |
| Cluster: `llama-server exited` in `slurm_<i>.out` | see `runs/<exp>/cluster_<stamp>/server_shard<i>.log` | driver too old (needs CUDA 12.8+) or GPU too small: `docs/design/cluster.md` |
| Results look different from the laptop's | different GPU or llama.cpp build changes some replies | expected; never mix laptop and cluster runs in one comparison |

## Versions

Every run records the code version (`sim_version`) and git commit in `run.json`. `docs/CHANGELOG.md` says what changed in each version, and whether older results stay comparable (MAJOR = no). `docs/DOE.md` §7 lists every experiment run so far and its headline numbers.
