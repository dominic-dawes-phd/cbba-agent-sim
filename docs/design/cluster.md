# Running sweeps on the UMBC HPCF cluster (`chip`)

How to run an experiment on UMBC's `chip` cluster (Slurm, NVIDIA GPUs, Apptainer) without Claude Code, and bring the results and logs back to the laptop for analysis. Model server details are in `llm_setup.md`; the experiment design is in `../DOE.md`. The `chip` facts below come from the HPCF FAQ (umbc.atlassian.net/wiki/spaces/faq, read Nov 16, 2026); the scripts work on another Slurm cluster after editing `cluster/cluster.env`.

## How it works

- **The container holds the runtime, not the code.** The image `ghcr.io/dominic-dawes-phd/cbba-agent-sim:b12211` is llama.cpp's official CUDA server (build b12211, the laptop's build) plus Python 3.12, PyYAML, and git. It is built by `.github/workflows/container.yml` from the `Dockerfile` and is public, so Apptainer pulls it without a token.
- **The code is a `git clone` of this repo**, mounted into the container at `/work`. A code change needs only `git pull`. Every `run.json` records the commit.
- **One Slurm array task = one GPU, one llama-server, one shard of the sweep.** Each episode pins its six agents to the server's six slots, so episodes cannot share a server. The array spreads them across GPUs. Each task's server listens on its own port (18080 + shard), since several tasks can land on one 4-GPU node.
- **After the last shard ends, a small collect job** writes `summary.txt` and packs the launch into one tarball. With `MAIL_USER` set, Slurm emails you when that is done.
- **Analysis runs on the laptop.**

Simulated time never depends on the GPU (hard rules 1 and 3), so cluster and laptop episodes measure the same thing. Replies can still differ between GPU models and llama.cpp builds even at temperature 0, so **never mix laptop and cluster runs, or two GPU models, in one comparison** (`decisions.md`, Nov 14 and Nov 16). `cluster.env` pins the GPU model (`GPU_CONSTRAINT=L40S`).

## What `chip` requires (and how the scripts handle it)

| `chip` fact | Handled by |
|---|---|
| Login host `chip.rs.umbc.edu`, UMBC username and password; Duo prompt off campus unless on the UMBC VPN (GlobalProtect) | You, at login and for `scp` |
| GPUs are their own Slurm cluster: every Slurm command needs `-M chip-gpu` (`squeue`, `sacct`, `scancel` too) | `CLUSTER` in `cluster.env` |
| `--account=pi_<group>` is required | `GROUP` / `ACCOUNT` |
| GPU model is picked with `--constraint` (`L40S`, `RTX_6000`, `RTX_8000`, `H100`, `RTX_2080`); `--gres` is always `gpu:N` | `GPU_CONSTRAINT`, `GPU_REQUEST` |
| Partitions `gpu` (default) and `gpu-general` (no contributor preemption); jobs may be preempted after 3 days and are requeued | `PARTITION`; `TIME=12:00:00`; logs are appended across a requeue and `summary.txt` reports it |
| Home is **500 MB**, and a full home can block login. The research volume is `/umbc/rs/<group>/users/<you>`; `/scratch` is deleted when each job ends | The repo, image, model, caches, and `runs/` all live under `DATA_DIR` on the research volume |
| Modules (incl. `apptainer`) work only on compute nodes; `module load shared` on a GPU node switches Slurm to chip-cpu | `setup.sh` runs inside a GPU job; `sweep.sbatch` loads `apptainer` itself and never `shared` |
| Login nodes: no computation, under 2 GB of memory per user | Only `git`, `sbatch`, `squeue`, `tail`, and the collect step run there |

Not in the HPCF docs, so `setup.sh` records them in `cluster_info.txt`: the chip-gpu QOS limits (GPUs or jobs per user), which partitions hold the L40S nodes, and the real driver version (a sample in the docs shows driver 575, CUDA 12.9; the image needs CUDA 12.8 or newer).

## One-time setup

1. **Get an account.** On hpcf.umbc.edu, open Forms and submit the account request. It needs a sponsoring UMBC faculty member (their group becomes `pi_<name>`), a project title, and an abstract. Accounts are reviewed each fall.
2. **Log in and clone onto the research volume** (not home):

   ```bash
   ssh <you>@chip.rs.umbc.edu
   mkdir -p /umbc/rs/<group>/users/<you> && cd /umbc/rs/<group>/users/<you>
   git clone https://github.com/dominic-dawes-phd/cbba-agent-sim.git
   cd cbba-agent-sim
   nano cluster/cluster.env        # set GROUP (and MAIL_USER if you want an email per launch)
   ```

3. **Run setup inside a short GPU job** (modules do not work on the login node). `bash cluster/setup.sh` on the login node prints this exact line for your settings:

   ```bash
   srun --cluster=chip-gpu --account=pi_<group> --partition=gpu --gres=gpu:1 --constraint=L40S --mem=16G --time=2:00:00 --pty bash -l
   cd /umbc/rs/<group>/users/<you>/cbba-agent-sim
   bash cluster/setup.sh           # pulls the image (a few GB) and the model (5 GB), then writes cluster_info.txt
   exit
   ```

4. **Send back `cluster_info.txt`** (in `DATA_DIR`, printed at the end of setup). Check its `nvidia-smi` block yourself: the top line must show **CUDA Version 12.8 or higher**. If it is lower, stop and tell Claude; the image needs an older llama.cpp base. The partition list shows whether `gpu-general` has L40S nodes (then use it, to avoid preemption) and the QOS block shows how many GPUs you may hold at once.

## Every run

1. **Update the code:** `git pull`.
2. **Smoke test** (one kitchen C3 episode, about 10-20 min of queue plus run):

   ```bash
   bash cluster/submit.sh configs/sweeps/cluster_smoke_llm.yaml 1
   ```

   `submit.sh` prints the experiment name, the stamp, and the job ids; keep them. When it finishes, `runs/smoke_llm_cluster/cluster_<stamp>/summary.txt` should show a COMPLETED shard and "No failed episodes". The laptop's run of the same episode finished 7 of 10 tasks; expect something close, not identical.
3. **The sweep:**

   ```bash
   bash cluster/submit.sh configs/sweeps/cluster_pilot_llm.yaml 8
   ```

   The second number is how many GPUs (shards) to use. Episodes are dealt to shards round-robin in seed order, so 24 episodes on 8 GPUs gives 3 each. Keep it at or below your QOS GPU limit (`cluster_info.txt`); extra tasks just wait in the queue.
4. **Watch:** `squeue -M chip-gpu --me`; `tail -f runs/<experiment>/cluster_<stamp>/slurm_0.out`. Cancel with `scancel -M chip-gpu <job id>`.
5. **When it ends**, the collect job has already written `summary.txt` and the tarball (`runs/<experiment>_<stamp>.tar.gz`). Its last lines (also in `collect.out`) are the exact `scp` and `tar` commands. If the collect job was refused at submit time, run `bash cluster/collect.sh <experiment> <stamp> <array job id>` on the login node.

## What comes back

Everything for one launch is in `runs/<experiment>/cluster_<stamp>/`, and it is all inside the tarball:

| File | What it tells you |
|---|---|
| `summary.txt` | Read this first. Per-shard Slurm state, exit code, run time, memory, node; requeues; episodes with metrics vs started; failed episode labels and a ready `only:` line to rerun them; counts of out-of-memory and context-overflow errors per server log; GPU, driver, CUDA, llama.cpp version per shard; the last Python errors |
| `slurm_<i>.out` | Everything shard `i` printed (the sweep's progress and summary) |
| `server_shard<i>.log` | llama-server's own log |
| `server_shard<i>.txt` | Host, GPU, driver, CUDA, llama.cpp version, model file, server flags, git commit, VRAM after loading |
| `gpu_shard<i>.csv` | GPU utilization and memory, one sample a minute (cost section) |
| `collect.out` | The collect job's output, ending with the `scp` command |

The episode logs themselves sit next to them as usual (`runs/<experiment>/<env>/<cell>/<stamp>/seed_<n>/`). The response cache stays on the cluster.

## Back on the laptop

From the repo folder (Git Bash), using the command `summary`/`collect.out` printed:

```bash
scp <you>@chip.rs.umbc.edu:/umbc/rs/<group>/users/<you>/cbba-agent-sim/runs/<experiment>_<stamp>.tar.gz runs/
tar -xzf runs/<experiment>_<stamp>.tar.gz
```

Then tell Claude: "results in runs/<experiment>, stamp <stamp>". The analysis is `python -m experiments.analyze runs/<experiment> --stamp <stamp>` and `python -m experiments.plot`.

## Rerunning failed episodes

`summary.txt` ends its episode block with a line like:

```yaml
only: [house_48x32_s1/greedy_llm/seed_0, house_48x32_s2/llm_only/seed_1]
```

Copy the spec, add that line, and submit the copy with as many shards as labels. The rerun gets a new stamp; give Claude both stamps.

## When something goes wrong

- **`setup.sh` says "apptainer not found":** you are on the login node; start the GPU job it prints.
- **`sbatch: error: invalid account` or `invalid partition`:** check `GROUP` in `cluster.env` against the accounts block in `cluster_info.txt`.
- **Jobs stay pending (`squeue` reason `QOSMaxGRESPerUser` or `Resources`):** fewer shards, or a second GPU model in a separate experiment; never mix models in one comparison.
- **`Disk quota exceeded` or you cannot log in:** home is full. Something wrote to `~` (look at `du -sh ~/.* ~/*`); move it to the research volume and link it back.
- **`llama-server exited` in `slurm_<i>.out`:** read `server_shard<i>.log`. "CUDA error: out of memory" should not happen on 48 GB cards; "no kernel image" or "unsupported PTX" means the driver is older than the image needs (CUDA 12.8).
- **HTTP 500 "Context size has been exceeded"** (counted in `summary.txt`): raise `--ctx-size` in `cluster/run_shard.sh` (it is 32768, sized for six house prompts of up to 4.4k tokens).
- **A shard hit its time limit** (`TIMEOUT` in `summary.txt`): raise `TIME` in `cluster.env`, or use more shards.
- **A shard was requeued:** it reran its episodes under the same stamp; finished LLM calls replay from the response cache. Nothing to do unless it happens often (then try `PARTITION=gpu-general`).
- **`sha256 mismatch`** from `scripts/fetch_model.sh`: delete the file in `$DATA_DIR/models` and rerun `setup.sh`.

## Files

| File | What |
|---|---|
| `Dockerfile`, `.github/workflows/container.yml` | The runtime image and its build |
| `cluster/cluster.env` | Your cluster's settings: the only file to edit |
| `cluster/setup.sh`, `scripts/fetch_model.sh` | One-time image and model download, with sha256 check, and `cluster_info.txt` |
| `cluster/submit.sh` | Submits a spec as a Slurm array under one stamp, plus the collect job |
| `cluster/sweep.sbatch`, `cluster/run_shard.sh` | One array task: start the server and GPU sampler, run the shard, stop them |
| `cluster/collect.sh` | After the last shard: `summary.txt`, then `pack_results.sh` |
| `cluster/pack_results.sh` | Tars one launch for the laptop |
| `configs/sweeps/cluster_*.yaml` | Cluster specs (`llm.model_path: /models/...`) |
