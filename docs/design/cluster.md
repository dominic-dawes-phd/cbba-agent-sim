# Running sweeps on the university GPU cluster

How to run an experiment on a Slurm cluster with NVIDIA GPUs and Apptainer, without Claude Code, and bring the results back to the laptop for analysis. Model server details are in `llm_setup.md`; the experiment design is in `../DOE.md`.

## How it works

- **The container holds the runtime, not the code.** The image `ghcr.io/dominic-dawes-phd/cbba-agent-sim:b12211` is llama.cpp's official CUDA server (build b12211, the laptop's build) plus Python 3.12, PyYAML, and git. It is built by `.github/workflows/container.yml` from the `Dockerfile`.
- **The code is a `git clone` of this repo**, mounted into the container at `/work`. A code change needs only `git pull`. Every `run.json` records the commit.
- **One Slurm array task = one GPU, one llama-server, one shard of the sweep.** Each episode pins its six agents to the server's six slots, so episodes cannot share a server. The array spreads them across GPUs. Each task's server listens on its own port (18080 + shard).
- **The model file stays on scratch**, mounted at `/models`. It is checked against the laptop's sha256, so `run.json` hashes and response cache keys match.
- **Results come back as one tarball** per launch. Analysis runs on the laptop.

Simulated time never depends on the GPU (hard rules 1 and 3), so cluster and laptop episodes measure the same thing. Replies can still differ between GPUs and llama.cpp builds even at temperature 0, so **never mix laptop and cluster runs in one comparison** (`decisions.md`, Nov 14).

## One-time setup

1. **On GitHub (once):** after the first push, open the Actions tab and check that the `container` workflow ran green. Then open the package (your profile, Packages, `cbba-agent-sim`), go to Package settings, and set its visibility to **Public**. GHCR packages start private even for a public repo, and Apptainer then needs a token.
2. **On a cluster login node:**

   ```bash
   git clone https://github.com/dominic-dawes-phd/cbba-agent-sim.git
   cd cbba-agent-sim
   nano cluster/cluster.env        # partition, account, GPU request, scratch path
   bash cluster/setup.sh           # pulls the image (a few GB) and the model (5 GB) to scratch
   ```

   `sinfo -s` lists partitions. If your cluster loads Apptainer as a module, run `module load apptainer` (or `singularity`) first.
3. **Check the GPU driver** on a GPU node: `srun --partition <gpu partition> --gres gpu:1 --pty nvidia-smi`. The top line must show **CUDA Version 12.8 or higher** (driver 570+). If it is lower, stop and tell Claude; the image needs an older llama.cpp base.

## Every run

1. **Update the code:** `git pull`.
2. **Smoke test** (one kitchen C3 episode, about 10-20 min):

   ```bash
   bash cluster/submit.sh configs/sweeps/cluster_smoke_llm.yaml 1
   ```

   When it finishes, `runs/smoke_llm_cluster/cluster_<stamp>/slurm_0.out` should end with a summary and "Logs in runs/smoke_llm_cluster". The laptop's run of the same episode finished 7 of 10 tasks; expect something close, not identical.
3. **The sweep:**

   ```bash
   bash cluster/submit.sh configs/sweeps/cluster_pilot_llm.yaml 8
   ```

   The second number is how many GPUs (shards) to use. Episodes are dealt to shards round-robin in seed order, so 24 episodes on 8 GPUs gives 3 each.
4. **Watch:** `squeue --me`; `tail -f runs/<experiment>/cluster_<stamp>/slurm_0.out`. Each shard's server log, GPU name, driver, and llama.cpp version are in the same folder (`server_shard<i>.log`, `server_shard<i>.txt`).
5. **Pack:** `bash cluster/pack_results.sh <experiment> <stamp>`. It prints how many episodes finished and the `scp` command to run.

`submit.sh` prints the experiment name and the stamp; keep them.

## Back on the laptop

From the repo folder (Git Bash):

```bash
scp <you>@<cluster host>:<path printed by pack_results.sh> runs/
tar -xzf runs/<experiment>_<stamp>.tar.gz
```

Then tell Claude: "results in runs/<experiment>, stamp <stamp>". The analysis is `python -m experiments.analyze runs/<experiment> --stamp <stamp>` and `python -m experiments.plot`.

## Rerunning failed episodes

A shard lists failed episodes at the end of its `slurm_<i>.out` as `<env>/<cell>/seed_<n>`. Copy the spec, add an `only:` list with those labels, and submit it with as many shards as episodes:

```yaml
only: [house_48x32_s1/greedy_llm/seed_0, house_48x32_s2/llm_only/seed_1]
```

The rerun gets a new stamp; give Claude both stamps.

## When something goes wrong

- **The image does not pull** (`401` or `denied`): the GHCR package is still private (setup step 1).
- **`llama-server exited` in `slurm_<i>.out`:** read `server_shard<i>.log`. "CUDA error: out of memory" means the GPU has under about 8 GB free; ask for a larger GPU type in `GPU_REQUEST`.
- **HTTP 500 "Context size has been exceeded"** in a server log: raise `--ctx-size` in `cluster/run_shard.sh` (it is 32768, sized for six house prompts of up to 4.4k tokens).
- **A task hits its time limit:** raise `TIME` in `cluster.env`, or use more shards.
- **`sha256 mismatch`** from `scripts/fetch_model.sh`: delete the file in `$SCRATCH_DIR/models` and rerun `setup.sh`.

## Files

| File | What |
|---|---|
| `Dockerfile`, `.github/workflows/container.yml` | The runtime image and its build |
| `cluster/cluster.env` | Your cluster's settings: the only file to edit |
| `cluster/setup.sh`, `scripts/fetch_model.sh` | One-time image and model download, with sha256 check (the fetch script is shared with the laptop) |
| `cluster/submit.sh` | Submits a spec as a Slurm array under one stamp |
| `cluster/sweep.sbatch`, `cluster/run_shard.sh` | One array task: start the server, run the shard, stop the server |
| `cluster/pack_results.sh` | Tars one launch for the laptop |
| `configs/sweeps/cluster_*.yaml` | Cluster specs (`llm.model_path: /models/...`) |
