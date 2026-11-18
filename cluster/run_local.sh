#!/usr/bin/env bash
# Runs a sweep spec on a GPU machine without Slurm (docs/design/cluster.md, "A GPU machine without
# Slurm"): one shard per GPU, each in the public image with its own llama-server
# (cluster/run_shard.sh, as under Slurm), then cluster/collect.sh writes summary.txt and the tarball.
#
# Usage, from the repo (in tmux for anything long):
#   bash cluster/run_local.sh <spec> [gpus]
#   e.g. bash cluster/run_local.sh configs/sweeps/cluster_smoke_llm.yaml
#        bash cluster/run_local.sh configs/sweeps/cluster_pilot_llm.yaml 0,1,2,3
# gpus:      comma-separated GPU indices, one shard each (default 0).
# MODEL_DIR: holds the .gguf files (default ~/models; fill it with scripts/fetch_model.sh).
# RUNTIME:   docker or apptainer (default docker when installed). Apptainer keeps its image in SIF.
# Ctrl+C stops every shard; the episodes finished so far are still collected.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: bash cluster/run_local.sh <spec> [gpus, e.g. 0,1]" >&2
  exit 2
fi
spec=$1
IFS=, read -r -a gpus <<< "${2:-0}"
shards=${#gpus[@]}
model_dir=${MODEL_DIR:-$HOME/models}
sif=${SIF:-$HOME/cbba-agent-sim.sif}
if [ -z "${RUNTIME:-}" ]; then
  RUNTIME=apptainer
  if command -v docker > /dev/null; then
    RUNTIME=docker
  fi
fi

experiment=$(sed -n 's/^experiment: *//p' "$spec")
# The spec's model path is where the container sees the file; on this machine it is in MODEL_DIR.
model=$(sed -n 's|^  model_path: */models/||p' "$spec")
if [ -z "$model" ]; then
  echo "$spec sets no llm.model_path under /models/ (copy configs/sweeps/cluster_smoke_llm.yaml)" >&2
  exit 2
fi
if [ ! -f "$model_dir/$model" ]; then
  echo "model not found: $model_dir/$model (bash scripts/fetch_model.sh <model> $model_dir)" >&2
  exit 1
fi
if [ "$RUNTIME" = apptainer ] && [ ! -f "$sif" ]; then
  apptainer pull "$sif" "$IMAGE"
fi

stamp=$(date +%Y%m%d-%H%M%S)
log_dir=runs/$experiment/cluster_$stamp
mkdir -p "$log_dir"
echo "experiment $experiment, stamp $stamp: $shards shard(s) on GPU(s) ${gpus[*]} with $RUNTIME"
echo "logs: $log_dir/slurm_<i>.out (each line below is prefixed with its shard)"

# Killing a shard's sweep (and its worker) ends run_shard.sh, whose EXIT trap stops its server.
stop_shards() {
  echo "stopping the shards of stamp $stamp"
  pkill -f "experiments.sweep .* --stamp $stamp" || true
}
trap stop_shards INT TERM

for shard in "${!gpus[@]}"; do
  gpu=${gpus[$shard]}
  # The Slurm file name, so cluster/collect.sh reads it unchanged.
  out=$log_dir/slurm_$shard.out
  {
    echo "node $(hostname), gpu $gpu, shard $shard of $shards, stamp $stamp, restart 0, $(date)"
    status=0
    if [ "$RUNTIME" = docker ]; then
      docker run --rm --name "cbba_${stamp}_$shard" --gpus "device=$gpu" \
        --user "$(id -u):$(id -g)" -e CUDA_CACHE_PATH=/work/runs/cuda_cache \
        -v "$model_dir:/models" -v "$PWD:/work" -w /work \
        "${IMAGE#docker://}" bash cluster/run_shard.sh "$spec" "$stamp" "$shard" "$shards"
    else
      # --nv shows every GPU to nvidia-smi; CUDA (the server) uses only this one.
      APPTAINERENV_CUDA_VISIBLE_DEVICES=$gpu APPTAINERENV_CUDA_CACHE_PATH=/work/runs/cuda_cache \
        apptainer exec --nv --bind "$model_dir:/models" --bind "$PWD:/work" --pwd /work \
        "$sif" bash cluster/run_shard.sh "$spec" "$stamp" "$shard" "$shards"
    fi || status=$?
    echo "shard $shard ended with status $status, $(date)"
  } 2>&1 | tee "$out" | sed -u "s/^/[shard $shard] /" &
done
# A trapped Ctrl+C interrupts wait; wait again until every shard has stopped.
until wait; do
  :
done

bash cluster/collect.sh "$experiment" "$stamp" local
if grep -q "ended with status [1-9]" "$log_dir"/slurm_*.out; then
  exit 1
fi
