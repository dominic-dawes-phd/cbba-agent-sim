#!/usr/bin/env bash
# Runs one shard of a sweep inside the container: starts llama-server on this shard's own port,
# runs the shard against it, then stops the server. Called by cluster/sweep.sbatch:
#   run_shard.sh <spec> <stamp> <shard> <shards>
# Server flags follow docs/design/llm_setup.md.
set -euo pipefail
spec=$1
stamp=$2
shard=$3
shards=$4
cd /work

experiment=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"])' "$spec")
# The spec's model path is the file every episode hashes, so the server loads that same file.
model=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["llm"]["model_path"])' "$spec")
# Two shards may share a multi-GPU node, so each listens on its own port.
port=$((18080 + shard))
log_dir=runs/$experiment/cluster_$stamp
mkdir -p "$log_dir"

flags=(--host 127.0.0.1 --port "$port" -ngl 99 --jinja
  --parallel 6 --kv-unified --ctx-size 32768 -b 512 -ub 256
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --cache-ram 0)
{
  echo "host: $(hostname)"
  echo "gpu: $(nvidia-smi -L | tr '\n' ' ')"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "server: $(llama-server --version 2>&1 | tr '\n' ' ')"
  echo "model: $model"
  echo "flags: ${flags[*]}"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$log_dir/server_shard$shard.txt"

llama-server -m "$model" "${flags[@]}" > "$log_dir/server_shard$shard.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then
    break
  fi
  if ! kill -0 "$server" 2>/dev/null; then
    echo "llama-server exited; see $log_dir/server_shard$shard.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:$port/health" > /dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$log_dir/server_shard$shard.txt"

python3 -m experiments.sweep "$spec" --shard "$shard" --of "$shards" --stamp "$stamp" \
  --endpoint "http://127.0.0.1:$port"
