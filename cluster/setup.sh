#!/usr/bin/env bash
# One-time setup, run inside an interactive GPU job (docs/design/cluster.md): pulls the runtime
# image as an Apptainer .sif, fetches Qwen3-8B, and writes cluster_info.txt (partitions, accounts,
# GPU driver, disk space) to send back. Safe to rerun; it skips what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

if ! command -v apptainer > /dev/null && command -v module > /dev/null; then
  # Lmod's scripts read unset variables.
  set +u
  module load "$APPTAINER_MODULE"
  set -u
fi
if ! command -v apptainer > /dev/null; then
  echo "apptainer not found: modules work only on compute nodes. Start a GPU job first:" >&2
  echo "  srun --cluster=$CLUSTER --account=$ACCOUNT --partition=$PARTITION --gres=$GPU_REQUEST" \
    "--constraint=$GPU_CONSTRAINT --mem=16G --time=2:00:00 --pty bash -l" >&2
  echo "then rerun: bash cluster/setup.sh" >&2
  exit 1
fi

mkdir -p "$DATA_DIR/apptainer_cache" "$DATA_DIR/apptainer_tmp"
# Apptainer's layer cache and build space would otherwise fill the 500 MB home.
export APPTAINER_CACHEDIR=$DATA_DIR/apptainer_cache
export APPTAINER_TMPDIR=$DATA_DIR/apptainer_tmp
sif=$DATA_DIR/cbba-agent-sim.sif
if [ ! -f "$sif" ]; then
  apptainer pull "$sif" "$IMAGE"
fi
echo "image: $sif"
apptainer exec "$sif" python3 --version
apptainer exec --nv "$sif" llama-server --version

bash scripts/fetch_model.sh qwen3-8b "$DATA_DIR/models"

info=$DATA_DIR/cluster_info.txt
{
  echo "== date, host, git"
  date
  hostname
  git rev-parse HEAD
  echo "== GPU partitions: name, time limit, nodes, GRES, features"
  sinfo -M "$CLUSTER" -o "%P %l %D %G %f"
  echo "== your accounts, partitions, QOS"
  sacctmgr -n show assoc user="$USER" format=cluster,account,partition,qos%40
  echo "== QOS limits"
  sacctmgr -n show qos format=name,maxwall,maxtrespu%30,maxjobspu,maxsubmitpu
  echo "== apptainer"
  apptainer --version
  echo "== GPU on this node"
  nvidia-smi
  echo "== disk"
  df -h "$HOME" "$DATA_DIR"
} > "$info" 2>&1 || true
echo "wrote $info (send it back once)"
echo "setup done; next: the smoke run (docs/design/cluster.md)"
