#!/usr/bin/env bash
# One-time setup on a login node (docs/design/cluster.md): pulls the runtime image to scratch as
# an Apptainer .sif and fetches Qwen3-8B. Safe to rerun; it skips what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

mkdir -p "$SCRATCH_DIR"
# Apptainer's layer cache would otherwise fill $HOME.
export APPTAINER_CACHEDIR=$SCRATCH_DIR/apptainer_cache
sif=$SCRATCH_DIR/cbba-agent-sim.sif
if [ ! -f "$sif" ]; then
  apptainer pull "$sif" "$IMAGE"
fi
echo "image: $sif"
apptainer exec "$sif" python3 --version
apptainer exec "$sif" llama-server --version

cluster/fetch_model.sh qwen3-8b
echo "setup done; next: a smoke run (docs/design/cluster.md)"
