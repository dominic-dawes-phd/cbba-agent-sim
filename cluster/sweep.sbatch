#!/usr/bin/env bash
# One Slurm array task: one GPU, one model server, one shard of the sweep, inside the container.
# Submitted by cluster/submit.sh, which sets the resources and exports SPEC, STAMP, SHARDS,
# REPO_DIR, and SCRATCH_DIR.
set -euo pipefail
echo "node $(hostname), shard $SLURM_ARRAY_TASK_ID of $SHARDS, stamp $STAMP"
apptainer exec --nv \
  --bind "$SCRATCH_DIR/models:/models" \
  --bind "$REPO_DIR:/work" \
  --pwd /work \
  "$SCRATCH_DIR/cbba-agent-sim.sif" \
  bash cluster/run_shard.sh "$SPEC" "$STAMP" "$SLURM_ARRAY_TASK_ID" "$SHARDS"
