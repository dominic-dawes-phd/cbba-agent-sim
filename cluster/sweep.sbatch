#!/bin/bash -l
# One Slurm array task: one GPU, one model server, one shard of the sweep, inside the container.
# Submitted by cluster/submit.sh, which sets the resources and exports SPEC, STAMP, SHARDS,
# EXPERIMENT, and REPO_DIR. The login shell (-l) defines `module`.
set -euo pipefail
cd "$REPO_DIR"
source cluster/cluster.env
echo "node $(hostname), job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID," \
  "shard $SLURM_ARRAY_TASK_ID of $SHARDS, stamp $STAMP, restart ${SLURM_RESTART_COUNT:-0}, $(date)"

# Lmod's scripts read unset variables. Never load "shared" here: it switches Slurm to chip-cpu.
set +u
module load "$APPTAINER_MODULE"
set -u
# A GPU without native kernels in the image JIT-compiles them into this cache (not the 500 MB home).
export CUDA_CACHE_PATH=$DATA_DIR/cuda_cache
mkdir -p "$CUDA_CACHE_PATH"

status=0
apptainer exec --nv \
  --bind "$DATA_DIR/models:/models" \
  --bind "$REPO_DIR:/work" \
  --pwd /work \
  "$DATA_DIR/cbba-agent-sim.sif" \
  bash cluster/run_shard.sh "$SPEC" "$STAMP" "$SLURM_ARRAY_TASK_ID" "$SHARDS" || status=$?
echo "shard $SLURM_ARRAY_TASK_ID ended with status $status, $(date)"
exit "$status"
