#!/usr/bin/env bash
# Submits a sweep as a Slurm array, one shard per GPU, all under one stamp.
# Usage, from the repo on a login node: cluster/submit.sh configs/sweeps/<spec>.yaml <shards>
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

if [ $# -ne 2 ]; then
  echo "usage: cluster/submit.sh <spec> <shards>" >&2
  exit 2
fi
spec=$1
shards=$2
experiment=$(sed -n 's/^experiment: *//p' "$spec")
# The same format as sim.config.RUN_STAMP_FORMAT.
stamp=$(date +%Y%m%d-%H%M%S)
log_dir=runs/$experiment/cluster_$stamp
mkdir -p "$log_dir"

account=()
if [ -n "$ACCOUNT" ]; then
  account=(--account "$ACCOUNT")
fi
# shellcheck disable=SC2086  # EXTRA_SBATCH is split on spaces on purpose.
sbatch --job-name "cbba-$experiment" --array "0-$((shards - 1))" \
  --partition "$PARTITION" ${account[@]+"${account[@]}"} --gres "$GPU_REQUEST" \
  --cpus-per-task "$CPUS" --mem "$MEM" --time "$TIME" $EXTRA_SBATCH \
  --output "$log_dir/slurm_%a.out" \
  --export "ALL,SPEC=$spec,STAMP=$stamp,SHARDS=$shards,REPO_DIR=$PWD,SCRATCH_DIR=$SCRATCH_DIR" \
  cluster/sweep.sbatch

echo "experiment $experiment, stamp $stamp, $shards shards"
echo "watch:  squeue --me    and    tail -f $log_dir/slurm_0.out"
echo "after:  cluster/pack_results.sh $experiment $stamp"
