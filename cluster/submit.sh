#!/usr/bin/env bash
# Submits a sweep as a Slurm array, one shard per GPU, all under one stamp, plus one collect job
# that runs after every shard has ended and packs the results.
# Usage, from the repo on the login node: bash cluster/submit.sh configs/sweeps/<spec>.yaml <shards>
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env

if [ $# -ne 2 ]; then
  echo "usage: cluster/submit.sh <spec> <shards>" >&2
  exit 2
fi
if [ "$GROUP" = CHANGEME ]; then
  echo "set GROUP in cluster/cluster.env first" >&2
  exit 2
fi
spec=$1
shards=$2
experiment=$(sed -n 's/^experiment: *//p' "$spec")
# The same format as sim.config.RUN_STAMP_FORMAT.
stamp=$(date +%Y%m%d-%H%M%S)
log_dir=runs/$experiment/cluster_$stamp
mkdir -p "$log_dir"

mail=()
if [ -n "$MAIL_USER" ]; then
  mail=(--mail-type END,FAIL --mail-user "$MAIL_USER")
fi
common=(--cluster "$CLUSTER" --account "$ACCOUNT" --partition "$PARTITION")
exports="ALL,SPEC=$spec,STAMP=$stamp,SHARDS=$shards,EXPERIMENT=$experiment,REPO_DIR=$PWD"

# --open-mode append keeps the first attempt's log if a preempted task is requeued.
# shellcheck disable=SC2086  # EXTRA_SBATCH is split on spaces on purpose.
array=$(sbatch --parsable --job-name "cbba-$experiment" --array "0-$((shards - 1))" \
  "${common[@]}" --gres "$GPU_REQUEST" --constraint "$GPU_CONSTRAINT" \
  --cpus-per-task "$CPUS" --mem "$MEM" --time "$TIME" $EXTRA_SBATCH \
  --output "$log_dir/slurm_%a.out" --open-mode append \
  --export "$exports" \
  cluster/sweep.sbatch)
# sbatch -M answers "<job id>;<cluster>".
array_id=${array%%;*}
echo "experiment $experiment, stamp $stamp, $shards shards: array job $array_id on $CLUSTER"

if collect=$(sbatch --parsable --job-name "cbba-collect-$experiment" \
  "${common[@]}" --gres gpu:0 --cpus-per-task 1 --mem 1G --time 00:20:00 \
  --dependency "afterany:$array_id" ${mail[@]+"${mail[@]}"} \
  --output "$log_dir/collect.out" \
  --export "$exports,ARRAY_ID=$array_id" \
  cluster/collect.sh); then
  echo "collect job ${collect%%;*} runs when every shard has ended; it writes $log_dir/summary.txt and the tarball"
else
  echo "the collect job was refused; when the shards end, run by hand:" >&2
  echo "  bash cluster/collect.sh $experiment $stamp $array_id" >&2
fi
echo "watch:   squeue -M $CLUSTER --me    and    tail -f $log_dir/slurm_0.out"
echo "cancel:  scancel -M $CLUSTER $array_id"
