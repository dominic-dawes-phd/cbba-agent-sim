#!/usr/bin/env bash
# Packs one launch of an experiment for the trip back to the laptop: every run of that stamp, the
# spec copy, and the cluster logs (Slurm output, summary.txt, server logs, GPU samples, build info).
# The response cache stays on the cluster. cluster/collect.sh calls this when a launch ends.
# Usage, from the repo on the login node: bash cluster/pack_results.sh <experiment> <stamp>
set -euo pipefail
cd "${REPO_DIR:-$(dirname "$0")/..}"
source cluster/cluster.env

if [ $# -ne 2 ]; then
  echo "usage: cluster/pack_results.sh <experiment> <stamp>" >&2
  exit 2
fi
experiment=$1
stamp=$2
shopt -s nullglob
# A shard that died before its first episode leaves no run folder, so pack whatever exists.
paths=(runs/"$experiment"/*/*/"$stamp" "runs/$experiment/sweep_$stamp.yaml" "runs/$experiment/cluster_$stamp")
existing=()
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    existing+=("$path")
  fi
done
archive=runs/${experiment}_${stamp}.tar.gz
tar -czf "$archive" "${existing[@]}"

finished=(runs/"$experiment"/*/*/"$stamp"/seed_*/metrics.json)
started=(runs/"$experiment"/*/*/"$stamp"/seed_*/run.json)
echo "packed $archive ($(du -h "$archive" | cut -f1)): ${#finished[@]} of ${#started[@]} episodes have metrics"
# On a machine without Slurm (cluster/run_local.sh), the files come from this host instead.
host=$LOGIN_HOST
if ! command -v sbatch > /dev/null; then
  host=$(hostname)
fi
echo "on the laptop, from the repo folder (Git Bash):"
echo "  scp $(id -un)@$host:$PWD/$archive runs/"
echo "  tar -xzf runs/$(basename "$archive")"
echo "then tell Claude: results in runs/$experiment, stamp $stamp"
