#!/usr/bin/env bash
# Packs one launch of an experiment for the trip back to the laptop: every run of that stamp, the
# spec copy, and the cluster logs (Slurm output, server logs, GPU and build info). The response
# cache stays on the cluster.
# Usage, from the repo on a login node: cluster/pack_results.sh <experiment> <stamp>
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 2 ]; then
  echo "usage: cluster/pack_results.sh <experiment> <stamp>" >&2
  exit 2
fi
experiment=$1
stamp=$2
archive=runs/${experiment}_${stamp}.tar.gz
tar -czf "$archive" \
  runs/"$experiment"/*/*/"$stamp" \
  "runs/$experiment/sweep_$stamp.yaml" \
  "runs/$experiment/cluster_$stamp"

finished=$(find runs/"$experiment"/*/*/"$stamp" -name metrics.json | wc -l)
started=$(find runs/"$experiment"/*/*/"$stamp" -name run.json | wc -l)
echo "packed $archive ($(du -h "$archive" | cut -f1)): $finished of $started episodes finished"
echo "on the laptop, from the repo folder:"
echo "  scp $(whoami)@$(hostname -f):$PWD/$archive runs/"
echo "  tar -xzf runs/$(basename "$archive")"
echo "then tell Claude: results in runs/$experiment, stamp $stamp"
