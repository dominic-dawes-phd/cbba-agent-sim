#!/usr/bin/env bash
# Runs once every shard of a launch has ended (submitted by cluster/submit.sh with afterany), or by
# hand. Writes runs/<experiment>/cluster_<stamp>/summary.txt (Slurm states, finished and failed
# episodes, server errors, a rerun line), then packs the launch with cluster/pack_results.sh.
# Usage by hand, from the repo on the login node: bash cluster/collect.sh <experiment> <stamp> <array job id>
# cluster/run_local.sh (no Slurm) calls it with the array job id "local".
set -euo pipefail
# Under sbatch this script runs from a spool copy, so the repo comes from REPO_DIR.
cd "${REPO_DIR:-$(dirname "$0")/..}"
source cluster/cluster.env
experiment=${1:-$EXPERIMENT}
stamp=${2:-$STAMP}
array_id=${3:-$ARRAY_ID}
log_dir=runs/$experiment/cluster_$stamp
shopt -s nullglob

{
  if command -v sacct > /dev/null; then
    echo "experiment $experiment, stamp $stamp, array job $array_id on $CLUSTER, collected $(date)"
    echo
    echo "== Slurm: one line per shard (State COMPLETED and ExitCode 0:0 is a clean shard)"
    sacct -M "$CLUSTER" -j "$array_id" -X \
      --format JobID%16,State%12,ExitCode,Elapsed,MaxRSS,NodeList%16,Partition || echo "(sacct failed)"
  else
    # cluster/run_local.sh: no Slurm; each shard's last line is its exit status.
    echo "experiment $experiment, stamp $stamp, run on $(hostname) without Slurm, collected $(date)"
    echo
    echo "== Shards (status 0 is a clean shard)"
    grep -h "ended with status" "$log_dir"/slurm_*.out || echo "(no shard has ended)"
  fi
  for out in "$log_dir"/slurm_*.out; do
    restarts=$(grep -c "^node .* restart [1-9]" "$out" || true)
    if [ "$restarts" -gt 0 ]; then
      echo "$(basename "$out"): requeued $restarts times (preempted or node failure)"
    fi
  done
  echo

  echo "== Episodes"
  run_jsons=(runs/"$experiment"/*/*/"$stamp"/seed_*/run.json)
  metrics=(runs/"$experiment"/*/*/"$stamp"/seed_*/metrics.json)
  echo "${#metrics[@]} of ${#run_jsons[@]} started episodes have metrics.json"
  echo "Shards whose sweep stopped before its summary leave no metrics.json, even for finished episodes."
  failed=()
  for out in "$log_dir"/slurm_*.out; do
    # experiments.sweep ends a shard with "N episodes failed: <env>/<cell>/seed_<n>, ...".
    for label in $(sed -n 's/.*episodes failed: //p' "$out" | tr ',' ' '); do
      failed+=("$label")
    done
  done
  for run_json in "${run_jsons[@]}"; do
    run_dir=$(dirname "$run_json")
    if [ ! -f "$run_dir/metrics.json" ]; then
      # runs/<experiment>/<env>/<cell>/<stamp>/seed_<n> -> <env>/<cell>/seed_<n>
      IFS=/ read -r _ _ env cell _ seed <<< "$run_dir"
      failed+=("$env/$cell/$seed")
    fi
  done
  if [ ${#failed[@]} -gt 0 ]; then
    unique=$(printf '%s\n' "${failed[@]}" | sort -u)
    echo "Failed or unfinished:"
    echo "$unique" | sed 's/^/  /'
    echo "Rerun them: copy the spec, add this line, submit with as many shards as labels:"
    echo "only: [$(echo "$unique" | paste -sd, - | sed 's/,/, /g')]"
  else
    echo "No failed episodes."
  fi
  echo

  echo "== Server logs: count of each known problem (docs/design/cluster.md)"
  for log in "$log_dir"/server_shard*.log; do
    oom=$(grep -ci "out of memory" "$log" || true)
    context=$(grep -c "Context size has been exceeded" "$log" || true)
    errors=$(grep -ci "error" "$log" || true)
    echo "$(basename "$log"): out of memory $oom, context exceeded $context, lines with 'error' $errors"
  done
  echo
  echo "== GPU and build per shard"
  for txt in "$log_dir"/server_shard*.txt; do
    echo "-- $(basename "$txt")"
    cat "$txt"
  done
  echo
  echo "== Python errors in the shard logs (last 20 lines with Traceback or Error)"
  grep -h -E "Traceback|Error" "$log_dir"/slurm_*.out | tail -n 20 || true
} > "$log_dir/summary.txt" 2>&1

cat "$log_dir/summary.txt"
bash cluster/pack_results.sh "$experiment" "$stamp"
