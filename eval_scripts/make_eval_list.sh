#!/usr/bin/env bash
# Build an eval-checkpoint list from the runs that actually finished on disk.
# The committed eval_checkpoints_*.txt files are best-guess; run THIS on Jean-Zay
# (from the repo root) to regenerate the authoritative list -- it picks up exactly
# the run dirs that exist, whatever the env count / contact-weight formatting / which
# array tasks completed, and writes the "latest" checkpoint (nn/<hand>.pth) per run.
#
# Usage:
#   bash eval_scripts/make_eval_list.sh                              # graphffcon -> eval_checkpoints_graph_ffcon.txt
#   bash eval_scripts/make_eval_list.sh graphcon eval_scripts/eval_checkpoints_graph_con.txt
#   bash eval_scripts/make_eval_list.sh '*' eval_scripts/eval_checkpoints_all.txt
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERN="${1:-graphffcon}"
OUT="${2:-eval_scripts/eval_checkpoints_graph_ffcon.txt}"

: > "$OUT"
for run_dir in logs/rl_games/*/*"${PATTERN}"*/; do
  [[ -d "$run_dir" ]] || continue
  hand="$(basename "$(dirname "$run_dir")")"   # logs/rl_games/<hand>/<run>/
  pth="${run_dir}nn/${hand}.pth"
  [[ -f "$pth" ]] && echo "$pth" >> "$OUT"
done
sort -o "$OUT" "$OUT"

n=$(wc -l < "$OUT")
echo "wrote $n checkpoint(s) to $OUT"
cat "$OUT"
echo "-> set the eval array to 1-$n (e.g. sbatch --array=1-$n eval_scripts/eval_array_graph_ffcon.slurm)"
