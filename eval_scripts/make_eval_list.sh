#!/usr/bin/env bash
# Build an eval-checkpoint list from the runs that actually finished on disk.
# The committed eval_checkpoints_*.txt files are best-guess; run THIS on Jean-Zay
# (from the repo root) to regenerate the authoritative list -- it picks up exactly
# the run dirs that exist, whatever the env count / contact-weight formatting / which
# array tasks completed.
#
# For every run it emits the "latest" checkpoint (nn/<hand>.pth) plus every
# intermediate epoch checkpoint whose epoch is a multiple of 1000 (ep1000, ep2000,
# ..., ep5000, ...), skipping the half-thousand ones (ep500, ep1500, ep3500, ...).
#
# NB: the glob is a substring match, so PATTERN=graphcon matches graphcon* runs
# WITHOUT matching graphffcon (the "ff" breaks the substring). Set EXCLUDE to skip
# run dirs whose path matches a regex, e.g. EXCLUDE=j17 to drop old graphcon_j178563
# runs in favour of the latest ones.
#
# Usage:
#   bash eval_scripts/make_eval_list.sh                              # glob graphffcon -> eval_checkpoints_graph_ffcon.txt
#   EXCLUDE=j17 bash eval_scripts/make_eval_list.sh graphcon eval_scripts/eval_checkpoints_graph_con.txt
#   bash eval_scripts/make_eval_list.sh '*' eval_scripts/eval_checkpoints_all.txt
#   # Augment an existing list in place: keep exactly its runs, (re)add epoch checkpoints
#   bash eval_scripts/make_eval_list.sh --from-list eval_scripts/eval_checkpoints_graph_ffcon.txt
#   bash eval_scripts/make_eval_list.sh --from-list <in.txt> <out.txt>
set -uo pipefail
cd "$(dirname "$0")/.."

EXCLUDE="${EXCLUDE:-}"          # optional regex; matching run dirs are skipped (glob mode)

# Print one run's checkpoints to stdout: the latest nn/<hand>.pth, then every
# epoch checkpoint whose epoch is a multiple of 1000.
emit_run() {
  local run_dir="${1%/}"                         # logs/rl_games/<hand>/<run>
  local hand; hand="$(basename "$(dirname "$run_dir")")"
  local latest="${run_dir}/nn/${hand}.pth"
  [[ -f "$latest" ]] && echo "$latest"
  local ck base ep
  declare -A pick=()                             # one checkpoint per epoch (multiple of 1000)
  for ck in "${run_dir}/nn/last_${hand}_ep_"*"_rew_"*.pth; do
    [[ -f "$ck" ]] || continue                   # no matches -> glob stays literal
    base="$(basename "$ck")"                      # last_<hand>_ep_<N>_rew_<r>.pth
    ep="${base#*_ep_}"; ep="${ep%%_rew_*}"
    [[ "$ep" =~ ^[0-9]+$ ]] || continue
    (( ep % 1000 == 0 )) || continue
    # some epochs are saved twice (e.g. ..._rew_5.0.pth and a ..._rew__5.0_.pth
    # final variant); keep one file per epoch, preferring the canonical name.
    if [[ -z "${pick[$ep]:-}" || ( "${pick[$ep]}" == *_rew__* && "$ck" != *_rew__* ) ]]; then
      pick[$ep]="$ck"
    fi
  done
  local e
  for e in $(printf '%s\n' "${!pick[@]}" | sort -n); do echo "${pick[$e]}"; done
}

if [[ "${1:-}" == "--from-list" ]]; then
  IN="${2:?usage: make_eval_list.sh --from-list <in.txt> [out.txt]}"
  OUT="${3:-$IN}"
  : > "$OUT.tmp"
  # Unique run dirs referenced by the input list (strip the trailing /nn/*.pth).
  while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] && emit_run "$run_dir" >> "$OUT.tmp"
  done < <(sed -n 's#/nn/[^/]*\.pth$##p' "$IN" | sort -u)
else
  PATTERN="${1:-graphffcon}"
  OUT="${2:-eval_scripts/eval_checkpoints_graph_ffcon.txt}"
  : > "$OUT.tmp"
  for run_dir in logs/rl_games/*/*"${PATTERN}"*/; do
    [[ -d "$run_dir" ]] || continue
    [[ -n "$EXCLUDE" && "$run_dir" =~ $EXCLUDE ]] && continue
    emit_run "$run_dir" >> "$OUT.tmp"
  done
fi

sort -u -o "$OUT" "$OUT.tmp"
rm -f "$OUT.tmp"

n=$(wc -l < "$OUT")
echo "wrote $n checkpoint(s) to $OUT"
cat "$OUT"
echo "-> eval with: EVAL_LIST=$OUT sbatch --array=1-$n eval_scripts/eval_array.slurm"
