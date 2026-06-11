#!/usr/bin/env bash
# Continuously sync offline W&B runs to wandb.ai. Compute nodes have no internet,
# so training runs with WANDB_MODE=offline and writes runs to persistent $WORK
# (see the train_array_*.slurm scripts); run THIS on a login node, detached:
#
#   tmux new -s wandb_sync 'bash training_scripts/wandb_sync_loop.sh'
#
# Optional arg overrides the log root (default matches the slurm scripts).
set -u
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"  # hush the pkg_resources deprecation spam
LOG_ROOT="${1:-$WORK/retargeting/dexmachina_logs}"
RUNS_DIR="$LOG_ROOT/wandb/wandb"          # wandb nests runs under WANDB_DIR/wandb (slurms set WANDB_DIR=$LOG_ROOT/wandb)
REPO="$(cd "$(dirname "$0")/.." && pwd)"  # this script lives in <repo>/training_scripts
# `wandb sync --sync-all` only scans ./wandb (it ignores both a path arg AND WANDB_DIR),
# so it never finds these runs. Instead pass each offline-run dir explicitly; run the wandb
# CLI via uv from the repo so its venv resolves regardless of the run dirs' location.
echo "Syncing offline runs in $RUNS_DIR every 60 s (ctrl-c to stop)"
shopt -s nullglob
while true; do
  runs=("$RUNS_DIR"/offline-run-* "$RUNS_DIR"/run-*)
  [[ ${#runs[@]} -gt 0 ]] && ( cd "$REPO" && uv run wandb sync "${runs[@]}" )
  sleep 60
done
