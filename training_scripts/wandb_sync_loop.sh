#!/usr/bin/env bash
# Continuously sync offline W&B runs to wandb.ai. Compute nodes have no internet,
# so training runs with WANDB_MODE=offline and writes runs to persistent $WORK
# (see the train_array_*.slurm scripts); run THIS on a login node, detached:
#
#   tmux new -s wandb_sync 'bash training_scripts/wandb_sync_loop.sh'
#
# Optional arg overrides the log root (default matches the slurm scripts).
set -u
LOG_ROOT="${1:-$WORK/retargeting/dexmachina_logs}"
# wandb writes offline runs to $WANDB_DIR/wandb/offline-run-*; the slurms set
# WANDB_DIR=$LOG_ROOT/wandb, so the runs live at $LOG_ROOT/wandb/wandb. Point sync at the
# directory that actually contains the offline-run-* dirs (fall back to the outer one).
SYNC_DIR="$LOG_ROOT/wandb/wandb"
[[ -d "$SYNC_DIR" ]] || SYNC_DIR="$LOG_ROOT/wandb"
echo "Syncing offline runs from $SYNC_DIR every 60 s (ctrl-c to stop)"
while true; do
  wandb sync --include-offline --sync-all "$SYNC_DIR"
  sleep 60
done
