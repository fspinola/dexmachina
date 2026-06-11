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
echo "Syncing offline runs from $LOG_ROOT/wandb every 60 s (ctrl-c to stop)"
while true; do
  wandb sync --include-offline --sync-all "$LOG_ROOT/wandb"
  sleep 60
done
