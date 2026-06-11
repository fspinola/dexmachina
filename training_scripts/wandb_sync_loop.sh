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
# `wandb sync --sync-all` IGNORES any positional path and only scans $WANDB_DIR/wandb
# (or ./wandb if WANDB_DIR is unset). So set WANDB_DIR to the same value the training
# slurms use -- then sync finds exactly the runs they wrote (at $LOG_ROOT/wandb/wandb).
export WANDB_DIR="$LOG_ROOT/wandb"
echo "Syncing offline runs from $WANDB_DIR/wandb every 60 s (ctrl-c to stop)"
while true; do
  wandb sync --include-offline --sync-all
  sleep 60
done
