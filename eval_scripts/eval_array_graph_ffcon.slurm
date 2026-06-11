#!/bin/bash
# Evaluate the graph + ff_residual + contact dev runs (exp graphffcon), one array
# task per checkpoint listed in eval_scripts/eval_checkpoints_graph_ffcon.txt.
# Regenerate that list from what actually finished:  bash eval_scripts/make_eval_list.sh
# a100 dev QOS caps submitted jobs at 10/user, so the array is 1-10 (the dev sweep is
# 10 tasks); if make_eval_list.sh reports more, run: sbatch --array=11-N <this file>.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=dexmachina_eval_ffcon
#SBATCH --output=slurm_logs/slurm-eval-ffcon-%A_%a.out
#SBATCH --array=1-10%10

set -euo pipefail

: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
# W&B on persistent $WORK so eval runs survive the job (synced by wandb_sync_loop.sh).
LOG_ROOT="$WORK/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

CKPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" eval_scripts/eval_checkpoints_graph_ffcon.txt)
if [[ -z "$CKPT" ]]; then
  echo "No checkpoint for task ${SLURM_ARRAY_TASK_ID}"; exit 0
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT (regenerate the list with make_eval_list.sh)"; exit 1
fi
echo "Evaluating: $CKPT"
HAND=$(basename "$(dirname "$(dirname "$CKPT")")")
echo "Detected hand: $HAND"

uv run python dexmachina/rl/eval_rl_games.py \
  -B 20 \
  --checkpoint "$CKPT" \
  -rv
