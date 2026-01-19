#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=dexmachina_eval
#SBATCH --output=slurm_logs/slurm-eval-%A_%a.out
#SBATCH --array=1-30%6

set -euo pipefail

# -----------------------------
# Scratch & environment
# -----------------------------
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
echo "Using scratch: $SLURM_TMPDIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export WANDB_MODE=offline
export WANDB_SILENT=true

export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
export WANDB_DIR="$SLURM_TMPDIR/wandb"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$WANDB_DIR"

# -----------------------------
# Project
# -----------------------------
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

# -----------------------------
# Select checkpoint
# -----------------------------
CKPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" eval_scripts/eval_checkpoints.txt)

if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found for task ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "Evaluating checkpoint:"
echo "  $CKPT"

# Optional: derive hand name from path
HAND=$(basename "$(dirname "$(dirname "$CKPT")")")
echo "Detected hand: $HAND"

# -----------------------------
# Run evaluation
# -----------------------------
uv run python dexmachina/rl/eval_rl_games_with_metrics.py \
  -B 1 \
  --checkpoint "$CKPT" \
  --add_auc_thresh_frac_diam 0.1 \
  -rv
