#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --job-name=dexmachina_array
#SBATCH --output=slurm-%A_%a.out
#SBATCH --array=1-12%8
#SBATCH --exclusive

set -euo pipefail

# Ensure we have a writable local scratch dir
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
echo "Using scratch: $SLURM_TMPDIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# avoid wandb networking stalls on compute nodes
export WANDB_MODE=offline
export WANDB_SILENT=true

# Caches to local disk
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

# Read the Nth non-comment line from runs.tsv
LINE=$(grep -v '^\s*#' training_scripts/runs.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS=$(echo "$LINE" | awk '{print $4}')

echo "Run: hand=$HAND exp=$EXP clip=$CLIP num_envs=$ENVS"
nvidia-smi

uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND"

