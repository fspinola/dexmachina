#!/bin/bash
# Train DexMachina RL on MY LEARNED FINE-TUNED kinematic references (--retarget_name ft),
# CONTACT OFF (no --use_retarget_contact, -con 0). Otherwise the paper hybrid recipe.
# Identical to train_array_graph.slurm except retarget_name graph -> ft (loads the
# *_vector_ft.pt files exported by learned_retargeter/scripts/export_dexmachina_kinrefs_ft.sh).
# Array over the 7 paper experiments x 2 hands (14 tasks) listed in runs_ft.tsv:
#   ketchup-30-130-s01-u01, box-30-230-s01-u01, mixer-30-200-s01-u01, ketchup-40-340-s01-u02,
#   mixer-40-340-s01-u01, notebook-40-340-s02-u02, waffleiron-40-340-s01-u01.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
# 1 of 4 GPUs -> request 1/4 of the 96-core node (24 cores -> ~1/4 RAM); do NOT use
# --exclusive (it reserves the whole node incl. the other 3 GPUs -> long queue + waste).
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --job-name=dexmachina_ft
#SBATCH --output=slurm_logs/slurm-ft-%A_%a.out
#SBATCH --array=1-14%8

set -euo pipefail

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
# W&B: offline runs must land on persistent $WORK (SLURM_TMPDIR is wiped at job end),
# so they survive the job and a login-node loop can sync them (see wandb_sync_loop.sh).
LOG_ROOT="$WORK/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

# Read the Nth non-comment line from runs_ft.tsv
LINE=$(grep -v '^\s*#' training_scripts/runs_ft.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS=$(echo "$LINE" | awk '{print $4}')

echo "Run: hand=$HAND exp=$EXP clip=$CLIP num_envs=$ENVS  (retarget=ft, contact OFF)"
nvidia-smi

# Same as the paper hybrid recipe MINUS contact: dropped --use_retarget_contact, set -con 0.
uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name ft --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 0 -ert 0.6 -exp "$EXP" --hand "$HAND"
