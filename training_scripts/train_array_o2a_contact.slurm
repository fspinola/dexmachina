#!/bin/bash
# OakInk->ARCTIC generalization RL: train DexMachina on the transfer kinrefs
# (RETARGET in {zeroshot,supwf,ftwf}) with the DEFAULT paper hybrid + contact
# recipe (identical to train_array_graph_contact.slurm), NO RSI (rand_init_ratio
# default 0). Non-dev h100 QOS, 10h. Array over runs_o2a.tsv (box + 2 mixer
# crops, allegro). Submit once per method:
#   RETARGET=zeroshot EXP_BASE=o2a_zeroshot sbatch training_scripts/train_array_o2a_contact.slurm
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --job-name=o2a_contact
#SBATCH --output=slurm_logs/o2a-contact-%A_%a.out
#SBATCH --array=1-3%3
set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
: "${WORK:=/lustre/fswork/projects/rech/ilc/uds34ao}"  # JZ site var, unset in non-interactive sbatch
LOG_ROOT="$WORK/retargeting/dexmachina_logs"; export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
export PATH="$HOME/.local/bin:$PATH"  # uv lives here; non-interactive sbatch does not source .bashrc
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"  # site default /uv_cache is unwritable in batch
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

LINE=$(grep -v '^\s*#' training_scripts/runs_o2a.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS=$(echo "$LINE" | awk '{print $4}')
EXP="${EXP_BASE:-$EXP}"
RETARGET="${RETARGET:-graph}"
GRIP_SCALE="${GRIP_SCALE:-1.0}"
STAMP="${STAMP:-j${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
EXP="${EXP}_${STAMP}"
echo "Run: hand=$HAND exp=$EXP clip=$CLIP envs=$ENVS retarget=$RETARGET (hybrid, contact -con 3, NO RSI, non-dev h100)"
nvidia-smi

uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name "$RETARGET" --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --grip_scale "$GRIP_SCALE" --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND"
