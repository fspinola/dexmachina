#!/bin/bash
# RESUME the graph + contact (graphcon) runs from their latest checkpoint.
# rl_games restores model+optimizer+epoch from nn/<hand>.pth and continues toward
# --max_epochs (5000). Per array task it auto-discovers the newest checkpoint for that
# (hand, clip) across any exp tag (the original graphcon_j<JOBID> dirs or a prior resume)
# by mtime, and writes to a STABLE -exp ("graphcon", no job id) so repeated resubmissions
# self-resume from their own newest checkpoint. Resuming with a different -B is fine
# (the policy depends on obs/action dims, not env count).
#
# Resources fixed per the IDRIS note: 1 GPU + 24 CPUs (1/4 of the 96-core H100 node =
# proportional share -> 1/4 node memory), NO --exclusive (was reserving all 4 GPUs and
# the whole node, hence the long queue).
#
# Usage (from the repo root, after `mkdir -p slurm_logs`):
#   sbatch training_scripts/train_array_graphcon_resume.slurm            # all 14 tasks
#   RESUME_ENVS=8000 sbatch --export=ALL,RESUME_ENVS ...                 # smaller batch
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --job-name=dexmachina_graphcon_resume
#SBATCH --output=slurm_logs/slurm-graphcon-resume-%A_%a.out
#SBATCH --array=1-14%8

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
# W&B: offline runs must land on persistent $WORK (SLURM_TMPDIR is wiped at job end),
# so they survive the job and a login-node loop can sync them (see wandb_sync_loop.sh).
LOG_ROOT="$WORK/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

LINE=$(grep -v '^\s*#' training_scripts/runs_graph.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS="${RESUME_ENVS:-$(echo "$LINE" | awk '{print $4}')}"
EXP="graphcon"   # stable output name (no job id) -> future resubmissions self-resume

# Clip slug as it appears in the exp dir name: box-30-230-s01-u01 -> box30-230-s01-u01
IFS='-' read -r OBJ START END SUBJ USEP <<< "$CLIP"
SLUG="${OBJ}${START}-${END}-${SUBJ}-${USEP}"

# Newest checkpoint for this (hand, clip), across any exp tag, by mtime.
CKPT=$(ls -dt logs/rl_games/${HAND}/${HAND%_hand}-graphcon*_${SLUG}_B*/nn/${HAND}.pth 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [[ -n "${CKPT}" ]]; then
  echo "Resuming hand=$HAND clip=$CLIP from: $CKPT (envs=$ENVS)"
  RESUME_ARG="--checkpoint $CKPT"
else
  echo "WARNING: no checkpoint found for $HAND $SLUG; starting fresh (envs=$ENVS)"
fi
nvidia-smi

# Same paper hybrid+contact recipe as train_array_graph_contact.slurm, plus --checkpoint.
uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND" $RESUME_ARG
