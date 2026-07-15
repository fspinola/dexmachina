#!/bin/bash
# CONTINUOUS training on the h100-t3 QOS (20h walltime) -- ONE job runs a (hand,clip) all the
# way to --max_epochs with NO resume. This is REQUIRED for correctness: the object-actuation
# curriculum (kp/kv decay that weans the object off phantom "help" so the hand learns to hold
# it) is ADAPTIVE (gated on a live reward deque), so it is NOT stored in the rl_games checkpoint
# and CANNOT be restored on resume -- every resume resets the object gains back to --kp_init (80)
# and the policy never learns to hold the FREE object. At eval the object gains are forced to 0,
# so a resume-trained policy drops the object even though its training reward looks fine (~168).
# The original dexmachina baseline (allegro-20012026) was a single continuous ~10h run; reproduce
# that here. Pick RETARGET=para to reproduce the baseline, RETARGET=graph for the learned refs.
#
# Usage (from repo root, after `mkdir -p slurm_logs`):
#   RETARGET=para  EXP_BASE=paracont  sbatch --array=1-2  training_scripts/train_array_continuous_t3.slurm   # box,ketchup rows
#   RETARGET=graph EXP_BASE=graphcont sbatch --array=1-14 training_scripts/train_array_continuous_t3.slurm
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --job-name=pilot_rsiann
#SBATCH --output=slurm_logs/pilot-rsiann-%A_%a.out
#SBATCH --array=1-3%3

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
LOG_ROOT="${WORK:-/lustre/fswork/projects/rech/ilc/uds34ao}/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
# Headless/non-login submits don't carry the login PATH (uv lives in ~/.local/bin) or a writable
# uv cache. Set both explicitly so the job runs regardless of how it was submitted.
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

LINE=$(grep -v '^\s*#' training_scripts/runs_o2a.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS=$(echo "$LINE" | awk '{print $4}')
EXP="${EXP/nc/con}"
EXP="${EXP_BASE:-$EXP}"
RETARGET="${RETARGET:-graph}"
GRIP_SCALE="${GRIP_SCALE:-1.0}"
ACTION_MODE="${ACTION_MODE:-hybrid}"   # hybrid (fingers free) | residual (all joints = ref + correction) | ff_residual (ref at t+1)
RES_CAP="${RES_CAP:-0}"                 # 1 -> add --res_cap: cap the WRIST residual to +-hybrid_scales (like hybrid). Only affects residual/ff_residual.
IMI="${IMI:-0.3}"                       # imitation reward weight (default 0.3; raise to pull the pose toward the kinref)
RAND_INIT="${RAND_INIT:-0.0}"           # RSI: prob. an env resets to a RANDOM ref timestep (0=off; e.g. 0.5)
MAX_EPOCHS="${MAX_EPOCHS:-5000}"           # total continuous training epochs (raise for hard/long clips)
RESCAP_FLAG=""; [ "$RES_CAP" = "1" ] && RESCAP_FLAG="--res_cap"
# continuous runs reach max_epochs in one job, so a per-submission stamp is optional; keep it for isolation.
STAMP="${STAMP:-c${SLURM_ARRAY_JOB_ID:-0}}"
EXP="${EXP}_${STAMP}"

echo "CONTINUOUS run: hand=$HAND exp=$EXP clip=$CLIP num_envs=$ENVS retarget=$RETARGET action_mode=$ACTION_MODE res_cap=$RES_CAP imi=$IMI rand_init=$RAND_INIT grip_scale=$GRIP_SCALE (h100-t3, 20h, NO resume)"
nvidia-smi

uv run --no-sync python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs "$MAX_EPOCHS" \
  --actuate_object --retarget_name "$RETARGET" --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am "$ACTION_MODE" --hybrid_scales 0.1 1.0 $RESCAP_FLAG \
  --kp_init 80 --kv_init 5 --grip_scale "$GRIP_SCALE" --rand_init_ratio 1.0 --rsi_horizon 32 --rsi_horizon_end 200 --rsi_anneal_epochs 1500 --clip "$CLIP" -imi "$IMI" -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND"
