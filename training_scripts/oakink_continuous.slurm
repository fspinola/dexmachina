#!/bin/bash
# OakInk-v2 CONTINUOUS RL on a single (hand, clip) to --max_epochs, NO resume.
# Task + fingertip-imitation only: rigid two-object bimanual, contacts OFF (single-part contact
# reward not implemented for OakInk yet) and RSI OFF. Demo (per-object) + retarget come from the
# exported .pt (learned_retargeter export_dexmachina_oakink.py); frames are baked into the .pt.
#
# Usage (from repo root, after `git pull`):
#   sbatch training_scripts/oakink_continuous.slurm
#   PT=dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt \
#     HAND=allegro_hand EXP=oakcont_e76b2at3 sbatch training_scripts/oakink_continuous.slurm
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --job-name=oakink_cont
#SBATCH --output=slurm_logs/oakink-cont-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
LOG_ROOT="${WORK:-/lustre/fswork/projects/rech/ilc/uds34ao}/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; mkdir -p slurm_logs

PT="${PT:-dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt}"
HAND="${HAND:-allegro_hand}"
ENVS="${ENVS:-12000}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
EXP="${EXP:-oakcont_e76b2at3}"
CLIP="${CLIP:-e76b2at3-0-769}"   # cosmetic (exp-name only); actual frames come from the .pt
CON="${CON:-0}"                  # contact reward weight (needs a .pt exported with --contacts); 0=off
CK="${CK:-}"                     # checkpoint for full-state resume (weights+optimizer+epoch); empty=fresh
CK_ARG=(); [ -n "$CK" ] && CK_ARG=(--checkpoint "$CK")
ACT="${ACT:-0}"                  # 1 = actuate objects + Kp gain-curriculum (assist-then-wean)
ACT_ARG=(); [ "$ACT" = "1" ] && ACT_ARG=(--actuate_object --kp_init 80 --kv_init 5 --num_zero_epoch 500)
echo "OAKINK CONTINUOUS: hand=$HAND envs=$ENVS pt=$PT exp=$EXP con=$CON act=$ACT resume=${CK:-none} (hybrid, NO RSI, h100-t3 20h)"
nvidia-smi || true

# Defaults mirror train_array_continuous_t3 (imi 0.3, imw 0.5, bc 0.3, ert 0.6, hybrid_scales 0.1 1.0,
# horizon 16) MINUS object actuation/curriculum (objects are rigid) and MINUS -con/-obt/-obf
# (contacts + tip-dist obs off for OakInk). rand_init_ratio 0 = no RSI.
uv run --no-sync python dexmachina/rl/train_rl_games.py \
  --oakink --oakink_pt "$PT" --hand "$HAND" -B "$ENVS" --max_epochs "$MAX_EPOCHS" --horizon 16 \
  -am hybrid --hybrid_scales 0.1 1.0 \
  -imi 0.3 -imw 0.5 -bc 0.3 -con "$CON" --rand_init_ratio 0.0 -ert 0.6 --aux_reset_thres 0 0 0 \
  "${CK_ARG[@]}" "${ACT_ARG[@]}" --clip "$CLIP" -exp "$EXP"
echo "=== EXIT $? ==="
