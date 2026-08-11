#!/bin/bash
# RL FINE-TUNE: train the mixer-40-340 sequence WARM-STARTED from the best mixer-30-200 graph policy.
# ---------------------------------------------------------------------------------------------------
# Design decisions (see chat rationale):
#  * WEIGHTS-ONLY warm-start (--warmstart_ckpt): loads policy+value weights, FRESH optimizer, epoch=0.
#    -> clean LR (adaptive KL) and clean object-curriculum schedule (which is driven by epoch_num).
#    Full-state resume (--checkpoint) would carry stale Adam moments + a curriculum that thinks it is
#    already at epoch ~4410 (past wait_epochs) and would wean object help before the policy adapts.
#  * Curriculum MID-POINT restart: kp_init/kv_init halved (80->40, 5->3). The pretrained policy already
#    holds the object on the 40-200 overlap, so full help would re-teach help-dependence; half help
#    still bootstraps the UNSEEN 200-340 extension.
#  * Exploration-collapse handling: RSI (--rand_init_ratio 0.5) initialises envs directly into the
#    extension frames (state coverage a converged policy can't reach on its own) + small entropy_coef
#    (0.0 -> 0.003) to keep the near-deterministic policy exploring. (Sigma reset is available via
#    --warmstart_sigma but left off; entropy+RSI are the robust levers.)
#  * Everything else = the graphcont continuous recipe (train_array_continuous_t3.slurm), so obs(510)/
#    action(44) dims match the checkpoint (a mismatch would make set_weights fail immediately).
# Base it on the continuous (NO-resume) QOS: the curriculum must anneal to kp~0 within ONE job.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --job-name=ft_mixer30to40
#SBATCH --output=slurm_logs/slurm-ft-mixer30to40-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
LOG_ROOT="${WORK:-/lustre/fswork/projects/rech/ilc/uds34ao}/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"; mkdir -p slurm_logs

CLIP="${CLIP:-mixer-40-340-s01-u01}"
CKPT="${CKPT:-logs/rl_games/allegro_hand/allegro-graphcont_c694036_mixer30-200-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_4410_rew__147.56226_.pth}"
EXP="${EXP:-graphcontFT_mixer30to40}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
KP_INIT="${KP_INIT:-40}"; KV_INIT="${KV_INIT:-3}"; RAND_INIT="${RAND_INIT:-0.5}"; ENT="${ENT:-0.003}"
ENVS="${ENVS:-12000}"; SAVE_FREQ="${SAVE_FREQ:-500}"
# Action mode for the FT run -- MUST match the donor checkpoint (obs dims differ per mode).
# ff_residual anchors actions to the NEW clip's kinref (bounded corrections + ref-error obs), so a
# warm-started policy cannot execute confident-wrong free-hand behaviors on the unseen extension.
AM="${AM:-hybrid}"
RETARGET="${RETARGET:-graph}"   # kinref: graph (default) | ft | para | ... MUST match the warmstart donor's kinref
RESCAP="${RESCAP:-0}"; RESCAP_FLAG=""; [ "$RESCAP" = "1" ] && RESCAP_FLAG="--res_cap"
SIGMA="${SIGMA:-}"; SIGMA_FLAG=""; [ -n "$SIGMA" ] && SIGMA_FLAG="--warmstart_sigma $SIGMA"  # log-std; e.g. -1.386 -> std 0.25
# CKPT=none (or empty) -> pristine FROM-SCRATCH control on current code (no warm-start), same recipe.
# RESUME=1 -> FULL-STATE resume (--checkpoint: keeps optimizer+value+epoch, continues smoothly,
# no warm-start reward crash). Correct tool for continuing the SAME clip. Default 0 = weights-only
# warm-start (--warmstart_ckpt, fresh optimizer/epoch) for CROSS-clip transfer.
RESUME="${RESUME:-0}"
WARM_FLAG=""
if [ -n "$CKPT" ] && [ "$CKPT" != "none" ]; then
  [ -f "$CKPT" ] || { echo "MISSING ckpt: $CKPT"; exit 1; }
  if [ "$RESUME" = "1" ]; then WARM_FLAG="--checkpoint $CKPT"; SIGMA_FLAG=""; else WARM_FLAG="--warmstart_ckpt $CKPT"; fi
else
  CKPT="(none: from-scratch)"; SIGMA_FLAG=""
fi
echo "RUN: clip=$CLIP  ckpt=$CKPT  exp=$EXP  retarget=$RETARGET am=$AM rescap=$RESCAP kp_init=$KP_INIT rand_init=$RAND_INIT ent=$ENT sigma=${SIGMA:-inherited} resume=$RESUME (h100-t3)"
nvidia-smi || true

uv run --no-sync python dexmachina/rl/train_rl_games.py -B "$ENVS" -obf -obt --max_epochs "$MAX_EPOCHS" \
  $WARM_FLAG --entropy_coef "$ENT" $SIGMA_FLAG \
  --actuate_object --retarget_name "$RETARGET" --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq "$SAVE_FREQ" --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am "$AM" $RESCAP_FLAG --hybrid_scales 0.1 1.0 \
  --kp_init "$KP_INIT" --kv_init "$KV_INIT" --grip_scale 1.0 --rand_init_ratio "$RAND_INIT" \
  --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand allegro_hand
echo "=== EXIT $? ==="
