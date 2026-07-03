#!/bin/bash
# Fixed-horizon RSI (ManipTrans-style) experiment: box-30 + mixer-30, annealed vs constant.
# Built on value_bootstrap fix (2e3f279) + RSI (ebf8f94). allegro_hand, graph, B=12000, h100-t3 20h,
# CONTINUOUS (no resume: object-actuation curriculum can't survive resume). horizon=32, rand_init=1.0.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --job-name=rsi_exp
#SBATCH --output=slurm_logs/rsi_exp-%A_%a.out
#SBATCH --array=1-4%4

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
LOG_ROOT="${WORK:-/lustre/fswork/projects/rech/ilc/uds34ao}/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"
cd /lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina

case $SLURM_ARRAY_TASK_ID in
  1) CLIP=box-30-230-s01-u01;   MODE=anneal ;;
  2) CLIP=box-30-230-s01-u01;   MODE=const  ;;
  3) CLIP=mixer-30-200-s01-u01; MODE=anneal ;;
  4) CLIP=mixer-30-200-s01-u01; MODE=const  ;;
  *) echo "bad array id $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
if [ "$MODE" = "anneal" ]; then
  RSI_FLAGS="--rsi_horizon 32 --rsi_horizon_end 400 --rsi_anneal_epochs 2000"   # end auto-caps to demo length
else
  RSI_FLAGS="--rsi_horizon 32"
fi
EXP="rsi${MODE}"
echo "RSI exp: hand=allegro_hand clip=$CLIP mode=$MODE flags=[$RSI_FLAGS] rand_init=1.0 (h100-t3 20h, NO resume)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync python dexmachina/rl/train_rl_games.py -B 12000 -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --grip_scale 1.0 --rand_init_ratio 1.0 $RSI_FLAGS \
  --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand allegro_hand
