#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:25:00
#SBATCH --job-name=rsi_test
#SBATCH --output=slurm_logs/rsi_test-%j.out
set -uo pipefail
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

COMMON="-B 1024 -obf -obt --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all \
  --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 \
  --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 100000 --group_collisions \
  --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 --dialback_ep_len 80 --skip_grad \
  --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact --aux_reset_thres 0 0 0 \
  --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
  --grip_scale 1.0 --clip box-30-230-s01-u01 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 --hand allegro_hand"

echo "######## TEST 1: fixed-horizon RSI + ANNEAL (randr=1.0, H 32->200 over 25 ep) ########"
uv run --no-sync python dexmachina/rl/train_rl_games.py $COMMON --max_epochs 40 \
  --rand_init_ratio 1.0 --rsi_horizon 32 --rsi_horizon_end 200 --rsi_anneal_epochs 25 -exp rsi_anneal
RC1=$?
echo "######## TEST 1 EXIT: $RC1 ########"

echo "######## TEST 2: fixed-horizon RSI CONSTANT (randr=0.5, H=64) ########"
uv run --no-sync python dexmachina/rl/train_rl_games.py $COMMON --max_epochs 20 \
  --rand_init_ratio 0.5 --rsi_horizon 64 -exp rsi_const
RC2=$?
echo "######## TEST 2 EXIT: $RC2 ########"
echo "######## SUMMARY: anneal=$RC1 const=$RC2 ########"
