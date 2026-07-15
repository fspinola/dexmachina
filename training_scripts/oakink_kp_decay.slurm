#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --job-name=oakkp_decay
#SBATCH --output=slurm_logs/oakkp-decay-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te" TRITON_CACHE_DIR="$SLURM_TMPDIR/tri"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR"
export PATH="$HOME/.local/bin:$PATH"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; mkdir -p slurm_logs
PT=dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt
# Kp-decay demonstrator: OakInk + object actuation + short-horizon RSI (so the curriculum's
# achieved-length gate opens) + low wait_epochs (so decay can start by ~ep 60). Mirrors ARCTIC
# rsi_const but on the two OakInk objects -- expect "New Gains: kp=..." to fall from 80 -> 0.
echo "OAKINK Kp-DECAY TEST"
uv run --no-sync python dexmachina/rl/train_rl_games.py \
  --oakink --oakink_pt "$PT" --hand allegro_hand -B 6000 --max_epochs 250 --horizon 16 \
  -am hybrid --hybrid_scales 0.1 1.0 \
  -imi 0.3 -imw 0.5 -bc 0.3 -con 0 -ert 0.6 --aux_reset_thres 0 0 0 \
  --actuate_object --kp_init 80 --kv_init 5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 20 --deque_len 25 --rand_init_ratio 1.0 --rsi_horizon 32 \
  --clip e76b2at3-0-769 -exp oakkp_decay
echo "=== EXIT $? ==="
