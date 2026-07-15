#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:35:00
#SBATCH --job-name=oakkp_fixed
#SBATCH --output=slurm_logs/oakkp-fixed-%j.out
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
# FIXED-schedule Kp curriculum: decays Kp deterministically every --interval epochs (no reward
# gating), so the multi-object weaning is provable fast. Expect New Gains: kp = 80 -> 40 -> 20 ...
echo "OAKINK Kp FIXED-schedule decay test"
uv run --no-sync python dexmachina/rl/train_rl_games.py \
  --oakink --oakink_pt "$PT" --hand allegro_hand -B 6000 --max_epochs 150 --horizon 16 \
  -am hybrid --hybrid_scales 0.1 1.0 \
  -imi 0.3 -imw 0.5 -bc 0.3 -con 0 -ert 0.6 --aux_reset_thres 0 0 0 \
  --actuate_object --kp_init 80 --kv_init 5 --gain_mode all \
  --curr_schedule fixed --fixed_mode exp --interval 20 --wait_epochs 10 \
  --rand_init_ratio 0.0 \
  --clip e76b2at3-0-769 -exp oakkp_fixed
echo "=== EXIT $? ==="
