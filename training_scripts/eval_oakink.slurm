#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=oak_eval
#SBATCH --output=slurm_logs/oak-eval-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te" TRITON_CACHE_DIR="$SLURM_TMPDIR/tri"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR"
export PATH="$HOME/.local/bin:$PATH"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; mkdir -p slurm_logs
CKPT="${CKPT:?set CKPT}"
OUT="${OUT:-oakink_eval.mp4}"
echo "EVAL ckpt=$CKPT -> $OUT"
uv run --no-sync python eval_oakink_video.py -ck "$CKPT" -B 1 -o "$OUT" \
  --cam_pos 0.6 -1.1 1.5 --cam_lookat -0.1 -0.1 1.0
echo "=== EXIT $? ==="
