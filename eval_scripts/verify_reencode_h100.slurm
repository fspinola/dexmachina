#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --job-name=verify_reencode
#SBATCH --output=slurm_logs/slurm-verify-%j.out
set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache" XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
cd /lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
uv run --no-sync python verify_reencode.py
