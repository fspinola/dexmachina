#!/bin/bash
#SBATCH -A ilc@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --job-name=reencode_check
#SBATCH --output=slurm_logs/slurm-reencode-%j.out
set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache" XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
cd /lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
uv run --no-sync python verify_reencode.py
