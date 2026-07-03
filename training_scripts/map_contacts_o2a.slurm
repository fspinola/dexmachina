#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --job-name=o2a_contacts
#SBATCH --output=slurm_logs/o2a-contacts-%j.out
set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export PATH="$HOME/.local/bin:$PATH"  # uv lives here; non-interactive sbatch does not source .bashrc
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"  # site default /uv_cache is unwritable in batch
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; nvidia-smi
for method in zeroshot supwf ftwf; do
  for clip in box_use_01 mixer_use_01; do
    echo "=== map_contacts $clip _$method ==="
    uv run python dexmachina/retargeting/map_contacts.py --hand allegro_hand \
      --load_fname dexmachina/assets/arctic/processed/s01/${clip}.npy \
      --retarget_pt dexmachina/assets/retargeted/allegro_hand/s01/${clip}_vector_${method}.pt \
      --save_suffix _${method}
  done
done
echo "=== O2A CONTACTS DONE ==="
