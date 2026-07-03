#!/bin/bash
# Render videos of the -30 policy on the -40 task (the transfer eval from eval_xfer_30to40.slurm).
# Uses a *separate* checkpoint name (..._vid.pth) so the -B 1 render writes to its own _vid_eval/
# dir and does NOT overwrite the 256-env eval_ep0.npy that backs the reported ADD/SR metrics.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --job-name=xfer_video
#SBATCH --output=slurm_logs/slurm-xfer-video-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
L=logs/rl_games/allegro_hand

# obj | task40_run | policy30_ckpt(rel to $L) | vid_name
SPECS=(
"mixer|allegro-graphcont_c797758_mixer40-340-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3|allegro-graphcont_c694036_mixer30-200-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_4410_rew__147.56226_.pth|xfer_graphcont_mixer30_ep4410_vid.pth"
"ketchup|allegro-graphcont_c797758_ketchup40-340-s01-u02_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3|allegro-graphcont_c797758_ketchup30-130-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_4000_rew_88.649414.pth|xfer_graphcont_ketchup30_ep4000_vid.pth"
)

nvidia-smi || true
for spec in "${SPECS[@]}"; do
  IFS='|' read -r obj task40 pol30 vname <<< "$spec"
  xfer="$L/$task40/nn/$vname"
  echo "==================== RENDER: $obj  (-30 policy on -40 task) ===================="
  cp -f "$L/$pol30" "$xfer"
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$xfer"
  echo "  video -> $L/$task40/${vname%.pth}_eval/video.mp4"
done
echo "=== DONE video render ==="
