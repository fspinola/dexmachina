#!/bin/bash
# Eval ep5000 o2acont (zeroshot/supwf/ftwf x box-30-230, mixer-30-200) at 256-env:
# ADD_AUC + eval_ep0.npy (for 256-env SPIDER SR) + rendered video.mp4 per run.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=o2ac_ev5k_vid
#SBATCH --output=slurm_logs/o2ac-ev5k-vid-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te" TRITON_CACHE_DIR="$SLURM_TMPDIR/tri" CUDA_CACHE_PATH="$SLURM_TMPDIR/cc"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
: "${WORK:=/lustre/fswork/projects/rech/ilc/uds34ao}"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; nvidia-smi
B="$ROOT/logs/rl_games/allegro_hand"
NPYS=""
for m in zeroshot supwf ftwf; do for c in box30-230 mixer30-200; do
  d=$(ls -d "$B"/allegro-o2acont_${m}_c*_${c}-s01-u01_* 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "SKIP no $m $c"; continue; }
  ck=$(ls "$d"/nn/*ep_5000_rew_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "SKIP no ep5000 $m $c"; continue; }
  echo "=== EVAL+RENDER $m $c using $(basename $ck) ==="
  uv run python dexmachina/rl/eval_rl_games_with_metrics.py --checkpoint "$ck" --num_envs 256 --eval_episodes 1 --record_video --show_reference \
    && echo "EVAL_OK $m $c" || echo "EVAL_FAIL $m $c"
  ckdir="$d/$(basename "$ck" .pth)_eval"
  [ -f "$ckdir/eval_ep0.npy" ] && NPYS="$NPYS $ckdir/eval_ep0.npy"
  [ -f "$ckdir/video.mp4" ] && echo "VIDEO $m $c -> $ckdir/video.mp4"
done; done
echo "############ SPIDER (env_idx0 quick) ############"
uv run python dexmachina/rl/evaluate_dexmachina_spider.py --hand allegro_hand --eval_files $NPYS --success --summary || echo SPIDER_FAIL
echo "=== O2AC EV5K VID DONE ==="
