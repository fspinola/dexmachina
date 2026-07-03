#!/bin/bash
# Eval the CONTINUOUS in-domain baselines (graphcont + ftcont, contact+hybrid,
# default imi) at ep3000, 256-env ADD + SPIDER SR, for box + mixer clips.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=cont_ev3k
#SBATCH --output=slurm_logs/cont-ev3k-%j.out
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
echo "############ CONTINUOUS graphcont+ftcont ADD ep3000 (256-env) ############"
for tag in graphcont ftcont; do
  for c in box30-230 mixer30-200 mixer40-340; do
    d=$(ls -d "$B"/allegro-${tag}_c*_${c}-s01-u01_* 2>/dev/null | head -1)
    [ -z "$d" ] && { echo "SKIP no $tag $c"; continue; }
    ck=$(ls "$d"/nn/*ep_3000_rew_*.pth 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "SKIP no ep3000 $tag $c"; continue; }
    echo "=== EVAL $(basename $d) using $(basename $ck) ==="
    uv run python dexmachina/rl/eval_rl_games_with_metrics.py --checkpoint "$ck" --num_envs 256 --eval_episodes 1 \
      && echo "EVAL_OK $tag $c" || echo "EVAL_FAIL $tag $c"
    ckdir="$d/$(basename "$ck" .pth)_eval"; [ -f "$ckdir/eval_ep0.npy" ] && NPYS="$NPYS $ckdir/eval_ep0.npy"
  done
done
echo "############ SPIDER SR (continuous ep3000) ############"
uv run python dexmachina/rl/evaluate_dexmachina_spider.py --hand allegro_hand --eval_files $NPYS --success --summary \
  && echo SPIDER_OK || echo SPIDER_FAIL
echo "=== CONT EP3000 EVAL DONE ==="
