#!/bin/bash
# Re-eval the 9 o2a policies at a UNIFORM epoch (ep3000 checkpoint, present for
# all runs) so the method comparison is same-epoch. 256-env ADD + SPIDER SR on
# the ep3000 rollouts. Fallback to allegro_hand.pth if ep3000 missing.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=o2a_eval3k
#SBATCH --output=slurm_logs/o2a-eval3k-%j.out
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
echo "############ ADD ep3000 (256-env) ############"
for d in "$B"/allegro-o2a_*; do
  ck=$(ls "$d"/nn/*ep_3000_rew_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && ck="$d/nn/allegro_hand.pth"
  echo "=== EVAL $(basename $d) using $(basename $ck) ==="
  uv run python dexmachina/rl/eval_rl_games_with_metrics.py --checkpoint "$ck" --num_envs 256 --eval_episodes 1 \
    && echo "EVAL_OK $(basename $d)" || echo "EVAL_FAIL $(basename $d)"
  ckdir="$d/$(basename "$ck" .pth)_eval"
  [ -f "$ckdir/eval_ep0.npy" ] && NPYS="$NPYS $ckdir/eval_ep0.npy"
done
echo "############ SPIDER SR (ep3000 rollouts) ############"
uv run python dexmachina/rl/evaluate_dexmachina_spider.py --hand allegro_hand --eval_files $NPYS --success --summary \
  && echo SPIDER_OK || echo SPIDER_FAIL
echo "=== O2A EP3000 EVAL DONE ==="
