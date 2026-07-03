#!/bin/bash
# Baseline: eval the in-domain GRAPH hybrid+contact runs (graphconfix, same
# recipe + post-forearm-fix code as the o2a runs) at ep3000, 256-env ADD +
# SPIDER SR, for box-30-230 / mixer-30-200 / mixer-40-340.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=graph_eval3k
#SBATCH --output=slurm_logs/graph-eval3k-%j.out
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
SUF=B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3
NPYS=""
echo "############ GRAPH (graphconfix) ADD ep3000 (256-env) ############"
for c in box30-230 mixer30-200 mixer40-340; do
  for d in "$B"/allegro-graphconfix*_${c}-s01-u01_${SUF}; do
    ck=$(ls "$d"/nn/*ep_3000_rew_*.pth 2>/dev/null | head -1)
    [ -n "$ck" ] || continue
    echo "=== EVAL $(basename $d) using $(basename $ck) ==="
    uv run python dexmachina/rl/eval_rl_games_with_metrics.py --checkpoint "$ck" --num_envs 256 --eval_episodes 1 \
      && echo "EVAL_OK $c" || echo "EVAL_FAIL $c"
    ckdir="$d/$(basename "$ck" .pth)_eval"; [ -f "$ckdir/eval_ep0.npy" ] && NPYS="$NPYS $ckdir/eval_ep0.npy"
    break
  done
done
echo "############ SPIDER SR (graph ep3000) ############"
uv run python dexmachina/rl/evaluate_dexmachina_spider.py --hand allegro_hand --eval_files $NPYS --success --summary \
  && echo SPIDER_OK || echo SPIDER_FAIL
echo "=== GRAPH EP3000 EVAL DONE ==="
