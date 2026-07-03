#!/bin/bash
# Evaluate ONE checkpoint (passed via --export=ALL,CKPT=...,OBJ=...). Used by the auto-evaluator
# loop to eval each continuous run as soon as it finishes. Writes add_stats.json + video next to
# the checkpoint and prints a RESULT_LINE the loop can scrape.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --job-name=eval_one
#SBATCH --output=slurm_logs/slurm-eval-one-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
cd /lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina

# Accept either an explicit CKPT, or a RUN dir whose latest checkpoint is resolved AT RUNTIME
# (so this can be submitted as a SLURM --dependency before training has produced the final ckpt).
: "${OBJ:?need OBJ}"
if [ -n "${CKPT:-}" ]; then
  :
elif [ -n "${RUN:-}" ]; then
  CKPT=$(ls -t "$RUN"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "RESULT_FAIL no ckpt in $RUN/nn"; exit 1; }
else
  echo "need CKPT or RUN"; exit 1
fi
echo "EVAL_ONE ckpt=$CKPT obj=$OBJ"
$UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$CKPT"
evaldir="$(dirname "$(dirname "$CKPT")")/$(basename "${CKPT%.pth}")_eval"
$UVRUN python dexmachina/eval/compute_add.py --input "$evaldir" --pattern 'eval_ep0.npy' --obj_name "$OBJ" --overwrite || true
$UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('RESULT_LINE %s AUC=%.4f mean_add=%.4f'%(sys.argv[2], s['auc']['overall'], s['mean_add']['overall']))" "$evaldir/add_stats.json" "$CKPT" || echo "RESULT_FAIL $CKPT"
