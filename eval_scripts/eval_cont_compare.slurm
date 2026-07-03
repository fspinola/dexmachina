#!/bin/bash
# Evaluate the CONTINUOUS para vs graph runs (paracont / graphcont), box + mixer, at ep2000 & ep3000.
# Eval forces object gains to 0, so AUC here is the real "does the hand hold the free object" metric.
# Anchors for comparison: Jan para baseline box ep2000=0.896/ep3000=0.94; paracheck (resumed) drops.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_cont_cmp
#SBATCH --output=slurm_logs/slurm-eval-cont-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"

EPOCHS="${EPOCHS:-2000 3000}"
nvidia-smi
for tag in paracont graphcont; do
  for spec in "box30-230:box" "mixer30-200:mixer"; do
    clip="${spec%%:*}"; obj="${spec##*:}"
    run=$(ls -d logs/rl_games/allegro_hand/allegro-${tag}*_${clip}*B12000* 2>/dev/null | head -1)
    [ -z "$run" ] && { echo "MISSING run: $tag $clip"; continue; }
    for ep in $EPOCHS; do
      ck=$(ls "$run"/nn/last_allegro_hand_ep_${ep}_*.pth 2>/dev/null | head -1)
      [ -z "$ck" ] && { echo "MISSING ckpt: $tag $clip ep$ep"; continue; }
      echo "==================== EVAL $tag $clip ep$ep ===================="
      $UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$ck"
      evaldir="$run/$(basename "${ck%.pth}")_eval"
      $UVRUN python dexmachina/eval/compute_add.py --input "$evaldir" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
    done
  done
done

echo "==================== RESULTS: continuous para vs graph (eval gains=0) ===================="
printf "%-11s %-12s %-6s %-10s %-8s\n" tag clip epoch mean_add AUC
for tag in paracont graphcont; do
  for spec in "box30-230:box" "mixer30-200:mixer"; do
    clip="${spec%%:*}"
    run=$(ls -d logs/rl_games/allegro_hand/allegro-${tag}*_${clip}*B12000* 2>/dev/null | head -1)
    [ -z "$run" ] && continue
    for ep in $EPOCHS; do
      f=$(ls "$run"/nn/last_allegro_hand_ep_${ep}_*_eval/add_stats.json 2>/dev/null | head -1)
      [ -z "$f" ] && f=$(ls "$run"/last_allegro_hand_ep_${ep}_*_eval/add_stats.json 2>/dev/null | head -1)
      [ -z "$f" ] && { printf "%-11s %-12s %-6s %s\n" "$tag" "$clip" "$ep" "(no stats)"; continue; }
      $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('%-11s %-12s %-6s %-10.4f %-8.4f'%(sys.argv[2],sys.argv[3],sys.argv[4],s['mean_add']['overall'],s['auc']['overall']))" "$f" "$tag" "$clip" "$ep"
    done
  done
done
