#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=rsi_eval256
#SBATCH --output=slurm_logs/slurm-rsi-eval256-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
RESULTS="$SLURM_TMPDIR/t.txt"; : > "$RESULTS"
nvidia-smi
for spec in "rsiconst:box30-230:box" "graphcont:box30-230:box" "rsiconst:mixer30-200:mixer" "graphcont:mixer30-200:mixer"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; clip="${rest%%:*}"; obj="${rest##*:}"
  run=$(ls -d logs/rl_games/allegro_hand/allegro-${tag}*_${clip}*B12000* 2>/dev/null | head -1)
  [ -z "$run" ] && { echo "$tag|$clip|noRUN|-|-" >> "$RESULTS"; continue; }
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$tag|$clip|noCKPT|-|-" >> "$RESULTS"; continue; }
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
  echo "==================== EVAL256 $tag $clip $ep $(basename $ck) ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck" || { echo "$tag|$clip|$ep|EVALFAIL|-" >> "$RESULTS"; continue; }
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" | tail -1 || echo NA)
  echo "$tag|$clip|$ep|$auc|$sr" >> "$RESULTS"
done
echo ""
echo "========= RSI (constant H=32) vs BASELINE (no-RSI graphcont), 256-env, eval gains=0 ========="
printf "%-11s %-14s %-9s %-9s %s\n" tag clip epoch ADD-AUC SPIDER-SR
while IFS='|' read -r tag clip ep auc sr; do printf "%-11s %-14s %-9s %-9s %s\n" "$tag" "$clip" "$ep" "$auc" "$sr"; done < "$RESULTS"
