#!/bin/bash
# Re-eval the main runs at -B 256 (256 parallel rollouts -> robust AUC + a real SPIDER success
# rate over 256 diverse rollouts), consistent across every run/sequence. Uses the LATEST-epoch
# checkpoint per run. AUC = compute_add (already averages over envs); SR = sr_over_envs.py.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_table_256
#SBATCH --output=slurm_logs/slurm-eval-table256-%j.out

set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
NENV="${NENV:-256}"
RESULTS="$SLURM_TMPDIR/table.txt"; : > "$RESULTS"

RUNS=(
  "graphcont:box30-230:box"        "graphcont:mixer30-200:mixer"
  "graphcont:ketchup30-130:ketchup" "graphcont:mixer40-340:mixer"
  "graphcont:notebook40-340:notebook" "graphcont:waffleiron40-340:waffleiron"
  "graphcont:ketchup40-340:ketchup"
  "graphreenc:ketchup40-340:ketchup" "graphreenc:waffleiron40-340:waffleiron"
  "paracont:box30-230:box"          "paracont:mixer30-200:mixer"
  "ftcont:box30-230:box"            "ftcont:mixer30-200:mixer"
)

nvidia-smi
for spec in "${RUNS[@]}"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; clip="${rest%%:*}"; obj="${rest##*:}"
  run=$(ls -d logs/rl_games/allegro_hand/allegro-${tag}*_${clip}*B12000* 2>/dev/null | head -1)
  [ -z "$run" ] && { echo "MISSING $tag $clip"; echo "$tag|$clip|-|-|-" >> "$RESULTS"; continue; }
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "MISSING ckpt $tag $clip"; echo "$tag|$clip|nockpt|-|-" >> "$RESULTS"; continue; }
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
  echo "==================== $tag $clip $ep  (-B $NENV) ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B "$NENV" --checkpoint "$ck" || { echo "$tag|$clip|$ep|EVALFAIL|-" >> "$RESULTS"; continue; }
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" | head -1 || echo NA)
  echo "$tag|$clip|$ep|$auc|$sr" >> "$RESULTS"
done

echo ""
echo "==================== TABLE (256-env): AUC + SPIDER SR ===================="
printf "%-12s %-20s %-9s %-9s %s\n" method clip epoch ADD-AUC SPIDER-SR
while IFS='|' read -r tag clip ep auc sr; do
  printf "%-12s %-20s %-9s %-9s %s\n" "$tag" "$clip" "$ep" "$auc" "$sr"
done < "$RESULTS"
