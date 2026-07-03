#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=graphwide_256
#SBATCH --output=slurm_logs/slurm-graphwide256-%j.out
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
for spec in "c1101166:ketchup40-340:ketchup" "c1169927:ketchup40-340:ketchup" \
            "c1101166:waffleiron40-340:waffleiron" "c1169927:waffleiron40-340:waffleiron"; do
  stamp="${spec%%:*}"; rest="${spec#*:}"; clip="${rest%%:*}"; obj="${rest##*:}"
  run=$(ls -d logs/rl_games/allegro_hand/allegro-graphwide_${stamp}_${clip}*B12000* 2>/dev/null | head -1)
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$stamp|$clip|nockpt|-" >> "$RESULTS"; continue; }
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+")
  echo "==================== graphwide $stamp $clip $ep (-B 256, ±12) ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck" || { echo "$stamp|$clip|$ep|EVALFAIL|-" >> "$RESULTS"; continue; }
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" || echo NA)
  echo "$stamp|$clip|$ep|$auc|$sr" >> "$RESULTS"
done
echo ""
echo "==================== graphwide (±12 no-cap), 256-env  [two run stamps = variance] ===================="
printf "%-10s %-18s %-9s %-9s %s\n" stamp clip epoch ADD-AUC SPIDER-SR
while IFS='|' read -r stamp clip ep auc sr; do printf "%-10s %-18s %-9s %-9s %s\n" "$stamp" "$clip" "$ep" "$auc" "$sr"; done < "$RESULTS"
