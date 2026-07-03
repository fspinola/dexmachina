#!/bin/bash
# Eval the BEST-REWARD checkpoint (nn/allegro_hand.pth) of each graph task at -B 256, and report
# its true epoch (read from the checkpoint) + AUC + SPIDER SR. Compare vs the LATEST-checkpoint
# table: in continuous curriculum training the best-reward point is often EARLY (object still
# heavily helped -> high reward) and may eval WORSE than the final checkpoint.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_best_256
#SBATCH --output=slurm_logs/slurm-eval-best256-%j.out
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

RUNS=( "box30-230:box" "mixer30-200:mixer" "ketchup30-130:ketchup" "mixer40-340:mixer"
       "notebook40-340:notebook" "waffleiron40-340:waffleiron" "ketchup40-340:ketchup" )
nvidia-smi
for spec in "${RUNS[@]}"; do
  clip="${spec%%:*}"; obj="${spec##*:}"
  run=$(ls -d logs/rl_games/allegro_hand/allegro-graphcont*_${clip}*B12000* 2>/dev/null | head -1)
  ck="$run/nn/allegro_hand.pth"
  [ -f "$ck" ] || { echo "$clip|noBest|-|-" >> "$RESULTS"; continue; }
  ep=$($UVRUN python -c "import torch;d=torch.load('$ck',map_location='cpu',weights_only=False);print(d.get('epoch',d.get('frame','?')))" 2>/dev/null || echo "?")
  echo "==================== graph $clip  best-ckpt ep$ep  (-B 256) ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck" || { echo "$clip|$ep|EVALFAIL|-" >> "$RESULTS"; continue; }
  ed="$run/allegro_hand_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" || echo NA)
  echo "$clip|$ep|$auc|$sr" >> "$RESULTS"
done
echo ""
echo "==================== graph BEST-checkpoint (nn/allegro_hand.pth), 256-env ===================="
printf "%-20s %-9s %-9s %s\n" clip best_ep ADD-AUC SPIDER-SR
while IFS='|' read -r clip ep auc sr; do printf "%-20s %-9s %-9s %s\n" "$clip" "$ep" "$auc" "$sr"; done < "$RESULTS"
