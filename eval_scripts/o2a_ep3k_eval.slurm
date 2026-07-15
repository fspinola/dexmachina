#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=o2a_ep3k
#SBATCH --output=slurm_logs/slurm-o2aep3k-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
L=logs/rl_games/allegro_hand
RESULTS="$SLURM_TMPDIR/r.txt"; : > "$RESULTS"
nvidia-smi
ITEMS=(
  "o2a_zeroshot_j1150738_box30-230:box"
  "o2a_zeroshot_j1150738_mixer30-200:mixer"
  "o2a_supwf_j1150739_box30-230:box"
  "o2a_supwf_j1150739_mixer30-200:mixer"
  "o2a_ftwf_j1150740_box30-230:box"
  "o2a_ftwf_j1150740_mixer30-200:mixer"
)
for it in "${ITEMS[@]}"; do
  key="${it%%:*}"; obj="${it##*:}"
  run=$(ls -d $L/allegro-${key}*B12000* 2>/dev/null | head -1)
  if [ -z "$run" ]; then echo "$key|NORUN|-|-" >> "$RESULTS"; continue; fi
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
  echo "==================== EVAL $key $ep ===================="
  if ! $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck"; then echo "$key|$ep|EVALFAIL|-" >> "$RESULTS"; continue; fi
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" | tail -1 || echo NA)
  echo "$key|$ep|$auc|$sr" >> "$RESULTS"
done
echo "========= o2a PRE-BUG ep3000 eval (retarget viability, gains=0) ========="
printf "%-40s %-8s %-9s %s\n" run epoch ADD-AUC SPIDER-SR
while IFS='|' read -r k ep auc sr; do printf "%-40s %-8s %-9s %s\n" "$k" "$ep" "$auc" "$sr"; done < "$RESULTS"
