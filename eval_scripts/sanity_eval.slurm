#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=sanity_eval
#SBATCH --output=slurm_logs/slurm-sanityeval-%j.out
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
# KNOWN-GOOD baselines: graphcont_c694036 previously eval'd box 0.872 / mixer 0.966 (job 1425133)
for it in "graphcont_c694036_box30-230:box" "graphcont_c694036_mixer30-200:mixer"; do
  key="${it%%:*}"; obj="${it##*:}"
  run=$(ls -d $L/allegro-${key}*B12000* 2>/dev/null | head -1)
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
  echo "==================== SANITY EVAL $key $ep ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck" || { echo "$key|EVALFAIL"; continue; }
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" | tail -1 || echo NA)
  echo "$key|$ep|$auc|$sr" >> "$RESULTS"
done
echo "========= SANITY (expect box~0.87 / mixer~0.97 if harness OK) ========="
cat "$RESULTS"
