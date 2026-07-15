#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=batch_eval
#SBATCH --output=slurm_logs/slurm-batcheval-%j.out
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
  "o2acont_zeroshot_c1193897_box30-230:box"
  "o2acont_zeroshot_c1193897_mixer30-200:mixer"
  "o2acont_zeroshot_c1193897_mixer40-340:mixer"
  "o2acont_supwf_c1193898_box30-230:box"
  "o2acont_supwf_c1193898_mixer30-200:mixer"
  "o2acont_supwf_c1193898_mixer40-340:mixer"
  "o2acont_ftwf_c1193899_box30-230:box"
  "o2acont_ftwf_c1193899_mixer30-200:mixer"
  "o2acont_ftwf_c1193899_mixer40-340:mixer"
  "o2acont_zeroshot3_c1794242_box30-230:box"
  "o2acont_zeroshot3_c1794242_mixer30-200:mixer"
  "graphcontFT3b_mixer30to40_clean_mixer40-340:mixer"
  "graphcontFT3b_scratch40_mixer40-340:mixer"
  "graphwide2_c1315840_waffleiron40-340:waffleiron"
  "graphwide2_c1315841_waffleiron40-340:waffleiron"
  "graphwide3_c1794378_waffleiron40-340:waffleiron"
  "graphwide3_c1794379_waffleiron40-340:waffleiron"
)
for it in "${ITEMS[@]}"; do
  key="${it%%:*}"; obj="${it##*:}"
  run=$(ls -d $L/allegro-${key}*B12000* 2>/dev/null | head -1)
  if [ -z "$run" ]; then echo "$key|NORUN|-|-" >> "$RESULTS"; continue; fi
  ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "$key|NOCKPT|-|-" >> "$RESULTS"; continue; fi
  ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
  echo "==================== EVAL $key $ep ===================="
  if ! $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck"; then
    echo "$key|$ep|EVALFAIL|-" >> "$RESULTS"; continue
  fi
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" | tail -1 || echo NA)
  echo "$key|$ep|$auc|$sr" >> "$RESULTS"
done
echo ""
echo "========= BATCH EVAL RESULTS (B=256, gains=0, full rollout) ========="
printf "%-46s %-8s %-9s %s\n" run epoch ADD-AUC SPIDER-SR
while IFS='|' read -r k ep auc sr; do printf "%-46s %-8s %-9s %s\n" "$k" "$ep" "$auc" "$sr"; done < "$RESULTS"
