#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=mixerlong_256
#SBATCH --output=slurm_logs/slurm-mixerlong256-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
run=$(ls -d logs/rl_games/allegro_hand/allegro-mixerlong*mixer40-340*B12000* | head -1)
nvidia-smi
declare -a OUT
for ep in 5000 10000; do
  ck=$(ls "$run"/nn/last_allegro_hand_ep_${ep}_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { OUT+=("ep$ep|no-ckpt|-"); continue; }
  echo "==================== mixerlong ep$ep (-B 256) ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 256 --checkpoint "$ck" || { OUT+=("ep$ep|EVALFAIL|-"); continue; }
  ed="$run/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name mixer --overwrite || true
  auc=$($UVRUN python -c "import json;print('%.4f'%json.load(open('$ed/add_stats.json'))['auc']['overall'])" 2>/dev/null || echo NA)
  sr=$($UVRUN python eval_scripts/sr_over_envs.py "$ed/eval_ep0.npy" 2>/dev/null | grep -oE "SR=[0-9/]+=[0-9.]+" || echo NA)
  OUT+=("ep$ep|$auc|$sr")
done
echo "==================== mixerlong 256-env: AUC + SPIDER SR ===================="
for o in "${OUT[@]}"; do echo "  mixer-40-340 $o" | tr '|' ' '; done
