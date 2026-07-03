#!/bin/bash
# Eval mixerlong ep5000 (to compare vs its OWN ep10000 -- same run, isolates the epoch effect),
# then report SPIDER-protocol metrics (pos L2 / rot geodesic / arti L1 + success rate) for both.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=mixerlong_spider
#SBATCH --output=slurm_logs/slurm-mixerlong-spider-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"

run=$(ls -d logs/rl_games/allegro_hand/allegro-mixerlong*mixer40-340*B12000* 2>/dev/null | head -1)
echo "run: $run"
ck5=$(ls "$run"/nn/last_allegro_hand_ep_5000_*.pth 2>/dev/null | head -1)
ed5="$run/$(basename "${ck5%.pth}")_eval"
ed10=$(ls -dt "$run"/last_allegro_hand_ep_10000_*_eval 2>/dev/null | head -1)

nvidia-smi
echo "==================== eval ep5000 ===================="
$UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$ck5"
$UVRUN python dexmachina/eval/compute_add.py --input "$ed5" --pattern 'eval_ep0.npy' --obj_name mixer --overwrite || true

echo "==================== ADD-AUC: ep5000 vs ep10000 (same run) ===================="
for ed in "$ed5" "$ed10"; do
  f="$ed/add_stats.json"
  [ -f "$f" ] && $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('  %-14s ADD-AUC=%.4f mean_add=%.4f m'%(sys.argv[2],s['auc']['overall'],s['mean_add']['overall']))" "$f" "$(basename "$ed"|grep -oE 'ep_[0-9]+')"
done

echo "==================== SPIDER protocol (pos/rot/arti + success rate) ===================="
$UVRUN python dexmachina/rl/evaluate_dexmachina_spider.py --eval_files "$ed5/eval_ep0.npy" "$ed10/eval_ep0.npy" --success --align shift
