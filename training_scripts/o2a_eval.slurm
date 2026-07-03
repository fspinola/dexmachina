#!/bin/bash
# Eval the 9 o2a RL policies: 256-env ADD (add_auc/success_at_thresh) via
# eval_rl_games_with_metrics + SPIDER SR via evaluate_dexmachina_spider.
# Self-contained env (non-interactive sbatch): PATH/UV_CACHE, sklearn already in venv.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=o2a_eval
#SBATCH --output=slurm_logs/o2a-eval-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"; mkdir -p "$UV_CACHE_DIR"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te" TRITON_CACHE_DIR="$SLURM_TMPDIR/tri" CUDA_CACHE_PATH="$SLURM_TMPDIR/cc"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
: "${WORK:=/lustre/fswork/projects/rech/ilc/uds34ao}"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"; nvidia-smi
B="$ROOT/logs/rl_games/allegro_hand"
echo "############ ADD (256-env) per o2a run ############"
for d in "$B"/allegro-o2a_*; do
  ck="$d/nn/allegro_hand.pth"
  [ -f "$ck" ] || { echo "SKIP (no ckpt) $(basename $d)"; continue; }
  echo "=== EVAL $(basename $d) ==="
  uv run python dexmachina/rl/eval_rl_games_with_metrics.py --checkpoint "$ck" --num_envs 256 --eval_episodes 1 \
    && echo "EVAL_OK $(basename $d)" || echo "EVAL_FAIL $(basename $d)"
done
echo "############ SPIDER SR summary ############"
uv run python dexmachina/rl/evaluate_dexmachina_spider.py --eval_root logs/rl_games --hand allegro_hand --success --summary \
  && echo "SPIDER_OK" || echo "SPIDER_FAIL"
echo "=== O2A EVAL DONE ==="
