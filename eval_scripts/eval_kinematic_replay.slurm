#!/bin/bash
# Open-loop KINEMATIC replay diagnostic (needs a GPU: genesis runs on gs.gpu).
# Drives the hands along the retargeted reference exactly (action_mode=kinematic, policy
# ignored) with the object moved only by contact, then computes object ADD. Run it for
# BOTH your graph kinref and the para baseline to localize the remaining gap:
#   - para tracks ~cm but graph tracks poorly  -> the graph REFERENCE is a bad manipulation
#     (fix the retargeter; RL can't rescue it)
#   - graph also tracks ~cm                     -> reference is fine; problem is RL/reward
# Results go to <run>/<hand>_eval_kinematic/ (add_stats.json + video.mp4); the normal
# policy evals are NOT touched.
#
# Usage (from repo root, after `git pull` so eval honors -am, and `mkdir -p slurm_logs`):
#   sbatch eval_scripts/eval_kinematic_replay.slurm
#   GRAPH_CKPT=... PARA_CKPT=... sbatch --export=ALL,GRAPH_CKPT,PARA_CKPT eval_scripts/eval_kinematic_replay.slurm
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=dexmachina_kinreplay
#SBATCH --output=slurm_logs/slurm-kinreplay-%j.out

set -euo pipefail

: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

# Checkpoints to replay open-loop (any checkpoint of the run works -- the policy is ignored
# in kinematic mode; the run's params/env.pkl provides the clip + reference).
GRAPH_CKPT="${GRAPH_CKPT:-logs/rl_games/allegro_hand/allegro-graphconfix_box30-230-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/allegro_hand.pth}"
PARA_CKPT="${PARA_CKPT:-logs/rl_games/allegro_hand/allegro-20012026_box30-230-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/allegro_hand.pth}"

nvidia-smi

for CKPT in "$GRAPH_CKPT" "$PARA_CKPT"; do
  if [[ ! -f "$CKPT" ]]; then echo "SKIP (missing): $CKPT"; continue; fi
  echo "==================== kinematic replay: $CKPT ===================="
  # -B 1 => single deterministic rollout (matches para's protocol); -am kinematic => open loop.
  uv run python dexmachina/rl/eval_rl_games.py -B 1 -am kinematic -rv --checkpoint "$CKPT"
done

echo "==================== compute ADD for the kinematic replays ===================="
uv run python dexmachina/eval/compute_add.py --input logs/rl_games --pattern '**/*_eval_kinematic/eval_ep0.npy'

echo "==================== RESULTS ===================="
for CKPT in "$GRAPH_CKPT" "$PARA_CKPT"; do
  RUN="$(dirname "$(dirname "$CKPT")")"
  for f in "$RUN"/*_eval_kinematic/add_stats.json; do
    [[ -f "$f" ]] || continue
    echo "--- $f ---"
    uv run python -c "import json,sys; s=json.load(open(sys.argv[1])); print('  mean_add overall = %.4f m | AUC overall = %.4f' % (s['mean_add']['overall'], s['auc']['overall']))" "$f"
  done
done
