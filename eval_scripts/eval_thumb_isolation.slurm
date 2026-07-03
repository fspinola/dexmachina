#!/bin/bash
# ISOLATION TEST: does the June-9 left-thumb URDF fix (joint_13 origin y 0.005 -> -0.005,
# commit 318470a) break the para baseline at eval, INDEPENDENT of any retraining/resume?
#
# We re-evaluate the SAME fixed January checkpoint (allegro-20012026, ep2000) that produced
# the "good" video back in Jan. In January it ran on the OLD thumb (+0.005). The repo file it
# loads now has the NEW thumb (-0.005). So:
#   - The preserved January eval folder  = OLD thumb (policy & sim matched)   -> known good.
#   - A fresh re-eval now                = NEW thumb (policy trained on old, sim is new) -> ?
# Same policy, same clip, only the thumb differs. If the fresh run drops the box while the
# January one holds it, the shared URDF change is the cause -- not the resume, not a code regression.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=thumb_isolation
#SBATCH --output=slurm_logs/slurm-thumb-isolation-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache" WANDB_DIR="$SLURM_TMPDIR/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"
# a100/gpu_p5 compute nodes don't put uv on PATH; add it explicitly. The profile exports
# UV_CACHE_DIR=/uv_cache (writable on h100, NOT here) -> force a per-job writable cache UNCONDITIONALLY.
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"   # venv already synced; --no-sync avoids extra cache writes/resolves

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

RUN=logs/rl_games/allegro_hand/allegro-20012026_box30-230-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3
CK=$RUN/nn/last_allegro_hand_ep_2000_rew_166.48695.pth
EVALDIR=$RUN/last_allegro_hand_ep_2000_rew_166.48695_eval   # default folder eval_rl_games writes to

nvidia-smi
echo "current thumb (joint_13) the eval will load:"
grep -n "0.0399" dexmachina/assets/allegro_hand/allegro_hand_left_6dof.urdf | grep "0.027" | head -1

# 1) preserve the January (OLD-thumb) eval so the re-eval can't clobber it
if [[ -d "$EVALDIR" && ! -d "${EVALDIR}_JAN_oldthumb" ]]; then
  cp -r "$EVALDIR" "${EVALDIR}_JAN_oldthumb"
  echo "preserved January eval -> ${EVALDIR}_JAN_oldthumb"
fi

# 2) re-eval the SAME checkpoint on the current (NEW-thumb) repo; -B 1 deterministic, -rv records video
$UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$CK"
rm -rf "${EVALDIR}_NOW_newthumb"
mv "$EVALDIR" "${EVALDIR}_NOW_newthumb"
echo "fresh new-thumb eval -> ${EVALDIR}_NOW_newthumb"

# 3) restore the canonical eval folder name from the January copy (leave the run dir as we found it)
cp -r "${EVALDIR}_JAN_oldthumb" "$EVALDIR"

# 4) object ADD/AUC for both conditions (obj_name passed explicitly so no dir-name inference needed)
echo "==================== ADD: OLD thumb (January, known-good) vs NEW thumb (now) ===================="
for COND in JAN_oldthumb NOW_newthumb; do
  $UVRUN python dexmachina/eval/compute_add.py --input "${EVALDIR}_${COND}" \
    --pattern 'eval_ep0.npy' --obj_name box --overwrite || true
done

echo "==================== RESULTS ===================="
for COND in JAN_oldthumb NOW_newthumb; do
  f="${EVALDIR}_${COND}/add_stats.json"
  [[ -f "$f" ]] || { echo "$COND: (no add_stats.json)"; continue; }
  $UVRUN python -c "import json,sys; s=json.load(open(sys.argv[1])); print('  %-14s mean_add=%.4f m  AUC=%.4f' % (sys.argv[2], s['mean_add']['overall'], s['auc']['overall']))" "$f" "$COND"
done
echo "videos:"
echo "  OLD thumb (Jan): ${EVALDIR}_JAN_oldthumb/video.mp4"
echo "  NEW thumb (now): ${EVALDIR}_NOW_newthumb/video.mp4"
