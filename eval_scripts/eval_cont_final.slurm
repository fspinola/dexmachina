#!/bin/bash
# Eval the FINAL converged checkpoint of each continuous run (paracont/graphcont x box/mixer).
# "Final" = latest-epoch periodic checkpoint (saved when the curriculum hit zero gains and stopped);
# NOT nn/allegro_hand.pth, which is best-REWARD and in continuous training is an early high-help one.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_cont_final
#SBATCH --output=slurm_logs/slurm-eval-final-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
ROWS="$SLURM_TMPDIR/rows.txt"; : > "$ROWS"

nvidia-smi
for tag in ${TAGS:-paracont graphcont}; do
  for spec in ${SPECS:-box30-230:box mixer30-200:mixer}; do
    clip="${spec%%:*}"; obj="${spec##*:}"
    run=$(ls -d logs/rl_games/allegro_hand/allegro-${tag}*_${clip}*B12000* 2>/dev/null | head -1)
    [ -z "$run" ] && { echo "MISSING run $tag $clip"; continue; }
    for EP in ${EPOCHS:-latest}; do        # EPOCHS="3000 5000" to eval specific epochs; default=newest
      if [ "$EP" = "latest" ]; then
        ck=$(ls -t "$run"/nn/last_allegro_hand_ep_*.pth 2>/dev/null | head -1)
      else
        ck=$(ls "$run"/nn/last_allegro_hand_ep_${EP}_*.pth 2>/dev/null | head -1)
      fi
      [ -z "$ck" ] && { echo "MISSING ckpt $tag $clip ep$EP"; continue; }
      ep=$(basename "$ck" | grep -oE "ep_[0-9]+" | head -1)
      echo "==================== EVAL $tag $clip $ep ($(basename "$ck")) ===================="
      $UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$ck"
      evaldir="$run/$(basename "${ck%.pth}")_eval"
      $UVRUN python dexmachina/eval/compute_add.py --input "$evaldir" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
      echo "${tag}|${clip}|${ep}|${evaldir}" >> "$ROWS"
    done
  done
done

echo "==================== FINAL RESULTS: continuous para vs graph (converged, eval gains=0) ===================="
while IFS='|' read -r tag clip ep evaldir; do
  f="$evaldir/add_stats.json"
  if [ -f "$f" ]; then
    $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('%-10s %-12s %-8s mean_add=%.4f m  AUC=%.4f'%(sys.argv[2],sys.argv[3],sys.argv[4],s['mean_add']['overall'],s['auc']['overall']))" "$f" "$tag" "$clip" "$ep"
  else echo "$tag $clip $ep: no add_stats.json"; fi
done < "$ROWS"
