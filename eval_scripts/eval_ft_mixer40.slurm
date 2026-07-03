#!/bin/bash
# Evaluate the mixer30->40 FINE-TUNED policies on the mixer-40 task, 256 envs.
# Standard eval (each ft run's env.pkl IS the mixer-40 task) -> ADD-AUC over 256 envs + graded
# SPIDER SR over 256. Object gains forced to 0 at eval (the real test: hold the FREE object).
# Two runs x {ep4000, ep5000}: RSI (rand0.5) vs clean (no RSI). Baselines for context:
# scratch mixer-40 own-policy = 0.773 ; zero-shot mixer30->40 transfer = 0.272.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_ft_mixer40
#SBATCH --output=slurm_logs/slurm-eval-ft-mixer40-%j.out
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
NENV="${NENV:-256}"

RSI="$L/allegro-graphcontFT_mixer30to40_mixer40-340-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_rand0.5_bc0.3"
CLEAN="$L/allegro-graphcontFT_mixer30to40_clean_mixer40-340-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3"

CKPTS=()
for run in "$RSI" "$CLEAN"; do
  for ep in 4000 5000; do
    ck=$(ls "$run"/nn/last_allegro_hand_ep_${ep}_*.pth 2>/dev/null | head -1)
    [ -n "$ck" ] && CKPTS+=("$ck")
  done
done

cat > "$SLURM_TMPDIR/sr_all_envs.py" <<'PY'
import sys, numpy as np
from dexmachina.rl.evaluate_dexmachina_spider import _align_obj_demo, _spider_sr_errors
POS_THR, QUAT_THR = 0.1, 0.5
d = np.load(sys.argv[1], allow_pickle=True).item()
obj = np.asarray(d["obj_state"], dtype=np.float64); demo = np.asarray(d["demo_state"], dtype=np.float64)
if demo.ndim == 3: demo = demo[:, 0, :]
if obj.ndim == 2:  obj = obj[:, None, :]
n = obj.shape[1]; ok = 0; pes = []; qes = []; div = 0
for i in range(n):
    o, dm = _align_obj_demo(obj[:, i, :], demo, "shift")
    pe, qe = _spider_sr_errors(o, dm)
    if not (np.isfinite(pe) and np.isfinite(qe)) or pe > 1e3: div += 1; continue
    pes.append(pe); qes.append(qe)
    if pe <= POS_THR and qe <= QUAT_THR: ok += 1
print(f"    SPIDER-SR (256-env)  : {ok}/{n} = {ok/n:.4f} ({100*ok/n:.1f}%)  pos_err med {np.nanmedian(pes):.4f} m  quat med {np.nanmedian(qes):.4f}  diverged {div}")
PY

nvidia-smi || true
for ck in "${CKPTS[@]}"; do
  echo "==================== EVAL $ck ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B "$NENV" --checkpoint "$ck"
  ed="$(dirname "$(dirname "$ck")")/$(basename "${ck%.pth}")_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name mixer --overwrite || true
  echo "$ck|$ed" >> "$SLURM_TMPDIR/rows.txt"
done

echo ""
echo "############ SUMMARY: mixer30->40 FINE-TUNE eval (256 envs) ############"
echo "  baselines: scratch mixer-40 own=0.773 | zero-shot transfer=0.272"
while IFS='|' read -r ck ed; do
  variant=$(echo "$ck" | grep -oE 'FT_mixer30to40(_clean)?'); ep=$(echo "$ck" | grep -oE 'ep_[0-9]+')
  echo "==== ${variant} ${ep} ===="
  f="$ed/add_stats.json"
  [ -f "$f" ] && $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('    ADD-AUC (256-env)    : %.4f   (mean_add=%.4f m)'%(s['auc']['overall'],s['mean_add']['overall']))" "$f" || echo "    ADD-AUC: none"
  $UVRUN python "$SLURM_TMPDIR/sr_all_envs.py" "$ed/eval_ep0.npy" || echo "    SPIDER-SR: failed"
done < "$SLURM_TMPDIR/rows.txt"
echo "#######################################################################"
