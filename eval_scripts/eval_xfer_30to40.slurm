#!/bin/bash
# Cross-clip transfer eval: run the -40 TASK with the best -30 POLICY, 256 envs.
# Mechanism (no code change): copy the -30 policy .pth into the -40 run's nn/ dir, so
# eval_rl_games.py builds the env from the -40 run's params/env.pkl (the TASK: -40 demo
# reference, object, episode length) while agent.restore() loads the -30 weights. Valid
# because graphcont -30/-40 share obs/action config (same object, hand, action_mode).
# Reports: ADD-AUC over 256 envs (compute_add loops all envs) + SPIDER success rate as
# the graded fraction of the 256 diverging rollouts that pass SPIDER's pos/rot thresholds.
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=xfer_30to40
#SBATCH --output=slurm_logs/slurm-xfer-30to40-%j.out

set -euo pipefail
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

# spec = obj | task40_run | policy30_ckpt(relative to $L) | xfer_name
SPECS=(
"mixer|allegro-graphcont_c797758_mixer40-340-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3|allegro-graphcont_c694036_mixer30-200-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_4410_rew__147.56226_.pth|xfer_graphcont_mixer30_ep4410.pth"
"ketchup|allegro-graphcont_c797758_ketchup40-340-s01-u02_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3|allegro-graphcont_c797758_ketchup30-130-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/last_allegro_hand_ep_4000_rew_88.649414.pth|xfer_graphcont_ketchup30_ep4000.pth"
)

nvidia-smi || true
EVALDIRS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r obj task40 pol30 xname <<< "$spec"
  task40_dir="$L/$task40"; pol30_ckpt="$L/$pol30"; xfer="$task40_dir/nn/$xname"
  echo "==================== TRANSFER: $obj  (-30 policy -> -40 task) ===================="
  echo "  task40 env : $task40_dir/params/env.pkl"
  echo "  policy30   : $pol30_ckpt"
  [ -f "$task40_dir/params/env.pkl" ] || { echo "MISSING task env.pkl"; exit 1; }
  [ -f "$pol30_ckpt" ] || { echo "MISSING policy ckpt"; exit 1; }
  cp -f "$pol30_ckpt" "$xfer"           # place -30 weights in the -40 run so env.pkl=-40 task
  $UVRUN python dexmachina/rl/eval_rl_games.py -B "$NENV" --checkpoint "$xfer"
  ed="$task40_dir/${xname%.pth}_eval"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ed" --pattern 'eval_ep0.npy' --obj_name "$obj" --overwrite || true
  echo "$obj|$ed" >> "$SLURM_TMPDIR/rows.txt"
  EVALDIRS+=("$ed")
done

# ---- graded SPIDER success rate over all NENV envs (reuses vetted spider helpers) ----
cat > "$SLURM_TMPDIR/sr_all_envs.py" <<'PY'
import sys, numpy as np
from dexmachina.rl.evaluate_dexmachina_spider import _align_obj_demo, _spider_sr_errors
POS_THR, QUAT_THR = 0.1, 0.5   # SPIDER defaults (m, rad)
npy = sys.argv[1]
d = np.load(npy, allow_pickle=True).item()
obj = np.asarray(d["obj_state"], dtype=np.float64)     # (T, nenv, >=8)
demo = np.asarray(d["demo_state"], dtype=np.float64)   # (T, >=8)
if demo.ndim == 3: demo = demo[:, 0, :]
if obj.ndim == 2:  obj = obj[:, None, :]
nenv = obj.shape[1]
ok = 0; pe_list = []; qe_list = []; diverged = 0
for i in range(nenv):
    o, dm = _align_obj_demo(obj[:, i, :], demo, "shift")
    pe, qe = _spider_sr_errors(o, dm)
    if not (np.isfinite(pe) and np.isfinite(qe)) or pe > 1e3:
        diverged += 1                                  # physics blow-up -> failure
        continue
    pe_list.append(pe); qe_list.append(qe)
    if pe <= POS_THR and qe <= QUAT_THR: ok += 1
pe_arr = np.array(pe_list); qe_arr = np.array(qe_list)
print(f"    SPIDER-SR (256-env, shift) : {ok}/{nenv} = {ok/nenv:.4f} ({100*ok/nenv:.1f}%)")
print(f"      thresholds               : pos_err<={POS_THR} m AND quat_err<={QUAT_THR} rad")
print(f"      pos_err  over envs (m)   : mean {np.nanmean(pe_arr):.4f}  median {np.nanmedian(pe_arr):.4f}")
print(f"      quat_err over envs (rad) : mean {np.nanmean(qe_arr):.4f}  median {np.nanmedian(qe_arr):.4f}")
print(f"      diverged (blow-up) envs  : {diverged}")
PY

echo ""
echo "############################ SUMMARY: -30 policy on -40 task (256 envs) ############################"
while IFS='|' read -r obj ed; do
  echo "==== $obj  ($ed) ===="
  f="$ed/add_stats.json"
  if [ -f "$f" ]; then
    $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('    ADD-AUC (256-env)          : %.4f   (mean_add=%.4f m)'%(s['auc']['overall'],s['mean_add']['overall']))" "$f"
  else echo "    ADD-AUC: NO add_stats.json"; fi
  $UVRUN python "$SLURM_TMPDIR/sr_all_envs.py" "$ed/eval_ep0.npy" || echo "    SPIDER-SR: failed"
done < "$SLURM_TMPDIR/rows.txt"
echo "####################################################################################################"
