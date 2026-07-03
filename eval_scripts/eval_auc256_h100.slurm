#!/bin/bash
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:50:00
#SBATCH --job-name=eval_auc256
#SBATCH --output=slurm_logs/slurm-auc256-%j.out
set -uo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
ALLE="$ROOT/logs/rl_games/allegro_hand"
NENV=${NENV:-256}
LIST="$SLURM_TMPDIR/targets.txt"
$UVRUN python - > "$LIST" <<'PYEOF'
import json, glob, re, os
ROOT="/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina/logs/rl_games/allegro_hand"
def mode_of(run):
    return "ff_residual" if "_ff_residual_" in run else ("residual" if "_residual_" in run else "hybrid")
def best(regex,clip):
    rgx=re.compile(regex); b=(-1,None,None)
    for f in glob.glob(f"{ROOT}/*/**/add_stats.json",recursive=True):
        run=f.split("/allegro_hand/")[1].split("/")[0]
        if not rgx.match(run) or clip not in run: continue
        try: auc=json.load(open(f))["auc"]["overall"]
        except: continue
        ckb=os.path.basename(os.path.dirname(f))[:-5]
        if auc>b[0]: b=(auc,run,ckb)
    return b
B=["box30-230-s01-u01","mixer30-200-s01-u01"]
G7=B+["ketchup30-130-s01-u01","ketchup40-340-s01-u02","mixer40-340-s01-u01","notebook40-340-s02-u02","waffleiron40-340-s01-u01"]
EXP=[("para",r"^allegro-paracont_c\d+_",B),("ft",r"^allegro-ftcont_c\d+_",B),
     ("graph_hybrid",r"^allegro-graphcont_c\d+_",G7),("residual",r"^allegro-graphres_",B),
     ("ff_residual",r"^allegro-graphffres_c\d",B),("ffres_rescap",r"^allegro-graphffres_rc_",B),
     ("hybrid_imi1",r"^allegro-graphcont_imi1_",B),("ffres_imi1",r"^allegro-graphffres_imi1_",B),
     ("ft_ffres_rc_imi1",r"^allegro-ftffres_rcimi1_",B)]
for lab,rgx,clips in EXP:
    for clip in clips:
        auc,run,ckb=best(rgx,clip)
        if not run: continue
        mode=mode_of(run); pth=f"{ROOT}/{run}/nn/{ckb}.pth"; ed=f"{ROOT}/{run}/{ckb}_eval_{mode}"
        if os.path.exists(pth): print(f"{lab}|{clip}|{mode}|{pth}|{ed}")
PYEOF
echo "=== targets ($(wc -l < "$LIST")) ==="; cat "$LIST"
echo "=== run eval -B $NENV for any MISSING eval files ==="
while IFS='|' read -r label clip mode pth ed; do
  if [ -f "$ed/eval_ep0.npy" ]; then echo "SKIP(exists) $label $clip"; continue; fi
  echo "==== EVAL $label $clip mode=$mode ===="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B "$NENV" -am "$mode" --checkpoint "$pth" \
    && echo "EVAL_OK $label $clip" || echo "EVAL_FAIL $label $clip"
done < "$LIST"
echo "=== compute_add (B=$NENV AUC) over all _eval_<mode> dirs ==="
for pat in "_eval_hybrid" "_eval_ff_residual" "_eval_residual"; do
  echo "--- pattern **/*${pat}/eval_ep0.npy ---"
  $UVRUN python dexmachina/eval/compute_add.py --input "$ALLE" --pattern "**/*${pat}/eval_ep0.npy" --overwrite || echo "ADD_FAIL $pat"
done
echo "ALL_DONE_AUC256"
