#!/bin/bash
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=eval_sr20
#SBATCH --output=slurm_logs/slurm-eval-sr20-%j.out
set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"; mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/te"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"
ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina; cd "$ROOT"
NENV=${NENV:-20}
LIST="$SLURM_TMPDIR/targets.txt"
$UVRUN python - > "$LIST" <<'PYEOF'
import json, glob, re, os
ROOT="/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina/logs/rl_games/allegro_hand"
def mode_of(run):
    if "_ff_residual_" in run: return "ff_residual"
    if "_residual_" in run: return "residual"
    if "_hybrid_" in run: return "hybrid"
    return "hybrid"
def best(regex,clip):
    rgx=re.compile(regex); b=(-1,None,None)
    for f in glob.glob(f"{ROOT}/*/**/add_stats.json",recursive=True):
        run=f.split("/allegro_hand/")[1].split("/")[0]
        if not rgx.match(run) or clip not in run: continue
        try: auc=json.load(open(f))["auc"]["overall"]
        except: continue
        ckname=os.path.basename(os.path.dirname(f))[:-5]
        pth=os.path.join(ROOT,run,"nn",ckname+".pth")
        if auc>b[0]: b=(auc,run,pth)
    return b
EXP=[("para",r"^allegro-paracont_c\d+_"),("ft",r"^allegro-ftcont_c\d+_"),
     ("graph_hybrid",r"^allegro-graphcont_c\d+_"),("residual",r"^allegro-graphres_"),
     ("ff_residual",r"^allegro-graphffres_c\d"),("ffres_rescap",r"^allegro-graphffres_rc_"),
     ("hybrid_imi1",r"^allegro-graphcont_imi1_"),("ffres_imi1",r"^allegro-graphffres_imi1_"),
     ("ft_ffres_rc_imi1",r"^allegro-ftffres_rcimi1_")]
for lab,rgx in EXP:
    for clip in ["box30-230-s01-u01","mixer30-200-s01-u01"]:
        auc,run,pth=best(rgx,clip)
        if pth and os.path.exists(pth): print(f"{lab}|{clip}|{mode_of(run)}|{pth}")
PYEOF
echo "=== targets ==="; cat "$LIST"; echo "================"
nvidia-smi | head -12
while IFS='|' read -r label clip mode pth; do
  echo "==================== EVAL $label $clip mode=$mode NENV=$NENV ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B "$NENV" -am "$mode" --checkpoint "$pth" \
    && echo "EVAL_OK $label $clip" || echo "EVAL_FAIL $label $clip"
done < "$LIST"
echo "ALL_DONE_SR20"
