#!/bin/bash
# CONTROLLED thumb A/B: same code, same fixed January policy (ep2000), ONLY the left-thumb
# origin differs (new -0.005 vs old +0.005). Race-free: builds two temp URDFs in $SLURM_TMPDIR
# and patches a COPY of env.pkl to point at each -- the shared repo URDF is never touched.
# Both evals run as separate processes (no in-memory caching); the two URDFs have different
# filenames (no on-disk mesh-cache collision). Also diffs the two object trajectories to PROVE
# the thumb actually changed the rollout. Answers: (1) did the eval really apply the thumb?
# (2) what is the thumb's TRUE isolated effect on ADD under current code?
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=thumb_ab
#SBATCH --output=slurm_logs/slurm-thumb-ab-%j.out

set -euo pipefail
: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=offline WANDB_SILENT=true
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache" TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda_cache" WANDB_DIR="$SLURM_TMPDIR/wandb"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SLURM_TMPDIR/uv_cache"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR" "$UV_CACHE_DIR"
UVRUN="uv run --no-sync"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"
RUN=logs/rl_games/allegro_hand/allegro-20012026_box30-230-s01-u01_B12000_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3
CKNAME=last_allegro_hand_ep_2000_rew_166.48695.pth
CK=$RUN/nn/$CKNAME
SRC=$ROOT/dexmachina/assets/allegro_hand/allegro_hand_left_6dof.urdf
T=$SLURM_TMPDIR/ab; mkdir -p "$T"
ln -sfn "$(dirname "$SRC")/meshes" "$T/meshes"                        # so the temp URDFs resolve their relative mesh paths

cp "$SRC" "$T/left_new.urdf"                                          # current repo = new thumb (-0.005)
sed 's/-0.027 -0.005 0.0399/-0.027 0.005 0.0399/' "$SRC" > "$T/left_old.urdf"   # reverted = old thumb (+0.005)
echo "new-thumb line: $(grep '0.0399' "$T/left_new.urdf" | grep '0.027' | head -1 | tr -s ' ')"
echo "old-thumb line: $(grep '0.0399' "$T/left_old.urdf" | grep '0.027' | head -1 | tr -s ' ')"
nvidia-smi

for V in new old; do
  mkdir -p "$T/run_$V/params" "$T/run_$V/nn"
  cp "$CK" "$T/run_$V/nn/"
  $UVRUN python patch_env_urdf.py "$RUN/params/env.pkl" "$T/left_$V.urdf" "$T/run_$V/params/env.pkl"
  echo "==================== EVAL thumb=$V ===================="
  $UVRUN python dexmachina/rl/eval_rl_games.py -B 1 -rv --checkpoint "$T/run_$V/nn/$CKNAME"
  $UVRUN python dexmachina/eval/compute_add.py --input "$T/run_$V/${CKNAME%.pth}_eval" \
    --pattern 'eval_ep0.npy' --obj_name box --overwrite || true
done

echo "==================== RESULTS (same code+policy, ONLY thumb differs) ===================="
for V in new old; do
  f="$T/run_$V/${CKNAME%.pth}_eval/add_stats.json"
  [[ -f "$f" ]] && $UVRUN python -c "import json,sys;s=json.load(open(sys.argv[1]));print('  %-4s thumb:  mean_add=%.4f m  AUC=%.4f'%(sys.argv[2],s['mean_add']['overall'],s['auc']['overall']))" "$f" "$V" || echo "  $V: no stats"
done

echo "==================== did the thumb actually change the rollout? ===================="
$UVRUN python -c "
import numpy as np
a=np.load('$T/run_new/${CKNAME%.pth}_eval/eval_ep0.npy',allow_pickle=True).item()
b=np.load('$T/run_old/${CKNAME%.pth}_eval/eval_ep0.npy',allow_pickle=True).item()
A=np.asarray(a['obj_state']); B=np.asarray(b['obj_state'])
d=np.abs(A-B)
print('  obj_state  max|new-old|=%.6f  mean=%.6f  (identical => thumb NOT applied / eval bug)'%(d.max(),d.mean()))
p=np.linalg.norm(A[...,:3]-B[...,:3],axis=-1).reshape(A.shape[0],-1).mean(-1)
print('  obj-pos L2 per-frame:  first=%.5f  mid=%.5f  last=%.5f  max=%.5f m'%(p[0],p[len(p)//2],p[-1],p.max()))
"
OUT=$RUN/_thumb_ab_controlled; mkdir -p "$OUT"
cp "$T/run_new/${CKNAME%.pth}_eval/video.mp4" "$OUT/newthumb.mp4" 2>/dev/null || true
cp "$T/run_old/${CKNAME%.pth}_eval/video.mp4" "$OUT/oldthumb.mp4" 2>/dev/null || true
echo "videos: $OUT/{newthumb,oldthumb}.mp4"
