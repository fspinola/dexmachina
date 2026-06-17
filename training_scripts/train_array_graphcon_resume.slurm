#!/bin/bash
# RESUME the graph + contact (graphcon) runs from their latest checkpoint, on the
# h100 DEV QOS (2h slots). rl_games restores model+optimizer+epoch from a checkpoint and
# continues toward --max_epochs (5000); 2h won't finish, so just resubmit -- each task
# auto-discovers the NEWEST checkpoint (by mtime) for that (hand, clip) across any exp tag
# (the original graphcon_j<JOBID> dirs or a prior resume) and writes to a STABLE -exp
# ("graphcon", no job id), so resubmissions self-resume from the furthest epoch. Recipe
# MUST match the checkpoint (hybrid + contact) -- the ff_residual dev script can't load
# these (its extra observation block changes the obs dim).
#
# Checkpointing: nn/<hand>.pth is saved only on a new best reward; last_<hand>_ep_N_*.pth
# every SAVE_FREQ epochs. We resume from whichever is newest, and set SAVE_FREQ small
# (default 500) so each 2h slot leaves a recent full-state snapshot (SLURM kills the job
# at the walltime without saving, so progress since the last snapshot is lost on timeout
# -- lower SAVE_FREQ to lose less).
#
# --array=1-10 == exactly the 10 checkpointed tasks (ketchup-30-130, box-30-230,
# mixer-30-200, ketchup-40-340, mixer-40-340 x both hands; tsv rows 1-10). Rows 11-14
# (notebook, waffleiron) have no checkpoint -- run those fresh elsewhere.
#
# Usage (from the repo root, after `mkdir -p slurm_logs`):
#   sbatch training_scripts/train_array_graphcon_resume.slurm                 # resume the 10
#   SAVE_FREQ=200 sbatch --export=ALL,SAVE_FREQ ...                           # finer snapshots
#   RESUME_ENVS=8000 sbatch --export=ALL,RESUME_ENVS ...                      # smaller batch (e.g. OOM)
# Verify the h100-dev QOS name and submit/job limits on your allocation first:
#   sacctmgr show qos qos_gpu_h100-dev format=Name,MaxWall,MaxSubmitJobsPerUser,MaxJobsPerUser
# If the submit limit is < 10, split the array (e.g. --array=1-5 then --array=6-10).
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=dexmachina_graphcon_resume
#SBATCH --output=slurm_logs/slurm-graphcon-resume-%A_%a.out
#SBATCH --array=1-10%10

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
# W&B: offline runs must land on persistent $WORK (SLURM_TMPDIR is wiped at job end),
# so they survive the job and a login-node loop can sync them (see wandb_sync_loop.sh).
LOG_ROOT="$WORK/retargeting/dexmachina_logs"
export WANDB_DIR="$LOG_ROOT/wandb"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WANDB_DIR"

ROOT=/lustre/fswork/projects/rech/ilc/uds34ao/retargeting/dexmachina
cd "$ROOT"

LINE=$(grep -v '^\s*#' training_scripts/runs_graph.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS="${RESUME_ENVS:-$(echo "$LINE" | awk '{print $4}')}"
SAVE_FREQ="${SAVE_FREQ:-500}"
EXP="${EXP_BASE:-graphcon}"   # stable output name (no job id) -> future resubmissions self-resume.
                              # Set EXP_BASE=graphconfix to resume an ISOLATED track that never reads or
                              # writes the old graphcon dirs (discovery glob below is keyed to $EXP).

# Clip slug as it appears in the exp dir name: box-30-230-s01-u01 -> box30-230-s01-u01
IFS='-' read -r OBJ START END SUBJ USEP <<< "$CLIP"
SLUG="${OBJ}${START}-${END}-${SUBJ}-${USEP}"

# Newest checkpoint for this (hand, clip), across any exp tag AND any checkpoint name
# (best nn/<hand>.pth or periodic last_<hand>_ep_N_*.pth), by mtime -> furthest epoch.
CKPT=$(ls -dt logs/rl_games/${HAND}/${HAND%_hand}-${EXP}*_${SLUG}_B*/nn/*.pth 2>/dev/null | head -1 || true)
RESUME_ARG=""
if [[ -n "${CKPT}" ]]; then
  echo "Resuming hand=$HAND clip=$CLIP from: $CKPT (envs=$ENVS)"
  RESUME_ARG="--checkpoint $CKPT"
else
  echo "WARNING: no checkpoint found for $HAND $SLUG; starting fresh (envs=$ENVS)"
fi
nvidia-smi

# Same paper hybrid+contact recipe as train_array_graph_contact.slurm, plus --checkpoint.
uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq $SAVE_FREQ --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND" $RESUME_ARG
