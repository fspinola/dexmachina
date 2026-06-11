#!/bin/bash
# Train DexMachina RL on MY graph kinematic references (--retarget_name graph),
# CONTACT ON: full paper hybrid recipe incl. --use_retarget_contact -con 3, using the
# graph-pose contact files (assets/contact_retarget/*_graph.npy, generated with
# retargeting/map_contacts.py --retarget_pt <*_vector_graph.pt> --save_suffix _graph).
# Array over the 7 paper experiments x 2 hands (14 tasks) listed in runs_graph.tsv:
#   ketchup-30-130-s01-u01, box-30-230-s01-u01, mixer-30-200-s01-u01, ketchup-40-340-s01-u02,
#   mixer-40-340-s01-u01, notebook-40-340-s02-u02, waffleiron-40-340-s01-u01.
#SBATCH -A ilc@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --job-name=dexmachina_graphcon
#SBATCH --output=slurm_logs/slurm-graphcon-%A_%a.out
#SBATCH --array=1-14%8
#SBATCH --exclusive

set -euo pipefail

: "${SLURM_TMPDIR:=/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
echo "Using scratch: $SLURM_TMPDIR"

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

# Read the Nth non-comment line from runs_graph.tsv
LINE=$(grep -v '^\s*#' training_scripts/runs_graph.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS=$(echo "$LINE" | awk '{print $4}')
EXP="${EXP/nc/con}"   # graphnc -> graphcon, so contact runs don't collide with contact-off logs

echo "Run: hand=$HAND exp=$EXP clip=$CLIP num_envs=$ENVS  (retarget=graph, contact ON -con 3)"
nvidia-smi

# Full paper hybrid recipe: --use_retarget_contact loads the _graph contact files
# (per-link matched contact reward), -con 3 weights it.
uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 1000 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am hybrid --hybrid_scales 0.1 1.0 \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND"
