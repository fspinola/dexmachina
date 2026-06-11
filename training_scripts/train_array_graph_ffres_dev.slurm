#!/bin/bash
# DEV run: graph kinematic references + CONTACT ON + ff_residual action mode.
# A100 dev QOS (2h max), smaller -B to fit dev GPUs and iterate quickly; checkpoints
# every 500 epochs so partial runs are usable. Same 14-task array as train_array_graph*.slurm
# (runs_graph.tsv); env count from the tsv is overridden by DEV_ENVS (default 4096).
#
# Usage:
#   sbatch --array=1 training_scripts/train_array_graph_ffres_dev.slurm     # one task
#   DEV_ENVS=2048 sbatch --export=ALL,DEV_ENVS training_scripts/train_array_graph_ffres_dev.slurm
#SBATCH -A ilc@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=dexmachina_graphffcon_dev
#SBATCH --output=slurm_logs/slurm-graphffcon-dev-%A_%a.out
#SBATCH --array=1-14%2

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

# Read the Nth non-comment line from runs_graph.tsv (env count overridden for dev GPUs)
LINE=$(grep -v '^\s*#' training_scripts/runs_graph.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
HAND=$(echo "$LINE" | awk '{print $1}')
EXP=$(echo "$LINE"  | awk '{print $2}')
CLIP=$(echo "$LINE" | awk '{print $3}')
ENVS="${DEV_ENVS:-4096}"
EXP="${EXP/nc/ffcon}"   # graphnc -> graphffcon (ff_residual + contact)

echo "Run: hand=$HAND exp=$EXP clip=$CLIP num_envs=$ENVS  (retarget=graph, -am ff_residual, contact ON -con 3, dev)"
nvidia-smi

# Paper recipe with -am ff_residual instead of hybrid (action anchored on ref[k+1],
# limit-based scaling, next-ref error in obs). --hybrid_scales is unused by ff_residual
# unless --res_cap is added to bound the wrist residual to +/- those scales.
uv run python dexmachina/rl/train_rl_games.py -B $ENVS -obf -obt --max_epochs 5000 \
  --actuate_object --retarget_name graph --horizon 16 -imw 0.5 --gain_mode all --curr_schedule uniform \
  --wait_epochs 100 --learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 \
  --save_freq 500 --group_collisions --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 \
  --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 -am ff_residual \
  --kp_init 80 --kv_init 5 --clip "$CLIP" -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp "$EXP" --hand "$HAND"
