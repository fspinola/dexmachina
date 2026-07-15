# Offline BC warm-start from kinematic references

Teacher-forced behavioral cloning of the rl_games PPO actor from exported
OakInk kinematic-reference clips — the checkpoint then initializes RL
fine-tuning through the existing `--warmstart_ckpt` flag. Design rationale and
audit trail: [BC_WARMSTART_PLAN.md](BC_WARMSTART_PLAN.md).

## What it does / does not do

Does:

- Regress the actor mean (`a2c_network.mu` + shared trunk) on the exact
  actions that make the hybrid controller reproduce the kinematic reference,
  using observations reconstructed offline to match the env bit-for-bit.
- Produce checkpoints that `train_rl_games.py --warmstart_ckpt` loads with no
  further changes (same yaml network, same state-dict schema).

Does NOT:

- Step the simulator or roll out anything (offline only).
- Initialize the critic. The value head and `value_mean_std` are saved at
  fresh-init values — kinematic references are not simulator transitions and
  cannot train a value function. Early RL must expect a random critic.
- Train the exploration noise: the fixed log-std parameter is written once
  (`--log_std_init`, default 0.0 = the yaml init) and never updated by BC.
- Make the motion dynamically feasible. The BC policy imitates the kinematic
  reference, contact physics included or not.

## Input data

OakInk exports from `learned_retargeter/kinematic/export_dexmachina_oakink.py`
(`torch.save({'demo_data', 'retarget_data'})`, one `.pt` per clip), e.g. the
five local clips under `dexmachina/assets/retargeted/allegro_hand/oakink/`.
Clips are loaded through `envs/demo_data.py::load_genesis_retarget_data`, so
the forearm 2π re-seating and the per-clip wrist limits match the env exactly.

ARCTIC clips are rejected: the ARCTIC env forces an 84-dim sim-measured
contact-force observation block (`envs/constructors.py`) that cannot be
reconstructed offline.

All clips in one BC run must share the observation layout (two-object = 341
dims vs shared-object = 319 for bimanual Allegro), and the resulting policy
can only warm-start RL runs with that same layout and the same hand.

## Exact action-label formula

Mirrors `envs/robot.py::translate_actions` (hybrid branch; `action_moving_avg`
is 1.0 everywhere so the EMA is a no-op; Allegro has no mimic joints). The
teacher commands the reference pose at `t + h` (`--label_horizon h`, default 1
— the pose the PD should reach by the end of the step, which is also the frame
the reward compares against):

```
wrist  (dofs 0-5):  a = (ref[t+h] - ref[t]) / (s_trans, s_trans, s_trans, s_rot, s_rot, s_rot)
finger (dofs 6-21): a = 2 * (ref[t+h] - lo) / (hi - lo) - 1        # URDF limits
a = clip(a, -1, 1)                                                  # counted, warned above --clip_warn_frac
```

`h=0` reproduces the `kinematic` action mode's teacher (wrist ≡ 0). The full
action vector is `[left 22, right 22]`. `--hybrid_scales` MUST match the RL
run (production: `0.1 1.0`); a mismatch is warned about at RL load time.

Observations are reconstructed under the perfect-tracking assumption
(`dof_pos = ref[t]`, object = demo[t], velocities = 60 Hz finite differences;
see the docstring of `bc_dataset.build_clip_observations` for the two known,
accepted mismatches: post-reset zeroed object obs and sim-measured velocities).
Non-velocity dims were verified to match a live env to float32 epsilon —
re-check anytime with:

```bash
python dexmachina/rl/verify_bc_obs.py --oakink \
    --oakink_pt dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt \
    --hand allegro_hand -B 1 -am hybrid --hybrid_scales 0.1 1.0
```

## Training

```bash
# venv note (local box): use /home/fspinola/venvs/dexmachina-venv2/bin/python —
# the repo-root .venv torch is broken locally. On JZ, `uv run python` as usual.
python dexmachina/rl/train_bc_kinref.py \
    --data dexmachina/assets/retargeted/allegro_hand/oakink/*_vector_oakink.pt \
    --out logs/bc_kinref/allegro_oakink \
    --val_clips 1 --epochs 200 --batch_size 4096 --lr 1e-3 \
    --hybrid_scales 0.1 1.0 --label_horizon 1
```

- Train/val split is trajectory-level (whole clips). With a single clip the
  split is disabled with a warning — overfitting that clip is intended for a
  single-sequence warm-start.
- Checkpoints: `<out>/nn/bc_latest.pth` + `<out>/nn/bc_best.pth` (best
  validation MSE, or best train MSE when no val split).
- Metrics: `<out>/metrics.csv` — train/val MSE, predicted-action
  reconstruction error (wrist mm / wrist rad / finger rad: compose the
  predicted actions and compare to `ref[t+h]`), prediction clip fraction.
  `<out>/metadata.json` — label convention, dims, clips, per-clip label stats,
  git commit. Per-clip diagnostics (label clipping %, per-frame wrist deltas,
  label-inversion reconstruction error) print at startup.

## Warm-starting RL

```bash
python dexmachina/rl/train_rl_games.py --oakink --oakink_pt <clip>.pt \
    --hand allegro_hand -B 12000 -am hybrid --hybrid_scales 0.1 1.0 ... \
    --warmstart_ckpt logs/bc_kinref/allegro_oakink/nn/bc_best.pth \
    --warmstart_sigma -1.6
```

The loader prints the BC metadata, warns on `--hybrid_scales` mismatch, and on
a dim mismatch reports exactly which keys/shapes diverge. Weights-only load:
fresh optimizer, epoch 0, curriculum/LR schedules start clean.

Recommended fine-tuning safeguards (the critic starts random and PPO will
happily unlearn the actor to satisfy it):

- `--warmstart_sigma -1.6` (std ≈ 0.2) so early rollouts stay near the BC
  policy instead of sampling with the default std = 1.0.
- Consider a lower initial `-lr` for the first epochs; the adaptive-KL
  schedule takes over quickly.
- The existing `-bc/--bc_rew_weight` reward (stay-near-kinref shaping) is a
  natural regularizer against BC forgetting during early critic training.
- Mind the fork's hard early-stop: mean reward < 35 at epoch > 2500 with no
  curriculum kills the run (`rl_games a2c_common.py`).

## Known failure modes

| symptom | likely cause / fix |
|---|---|
| high label clip % warning | wrong `--hybrid_scales`, wrong `--label_horizon`, or jerky references; inspect per-clip wrist-delta diagnostics |
| shape mismatch at `--warmstart_ckpt` | BC data layout ≠ RL clip layout (two-object 341 vs shared 319), or different hand |
| ValueError: joint set mismatch | clip exported for another hand, or exporter convention changed — bc_dataset's canonical dof order must be re-verified (`verify_bc_obs.py`) |
| BC val MSE plateaus high | expected with few clips (observation aliasing across clips); warm-start quality is driven by train-clip fit when fine-tuning the same clip |
| RL performance collapses right after warm-start | random-critic unlearning: lower sigma, keep `-bc` shaping on, shorten critic warmup (see safeguards above) |

## Limitations (v1)

- OakInk clips only (ARCTIC blocked by the contact-force obs block).
- Allegro only (other hands need mimic-joint action-map handling).
- Single-sequence `train_rl_games.py` only (`train_rl_games_multi_sequence.py`
  appends task-id features BC does not build).
- The 1-frame post-reset object-obs zeroing and sim-vs-finite-difference
  velocities are deliberate approximations; RL fine-tuning absorbs them.
