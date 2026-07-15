# Offline BC warm-start from kinematic references — implementation plan

Status: plan (implementation follows in subsequent commits on this branch).

Goal: teacher-forced behavioral cloning of the rl_games PPO actor from kinematic
retargeted reference trajectories only — no simulator rollouts, no env stepping,
no critic pretraining. The BC checkpoint must be loadable by the existing
`--warmstart_ckpt` path of `dexmachina/rl/train_rl_games.py` for RL fine-tuning.

## 1. Which downstream RL policy

The target is the **DexMachina rl_games PPO stack** (`dexmachina/rl/train_rl_games.py`,
bimanual Allegro, Genesis) in **`action_mode='hybrid'`** — the mode every recent
production run uses (`training_scripts/oakink_continuous.slurm:60`,
`train_array_graph.slurm:57`, `train_array_o2a_contact.slurm:47`; all pass
`-am hybrid --hybrid_scales 0.1 1.0`).

Why not `learned_retargeter/residual_rl` (the ManiSkill PPO stack): its action
composition is purely residual around the reference
(`target = kin_q[t+1] + action*scale`), so a teacher that tracks the reference
has action ≡ 0 — and its `ResidualAgent` already small-inits to zero output.
BC-from-references is degenerate there. In DexMachina hybrid mode the 16 finger
DoFs per hand are **absolute** joint-limit-normalized targets the policy must
actually learn, and the 6 wrist DoFs are residual around the reference — BC is
meaningful.

## 2. Audit facts the implementation relies on

Actor / config (`dexmachina/rl/configs/rl_games_ppo_cfg.yaml`, last of the 3
duplicate `params:` blocks wins under `yaml.full_load`):

- Network `actor_critic`, `separate: False` — shared MLP trunk
  `[512, 512, 256, 128]`, ELU, PyTorch-default init; heads `mu = Linear(128, A)`,
  `value = Linear(128, 1)`; `fixed_sigma: True` → `sigma = nn.Parameter(zeros(A))`
  (state-independent log-std, init 0 → std 1.0).
- Model `continuous_a2c_logstd` (`ModelA2CContinuousLogStd`).
- `normalize_input: False` (**no obs running-mean-std anywhere**),
  `normalize_value: True` (`value_mean_std` is a submodule of the model),
  `clip_observations: 5.0`, `clip_actions: 1.0`.
- Checkpoint = `{'model': OrderedDict, 'epoch', 'frame', 'optimizer',
  'last_mean_rewards', 'env_state'}`; under `model`:
  `a2c_network.{actor_mlp.0/2/4/6, mu, value, sigma}` +
  `value_mean_std.{running_mean, running_var, count}` (verified on a real run's
  `nn/allegro_hand.pth`).
- `--warmstart_ckpt` (`train_rl_games.py:191-212`) already implements
  weights-only warm-start: `torch_ext.load_checkpoint` → `agent.set_weights`,
  fresh optimizer/epoch, optional `--warmstart_sigma` log-std refill.
  `set_weights` = strict `model.load_state_dict(weights['model'])`;
  `set_stats_weights` no-ops for this config. **Contract: the BC checkpoint must
  contain the FULL model state dict (value head + value_mean_std included) under
  key `'model'`.**

Controller (hybrid; `dexmachina/envs/robot.py:564-644`):

- Env: `actions = clamp(a, ±1.0) * 1.0` (`base_env.py:653`); left hand =
  `a[0:22]`, right = `a[22:44]`.
- `demo_t = min(episode_length_buf, num_frames-1)` — reference at the **current**
  step t (`robot.py:573-577`); `curr_res_qpos = residual_qpos[demo_t]`.
- Wrist (dofs 0-5 = forearm tx,ty,tz,roll,pitch,yaw):
  `target = curr_res_qpos + hybrid_scales * clamp(a_wrist, ±1)`
  with `hybrid_scales = (0.1 m, 1.0 rad)` in production (`robot.py:627-628`).
- Fingers (dofs 6-21): `target = lo + (hi-lo)*(a+1)/2` — absolute,
  URDF limits (`robot.py:630-631`).
- Then `clamp(target, dof_limits)` (per-clip wrist limits! see below), EMA with
  `action_moving_avg = 1.0` (no-op), re-clamp (`robot.py:638-643`).
- No mimic joints on Allegro → identity `joint_from_idxs`, `ndof = action_dim = 22`/hand.
- Reset: `dof_pos = curr_targets = prev_targets = residual_qpos[start]`,
  `dof_vel = 0` (`robot.py:675-692`).
- 1 control step per reference frame, `dt = 1/60`, no decimation
  (`base_env.py:26-58,662`); OakInk refs are 60 Hz (1× speed), ARCTIC 30 Hz
  (played 2×; `KINREF_FPS_SYNC.md`).

Observation (`base_env.py:961-1021`; OakInk allegro hybrid):

| block | dim | content |
|---|---|---|
| per hand (left, then right) | 148 | `dof_target_pos = curr_targets − dof_pos` (22, raw); `dof_pos` unscaled to ≈[-1,1] with **per-clip wrist limits** (22); `dof_vel × 0.1` (22); `kpt_pos` world, flat (25×3); `wrist_pose` [pos, quat wxyz] (7) |
| per object (left's, then right's; one if shared) | 22 | `parts_pos` (3), `parts_quat` wxyz (4), `dof_pos` (1), `state_diff = demo[min(t+1,T-1)] − state[t]` (8), `root_ang_vel × 0.25` (3), `root_lin_vel × 2.0` (3) |
| episode phase | 1 | `2·t/max_episode_length − 1`, t = absolute demo frame |

Totals: OakInk two-object = **341**, shared-object = **319**. NaN→−5, clamp ±5.
`observe_tip_dist`/`observe_contact_force` are forced **off** for OakInk
(`constructors.py:102-103`). ARCTIC forces an 84-dim sim-measured contact-force
block **on** (`constructors.py:238-239`) — not reconstructible offline.

Kinref `.pt` (`torch.save({'demo_data', 'retarget_data'})`):

- `retarget_data[side]`: `joint_qpos` dict `{joint_name → (T,)}` (22 names:
  `{L_,R_}forearm_{tx,ty,tz,roll,pitch,yaw}_link_joint` + `joint_0.0..15.0`),
  `kpt_pos (T,25,3)` (FK of the stored joints — self-consistent), `kpt_names`,
  `wrist_pose (T,7)` wxyz, `wrist_link_name`.
- `demo_data` (OakInk): `objects {obj_id → {obj_pos (T,3), obj_quat (T,4) wxyz,
  obj_arti (T,)}}`, `object_left`/`object_right` hints, optional
  `contact_links_{side}` (reward-only, never obs).
- **Loading must go through `dexmachina.envs.demo_data.load_genesis_retarget_data`**:
  it applies `wrap_forearm_qpos_into_limits` (±12 rad re-seating) and derives the
  per-clip wrist limits (data min/max ± 0.2 m / 0.5 rad) that the env uses for
  the `dof_pos` obs normalization and target clamping. Reading the `.pt` raw can
  differ by 2π on the wrist.
- No velocity fields — finite-difference at the env-consumed rate (60 Hz).
- Local OakInk allegro clips (5): `dexmachina/assets/retargeted/allegro_hand/oakink/`
  `{e76b2_at3, 001ce74, 1469a_at0, 4102c_at0, 67132_at0}_vector_oakink.pt`.

## 3. Action-label formula (mirrors `robot.py:620-643`)

Teacher: the PD target commanded at obs time t should be the reference pose at
`t+h` (`--label-horizon h`, default 1 — the pose the hand should reach by the end
of the step; matches the reward, which compares post-step state to demo `t+1`).
Per hand, with `ref = residual_qpos` after `load_genesis_retarget_data`:

```
# wrist dofs 0-5 (env composes target = ref[t] + hybrid_scales * a):
a_wrist[t] = (ref[t+h, 0:6] − ref[t, 0:6]) / (s_trans, s_trans, s_trans, s_rot, s_rot, s_rot)

# finger dofs 6-21 (env composes target = lo + (hi-lo)(a+1)/2, URDF limits):
a_finger[t] = 2 · (ref[t+h, 6:22] − lo) / (hi − lo) − 1

a[t] = clip(concat(a_wrist, a_finger), −1, 1)        # counted + warned
```

`h=0` reproduces the `kinematic` action mode's teacher (wrist ≡ 0, fingers =
current frame). Labels for the full env action = concat(left 22, right 22).
Frames `t > T−1−h` are dropped (boundary). This is NOT the naive
`Δq/scale` formula on all dims — fingers are absolute in hybrid mode; the wrist
rows are exactly the delta formula with the hybrid scales.

Consistency check baked into the dataset: with h=1 the teacher's
`curr_targets(t) = ref[t]` (commanded at t−1), so the `dof_target_pos` obs block
is identically 0 under perfect tracking — matching the env's reset state.

## 4. Teacher-forced observation reconstruction (offline, no sim)

Assume perfect tracking: `dof_pos[t] = ref[t]`, object state = demo[t].

- `dof_target_pos` = `clamp(ref[t−1+h], dof_limits) − ref[t]` (= 0 for h=1 up to
  limit clamping); 0 at t=0 (reset).
- `dof_pos` = `unscale(ref[t], limits)` with the loader's per-clip wrist limits +
  URDF finger limits (limits taken from `retarget_data['limits']` exactly as
  `set_custom_joint_limits` applies them: forearm joints only).
- `dof_vel` = `(ref[t] − ref[t−1]) · 60 · 0.1`; 0 at t=0.
- `kpt_pos`, `wrist_pose` = the `.pt` arrays at t (already FK world-frame).
- Object: `parts_pos/quat` = demo pos/quat[t] (OakInk rigid: 1 link = root);
  `dof_pos` = `obj_arti[t]` (zeros); `state_diff` = `demo[min(t+1, T−1)] − demo[t]`
  (componentwise, raw quat subtraction); `root_lin_vel` = fd(pos)·60·2.0;
  `root_ang_vel` = axis-angle(q[t]·q[t−1]⁻¹)·60·0.25 (via `math_utils.quat_mul`/
  `quat_conjugate`); both 0 at t=0.
- Phase = `2t/T − 1` (OakInk `max_episode_length` = full clip length,
  `constructors.py:78,101`).
- Final `clamp(±5)`; NaN anywhere → hard error (offline data must be clean).

Known, documented mismatches vs the live env (accepted; RL fine-tuning corrects):

1. The env zeroes the object obs block on the first frame after every reset
   (`object.py:489-500` — not refreshed before the first `get_observations`).
   BC emits steady-state values instead (1 frame per episode in RL).
2. `dof_vel`/object velocities in RL are sim-measured (PD tracking error,
   contacts); BC uses reference finite differences.
3. Sim contact-force obs (ARCTIC only) cannot be reconstructed → **ARCTIC clips
   are rejected with a clear error in v1**; OakInk is unaffected (block absent).

Verification: `dexmachina/rl/verify_bc_obs.py` (optional, GPU) replays a clip
kinematically in the real env (`env.set_retarget_states` /
`robot.set_joint_position`, pattern of `rl/replay_oakink.py`) and reports
per-block max |Δobs| vs the offline builder. Run once per new data source; not
required for training.

## 5. Files

New (dexmachina repo, branch `bc-warmstart`):

| file | contents |
|---|---|
| `dexmachina/rl/bc_dataset.py` | `.pt` loading (via `load_genesis_retarget_data`), hybrid action labels, offline obs builder, per-clip diagnostics dataclass, trajectory-level train/val split, torch `Dataset` |
| `dexmachina/rl/train_bc_kinref.py` | entry point: seeds, dataset build, rl_games `ModelBuilder` actor from `rl_games_ppo_cfg.yaml`, MSE training loop, latest/best-val checkpoints in rl_games format, metrics + metadata JSON |
| `dexmachina/rl/verify_bc_obs.py` | optional GPU obs-parity check vs live env |
| `dexmachina/tests/test_bc_labels.py` | label formula on synthetic refs (linear motion, horizon, clipping, boundaries) |
| `dexmachina/tests/test_bc_obs.py` | obs dims + component values on synthetic clip |
| `dexmachina/tests/test_bc_checkpoint.py` | BC ckpt ↔ rl_games model strict-load round-trip; tiny overfit test |
| `BC_WARMSTART.md` | user doc (final commit) |

Modified:

- `dexmachina/rl/train_rl_games.py` — in the `--warmstart_ckpt` path: print BC
  metadata if present and a key/shape diff on mismatch (few lines).
- `README.md` — pointer subsection under the custom-fork section.

No new dependencies (pytest 9.0.2 already in the working venv; JZ `uv.lock`
untouched).

## 6. Training entry point & checkpoint format

```
# venv: /home/fspinola/venvs/dexmachina-venv2 (torch 2.9.1+cu128, rl_games fork,
# dexmachina editable). Repo-root .venv torch is broken on this box.
python dexmachina/rl/train_bc_kinref.py \
    --data dexmachina/assets/retargeted/allegro_hand/oakink/*_vector_oakink.pt \
    --val-clips 1 --epochs 200 --batch-size 4096 --lr 1e-3 \
    --hybrid-scales 0.1 1.0 --label-horizon 1 \
    --out logs/bc_kinref/<exp>
```

Checkpoint (`<out>/nn/bc_latest.pth`, `bc_best.pth`):
`{'model': <full model state dict>, 'epoch': 0, 'optimizer': None, 'frame': 0,
'last_mean_rewards': -inf, 'env_state': None, 'bc_metadata': {...}}` — extra key
is ignored by `torch_ext.load_checkpoint`/`set_weights`. `sigma` stays at the
yaml init (logstd 0); use `--warmstart_sigma` at RL time to lower initial
exploration noise. Value head random, `value_mean_std` fresh — **the critic is
NOT initialized by BC.**

Metadata JSON (also embedded in the ckpt): label convention + code refs, horizon,
hybrid_scales, data paths, obs/action dims, per-dim label stats, clip fractions,
git commit, seed, config.

## 7. RL fine-tuning

```
python dexmachina/rl/train_rl_games.py ... -am hybrid --hybrid_scales 0.1 1.0 \
    --warmstart_ckpt logs/bc_kinref/<exp>/nn/bc_best.pth --warmstart_sigma -1.6
```

Safeguards (documented, not new code): the critic starts random —
`value_mean_std` fresh, advantage normalization on; consider `--warmstart_sigma`
≈ −1.6 (σ≈0.2) so early rollouts stay near the BC policy, and a lower initial
`--learning_rate` (adaptive-KL schedule adjusts quickly). Note the fork's
hard early-stop (mean reward < 35 at epoch > 2500 when no curriculum,
`a2c_common.py:1502`) applies to warm-started runs too.

## 8. Test plan

1. Labels: synthetic constant-velocity refs, known scales → exact expected
   labels; horizon 0/1/2; clipping counted; last frames dropped; short clips
   rejected.
2. Obs: synthetic clip → dim formula (319/341), per-block values, phase range,
   t=0 velocity zeros, dof_target_pos ≡ 0 for h=1.
3. Checkpoint: build model via `ModelBuilder` from the real yaml → save BC ckpt →
   fresh model `load_state_dict(ckpt['model'], strict=True)`; assert key sets
   equal to a reference constructed the way `A2CAgent` does.
4. Overfit: 200 synthetic samples, loss ↓ ≥ 100×.
5. Integration (manual, GPU): `verify_bc_obs.py` on `e76b2_at3`; documented in
   BC_WARMSTART.md.

## 9. Risks / mitigations

| risk | mitigation |
|---|---|
| wrong action convention | labels mirror `robot.py:620-643` exactly (comment with line refs); forward-composition reconstruction error reported per epoch |
| wrong joint order | canonical allegro dof order hardcoded + asserted against the `.pt` `joint_qpos` keys; name-based column mapping like `set_residual_qpos`; `verify_bc_obs.py` checks against `robot.actuated_dof_names` |
| wrist 2π / limits drift | always load via `load_genesis_retarget_data` (never raw `.pt`) |
| timestep misalignment | `--label-horizon` configurable; reconstruction diagnostic catches off-by-one (error ≈ one-frame delta) |
| excessive clipping | per-dim clip fractions; hard warning above `--clip-warn-frac` (default 1%); wrist-rot saturation vs the 0.5 rad per-clip limit margin asserted |
| obs mismatch BC↔RL | offline builder mirrors `get_observations` field-by-field; GPU parity script; dims asserted vs `actor_mlp.0.weight` when warm-starting |
| infeasible references | BC = actor init only; critic untouched; documented |
| critic unlearning at RL start | documented: `--warmstart_sigma`, lower lr; optional actor-freeze NOT built (YAGNI — revisit if fine-tuning shows it) |
| obs aliasing | obs already contains phase + object goal diff (same as RL); no extra features added |
| BC overfitting | trajectory-level split; best-val checkpoint; per-dim stats. For single-clip warm-starts overfitting the clip is intended |

## 10. Out of scope (v1)

ARCTIC clips (contact-force obs block), `train_rl_games_multi_sequence.py`
(unused by any slurm), non-Allegro hands (labels need mimic-map handling),
DAgger/distributional losses, simulator-rollout evaluation.
