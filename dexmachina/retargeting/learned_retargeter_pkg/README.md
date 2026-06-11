# learned-retargeter-inference

Self-contained, inference-only wrapper around the learned MANO→robot retargeter.
Designed as a drop-in replacement for a downstream imitator network (e.g. in
ManipTrans). Depends only on `torch` and `numpy`.

## What you ship

This directory is the entire package:

```
inference/
├── retargeter.py     # public entry point: Retargeter.load / predict_step / predict_sequence
├── preprocess.py     # MANO->wrist-local pack + object sampling
├── _model.py         # vendored Stage1MLPBaseline
├── _geometry.py      # vendored Stage1RobotGeometry
├── _hand_fk.py       # vendored URDF FK
├── _hand_profile.py  # vendored hand profile loader
├── _frames.py        # vendored frame transforms
└── __init__.py       # re-exports Retargeter
```

Plus, separately, the **runtime assets** (not in this directory):

- the trained checkpoint (`model_state.pth`)
- the hand profile JSON
- the URDF + mesh files referenced from the hand profile

## Install

Pick one:

**A. Copy-paste.** Drop the `inference/` directory into your tree (rename as
you like, e.g. `maniptrans/retargeter/`) and import from it.

**B. Editable install.** From the repo root:

```bash
uv pip install -e learned_retargeter/inference
```

Either way, runtime deps are just `torch` and `numpy`.

## Usage

```python
import numpy as np
from learned_retargeter.inference import Retargeter  # or: from <your-name> import Retargeter

ret = Retargeter.load(
    ckpt_path="path/to/model_state.pth",   # or a directory containing it
    hand_config_path="path/to/hand.json",  # the URDF + fingertip mapping
    device="cuda",
    fps=30.0,
)

# Online / per-step (ManipTrans env rollout).
ret.reset()  # clears AR state + sliding-window buffer
for t in range(T):
    out = ret.predict_step(
        mano_kpts=mano_kpts_t,    # [21, 3]  world frame
        obj_points=obj_points_t,  # [N, 3]   world frame (raw, any N)
        wrist_world=wrist_world_t,  # [4, 4]  MANO wrist transform (world)
        obj_normals=normals_t,    # [N, 3]   optional; PCA-estimated if None
        hand_rot6d=rot6d_t,       # [21, 6]  optional; zeros if None
    )
    q_target = out["pred_q"]                  # [J] joint angles (rad, in URDF limits)
    wrist_pose = (out["pred_wrist_pos_world"],
                  out["pred_wrist_rot6d_world"])  # base-link pose in MANO-wrist frame

# Bulk / offline (data preprocessing).
seq = ret.predict_sequence(
    mano_kpts=mano_traj,        # [T, 21, 3]
    obj_points=obj_traj,        # [T, N, 3]
    wrist_world=wrist_traj,     # [T, 4, 4]
)
# Returns the same keys with a leading T dimension.
```

## Output schema

`predict_step` returns a dict (`predict_sequence` adds a leading `T`):

| Key | Shape (step) | Meaning |
|---|---|---|
| `pred_q` | `[J]` | Joint angles in radians, clamped to URDF limits. Directly usable as a target. |
| `pred_q_raw` | `[J]` | Pre-sigmoid decoder output. Mainly for debugging. |
| `pred_tips` | `[F, 3]` | Fingertip positions in **retarget-origin frame** (model FK output, no wrist transform applied). |
| `pred_wrist_pos_world` | `[3]` | Predicted wrist (= robot base_link) translation, in the model's **native frame**. |
| `pred_wrist_rot6d_world` | `[6]` | Predicted wrist rotation (6D Gram–Schmidt form). |
| `obj_points` | `[K, 3]` | Resampled object points in supervision-wrist-local frame (the actual model input). |

### Frame conventions

`pred_wrist_*` is in the model's **training-native frame**, which depends on
the checkpoint:

- **base-trained ckpts** (`--predicted-wrist-frame base`, e.g. `qsup_wsup`)
  → output is the robot base_link pose, expressed in the MANO-wrist supervision
  frame. Drop in directly as your robot wrist target.
- **origin-trained ckpts** (default) → output is the retarget-origin pose,
  expressed in the MANO-wrist supervision frame. Lift to base-link via the
  static URDF transform `T^B_O` (computed from FK at q=0; see
  `_compute_base_to_origin_transform` in the source).

To get the robot wrist pose in **world frame**, compose with the demo MANO
wrist transform: `T_world_robot = T_world_mano_wrist @ T_mano_wrist_robot`.

### AR / window semantics

The wrapper mirrors training and eval exactly:

- Checkpoints with `window_size > 1` (typical): each `predict_step` builds a
  sliding window of `window_size` frames (edge-padded with frame 0 on the
  first calls). Autoregressive state rolls **inside** the window only; it
  resets at every call. Identical to training.
- Checkpoints with `window_size == 1` (streaming): autoregressive state carries
  call-to-call via internal `q_prev_hint` / `wrist_prev_hint`. Call
  `ret.reset()` at sequence boundaries.

Verified to match the in-repo reference forward path to machine precision
(L∞ diff = 0) on real OakInk data with the `qsup_wsup` W=9 AR checkpoint.

### Notes on object inputs

- `obj_points` may have any `N` per frame; the wrapper resamples to the model's
  `K` (read it from `ret.object_points`) using a per-frame RNG.
- If you don't provide `obj_normals`, they're estimated via local-PCA on the
  point cloud each frame. For grasp-relevant geometry that's fine for convex
  objects and reasonable for mildly concave ones. Pass pre-computed normals
  when available for better fidelity.
- The training-side pipeline picks K bank-indices **once per sequence** and
  reuses them; the wrapper resamples each frame. The model is permutation-
  invariant over object points (PointNet→max-pool), so this is functionally
  fine, but you won't bit-exactly reproduce training-time predictions.

## Quick reference: instance attributes

```python
ret.joint_dim         # int — robot DOF
ret.window_size       # int — model's training window size
ret.object_points     # int — K (number of object points the model expects)
ret.joint_limits      # (lower, upper) — np.ndarray [J] each
```
