"""Offline BC dataset from exported OakInk kinematic-reference clips (.pt).

Builds teacher-forced (observation, action) pairs for behavioral cloning of the
rl_games PPO actor WITHOUT the simulator: the robot is assumed to track the
kinematic reference exactly, so every observation component is reconstructed
from the reference/demo arrays and every action label is the exact inverse of
the env's hybrid action composition.

Conventions mirrored (do not change one without the other):
- hybrid action composition + EMA/clamp chain: envs/robot.py::translate_actions
  (wrist target = ref[t] + hybrid_scales * a, fingers absolute limit-normalized;
  action_moving_avg == 1.0 in all configs so the EMA is a no-op; Allegro has no
  mimic joints).
- reference loading (forearm 2pi re-seating + per-clip wrist limits):
  envs/demo_data.py::load_genesis_retarget_data. Never read joint_qpos raw.
- observation assembly order and scaling: envs/base_env.py::get_observations,
  envs/robot.py::get_observations, envs/object.py::get_observations.

The canonical actuated-dof order below was dumped from a live Genesis env
(robot.actuated_dof_names); rl/verify_bc_obs.py re-checks it against the env.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from dexmachina.envs.demo_data import load_genesis_retarget_data
from dexmachina.envs.math_utils import quat_conjugate, quat_mul
from dexmachina.envs.robot import get_default_robot_cfg

# Genesis orders links breadth-first by kinematic-tree depth, so the 16 finger
# joints interleave across the 4 fingers by level (NOT per-finger consecutive).
_FOREARM_AXES = ("tx", "ty", "tz", "roll", "pitch", "yaw")
_FINGER_DOF_NAMES = [
    "joint_0.0", "joint_4.0", "joint_8.0", "joint_12.0",
    "joint_1.0", "joint_5.0", "joint_9.0", "joint_13.0",
    "joint_2.0", "joint_6.0", "joint_10.0", "joint_14.0",
    "joint_3.0", "joint_7.0", "joint_11.0", "joint_15.0",
]
WRIST_SLICE = slice(0, 6)
FINGER_SLICE = slice(6, 22)
NDOF = 22
SIM_FPS = 60.0  # one reference frame per control step, dt = 1/60 (envs/base_env.py)
OBS_CLIP = 5.0  # env_cfg['obs_clip']; rl_games clip_observations matches


def allegro_dof_names(side: str) -> list[str]:
    prefix = {"left": "L_", "right": "R_"}[side]
    forearm = [f"{prefix}forearm_{ax}_link_joint" for ax in _FOREARM_AXES]
    return forearm + _FINGER_DOF_NAMES


def _urdf_joint_limits(urdf_path: str) -> dict[str, tuple[float, float]]:
    """Joint limits by name for revolute/prismatic joints, straight from the URDF."""
    limits = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


@dataclass(frozen=True)
class HandRef:
    """Per-hand reference trajectory in the canonical env dof order."""

    ref_qpos: torch.Tensor      # (T, 22) float32, post wrap_forearm_qpos_into_limits
    dof_limits: torch.Tensor    # (22, 2) URDF finger limits + per-clip wrist limits
    kpt_pos: torch.Tensor       # (T, n_kpts, 3) world frame (exporter FK)
    wrist_pose: torch.Tensor    # (T, 7) [pos, quat wxyz] world frame


@dataclass(frozen=True)
class ObjectRef:
    name: str
    pos: torch.Tensor           # (T, 3)
    quat: torch.Tensor          # (T, 4) wxyz
    arti: torch.Tensor          # (T, 1)


@dataclass(frozen=True)
class ClipData:
    path: str
    num_frames: int
    hands: dict[str, HandRef]   # 'left', 'right' (env robot order)
    objects: list[ObjectRef]    # env object order: left hand's object, then right's


def _to_f32(x) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x))
    return t.to(torch.float32)


def _assert_finite(name: str, t: torch.Tensor, path: str) -> None:
    if not torch.isfinite(t).all():
        raise ValueError(f"{path}: non-finite values in '{name}'")


def load_clip(pt_path: str, hand: str = "allegro_hand") -> ClipData:
    """Load one exported OakInk clip exactly as the env does (constructors.py)."""
    raw = torch.load(pt_path, weights_only=False)
    demo_raw = raw["demo_data"]
    if "objects" not in demo_raw:
        raise NotImplementedError(
            f"{pt_path}: not an OakInk export. ARCTIC clips are unsupported: the ARCTIC "
            "env forces an 84-dim sim-measured contact-force obs block (constructors.py) "
            "that cannot be reconstructed offline."
        )
    num_frames = len(next(iter(demo_raw["objects"].values()))["obj_pos"])
    demo, retarget = load_genesis_retarget_data(
        given_data_fname=pt_path, frame_start=0, frame_end=num_frames
    )

    hands = {}
    for side in ("left", "right"):
        dof_names = allegro_dof_names(side)
        qpos_dict = retarget[side]["residual_qpos"]
        missing = set(dof_names) - set(qpos_dict)
        extra = set(qpos_dict) - set(dof_names)
        if missing or extra:
            raise ValueError(
                f"{pt_path} [{side}]: joint set mismatch vs {hand}; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        ref = torch.stack([_to_f32(qpos_dict[n]) for n in dof_names], dim=1)

        urdf_limits = _urdf_joint_limits(get_default_robot_cfg(name=hand, side=side)["urdf_path"])
        limits = torch.tensor([urdf_limits[n] for n in dof_names], dtype=torch.float32)
        # Per-clip wrist limits override URDF ones, forearm joints only
        # (robot.py::set_custom_joint_limits skips everything else).
        for jname, lim in retarget[side]["limits"].items():
            if "forearm" in jname:
                limits[dof_names.index(jname)] = torch.tensor(lim, dtype=torch.float32)

        hands[side] = HandRef(
            ref_qpos=ref,
            dof_limits=limits,
            kpt_pos=_to_f32(retarget[side]["kpts_data"]["kpt_pos"]),
            wrist_pose=_to_f32(retarget[side]["wrist_pose"]),
        )

    # Env object order = object_cfgs insertion order: left hand's object first,
    # then the right's unless it is the same (shared-object clip) — constructors.py.
    objects = []
    seen = set()
    for key in ("object_left", "object_right"):
        name = demo[key]
        if name in seen:
            continue
        seen.add(name)
        od = demo["objects"][name]
        arti = _to_f32(od["obj_arti"]).reshape(num_frames, -1)
        objects.append(
            ObjectRef(name=name, pos=_to_f32(od["obj_pos"]), quat=_to_f32(od["obj_quat"]), arti=arti)
        )

    clip = ClipData(path=pt_path, num_frames=num_frames, hands=hands, objects=objects)
    for side, h in clip.hands.items():
        for fname in ("ref_qpos", "kpt_pos", "wrist_pose"):
            arr = getattr(h, fname)
            _assert_finite(f"{side}.{fname}", arr, pt_path)
            if arr.shape[0] != num_frames:
                raise ValueError(f"{pt_path}: {side}.{fname} has {arr.shape[0]} frames != {num_frames}")
    for obj in clip.objects:
        for fname in ("pos", "quat", "arti"):
            _assert_finite(f"{obj.name}.{fname}", getattr(obj, fname), pt_path)
    return clip


def hybrid_scale_vector(hybrid_scales: tuple[float, float]) -> torch.Tensor:
    scale_trans, scale_rot = hybrid_scales
    return torch.tensor([scale_trans] * 3 + [scale_rot] * 3, dtype=torch.float32)


def hybrid_action_labels(
    ref_qpos: torch.Tensor,
    dof_limits: torch.Tensor,
    hybrid_scales: tuple[float, float],
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher actions that make the hybrid controller command ref[t+horizon] at step t.

    Exact inverse of robot.py::translate_actions (hybrid branch): the wrist rows
    are the delta formula a = (ref[t+h] - ref[t]) / hybrid_scales because the env
    anchors the wrist target on ref[t] (demo_t = current step); the finger rows
    are absolute joint-limit-normalized because hybrid fingers ignore the
    reference. horizon=0 reproduces the 'kinematic' replay teacher (wrist == 0).

    Returns (unclipped labels, clipped labels), both (T - horizon, 22).
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    T = ref_qpos.shape[0]
    if T <= horizon:
        raise ValueError(f"clip too short: {T} frames <= horizon {horizon}")
    n = T - horizon
    target = ref_qpos[horizon:horizon + n]
    anchor = ref_qpos[:n]

    labels = torch.empty((n, NDOF), dtype=torch.float32)
    labels[:, WRIST_SLICE] = (target[:, WRIST_SLICE] - anchor[:, WRIST_SLICE]) / hybrid_scale_vector(hybrid_scales)
    lo = dof_limits[FINGER_SLICE, 0]
    hi = dof_limits[FINGER_SLICE, 1]
    labels[:, FINGER_SLICE] = 2.0 * (target[:, FINGER_SLICE] - lo) / (hi - lo) - 1.0
    return labels, torch.clamp(labels, -1.0, 1.0)


def compose_hybrid_targets(
    actions: torch.Tensor,
    ref_t: torch.Tensor,
    dof_limits: torch.Tensor,
    hybrid_scales: tuple[float, float],
) -> torch.Tensor:
    """Forward hybrid composition, mirroring robot.py::translate_actions.

    actions (N, 22) in [-1, 1] (clamped here like the env's action_clip),
    ref_t (N, 22) = residual_qpos at the anchor step. EMA is a no-op
    (action_moving_avg == 1.0) and Allegro has no mimic joints.
    """
    a = torch.clamp(actions, -1.0, 1.0)
    targets = torch.empty_like(ref_t)
    targets[:, WRIST_SLICE] = ref_t[:, WRIST_SLICE] + hybrid_scale_vector(hybrid_scales) * a[:, WRIST_SLICE]
    lo = dof_limits[FINGER_SLICE, 0]
    hi = dof_limits[FINGER_SLICE, 1]
    targets[:, FINGER_SLICE] = lo + (hi - lo) * (a[:, FINGER_SLICE] + 1.0) / 2.0
    return torch.clamp(targets, dof_limits[:, 0], dof_limits[:, 1])


@dataclass
class ClipDiagnostics:
    path: str
    num_frames: int
    num_samples: int
    clipped_fraction: float                 # fraction of label entries outside [-1, 1]
    clipped_per_dim: np.ndarray             # (44,) fraction per action dim
    label_mean: np.ndarray                  # (44,)
    label_std: np.ndarray
    label_min: np.ndarray
    label_max: np.ndarray
    max_wrist_trans_delta: float            # m per frame, raw reference delta
    max_wrist_rot_delta: float              # rad per frame
    recon_err_max: float                    # max |composed target - ref[t+h]| over unclipped dims

    def summary(self) -> str:
        return (
            f"{self.path}: frames={self.num_frames} samples={self.num_samples} "
            f"clipped={100 * self.clipped_fraction:.3f}% "
            f"wrist_delta_max=({self.max_wrist_trans_delta * 1000:.1f} mm, "
            f"{np.degrees(self.max_wrist_rot_delta):.2f} deg)/frame "
            f"recon_err_max={self.recon_err_max:.2e}"
        )


def clip_action_labels(
    clip: ClipData,
    hybrid_scales: tuple[float, float],
    horizon: int,
) -> tuple[torch.Tensor, ClipDiagnostics]:
    """Full-env action labels (T - horizon, 44) = [left 22, right 22] + diagnostics."""
    per_side = {}
    raw_per_side = {}
    for side in ("left", "right"):
        h = clip.hands[side]
        raw, clipped = hybrid_action_labels(h.ref_qpos, h.dof_limits, hybrid_scales, horizon)
        per_side[side] = clipped
        raw_per_side[side] = raw
    labels = torch.cat([per_side["left"], per_side["right"]], dim=1)
    raw = torch.cat([raw_per_side["left"], raw_per_side["right"]], dim=1)

    n = labels.shape[0]
    # Reconstruction check: composing the clipped labels must land on ref[t+h]
    # wherever the label was not clipped (catches any convention drift).
    recon_err = 0.0
    for side, off in (("left", 0), ("right", NDOF)):
        h = clip.hands[side]
        targets = compose_hybrid_targets(
            per_side[side], h.ref_qpos[:n], h.dof_limits, hybrid_scales
        )
        unclipped = (raw[:, off:off + NDOF].abs() <= 1.0)
        err = (targets - h.ref_qpos[horizon:horizon + n]).abs()
        if unclipped.any():
            recon_err = max(recon_err, float(err[unclipped].max()))

    wrist_deltas = torch.cat(
        [(clip.hands[s].ref_qpos[1:, WRIST_SLICE] - clip.hands[s].ref_qpos[:-1, WRIST_SLICE]).abs()
         for s in ("left", "right")]
    )
    clipped_mask = (raw.abs() > 1.0).float()
    diag = ClipDiagnostics(
        path=clip.path,
        num_frames=clip.num_frames,
        num_samples=n,
        clipped_fraction=float(clipped_mask.mean()),
        clipped_per_dim=clipped_mask.mean(dim=0).numpy(),
        label_mean=labels.mean(dim=0).numpy(),
        label_std=labels.std(dim=0).numpy(),
        label_min=labels.min(dim=0).values.numpy(),
        label_max=labels.max(dim=0).values.numpy(),
        max_wrist_trans_delta=float(wrist_deltas[:, :3].max()),
        max_wrist_rot_delta=float(wrist_deltas[:, 3:].max()),
        recon_err_max=recon_err,
    )
    return labels, diag
