#!/usr/bin/env python3
"""Produce a DexMachina kinematic-reference .pt from a learned retargeter.

Inputs:
- A learned_retargeter checkpoint (Allegro right hand, e.g. qsup_wsup or geometric-only)
- An ARCTIC raw .npz from the learned_retargeter project
  (data/arctic_rh_with_objects_train/<seq>.npz)
- An existing _vector_para.pt -- used as the *source of truth* for
    (a) the frame range to retarget over (matched by aligning frame 0)
    (b) the bimanual left-side reference (copied verbatim -- DexMachina's
        loader requires both 'left' and 'right' keys)
    (c) demo_data (already sliced to the same range)

Output:
- One .pt with:
    retarget_data['right'] = learned model outputs converted to DexMachina schema
    retarget_data['left']  = copied from existing .pt
    demo_data              = copied from existing .pt

Conventions verified against box_use_01_vector_para.pt:
- wrist_pose layout: [tx, ty, tz, qw, qx, qy, qz] (WXYZ, not XYZW).
- URDF wrist chain: T_root_base_dummy = Trans(tx,ty,tz) @ Rz(roll) @ Rx(pitch) @ Ry(-yaw).
- Inverse: roll=ZXY[0], pitch=ZXY[1], yaw=-ZXY[2] via intrinsic scipy euler.
- kpt_pos[T, 25, 3] in world frame.

Always uses the model's predicted wrist head (use_predicted_wrist=1).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dexmachina.retargeting.learned_retargeter_pkg import Retargeter  # noqa: E402

LOG = logging.getLogger("learned_retarget")

DEX_KPT_NAMES: tuple[str, ...] = (
    "link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip",
    "base_link",
    "link_0.0", "link_4.0", "link_8.0", "link_12.0",
    "link_1.0", "link_5.0", "link_9.0", "link_13.0",
    "link_2.0", "link_6.0", "link_10.0", "link_14.0",
    "link_3.0", "link_7.0", "link_11.0", "link_15.0",
    "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip",
)


def _reconstruct_training_style_obj_world(
    *,
    obj_canon_first_frame: np.ndarray,
    obj_T_world: np.ndarray,
) -> np.ndarray:
    """Per-frame world-frame object points matching the training pipeline.

    build_canonical_sequence_from_raw stores `object_bank_pts_obj := obj_points[0]`,
    and build_window_samples_from_sequence then computes
    `pts_world = obj_T_world[t] @ object_bank_pts_obj`. We replicate that here so
    the Retargeter wrapper sees inputs identical to those used at training.
    """
    T = int(obj_T_world.shape[0])
    rot = obj_T_world[:, :, :3, :3]
    trn = obj_T_world[:, :, :3, 3]
    obj_world = (
        np.einsum("toij,opj->topi", rot, obj_canon_first_frame.astype(np.float32))
        + trn[:, :, None, :]
    )
    return obj_world.reshape(T, -1, 3).astype(np.float32)


def _detect_frame_start(
    hand_wrist_T_w_raw: np.ndarray,
    existing_wrist_pose_first_frame: np.ndarray,
    tol_m: float = 0.05,
) -> int:
    target = existing_wrist_pose_first_frame[:3]
    dists = np.linalg.norm(hand_wrist_T_w_raw[:, :3, 3] - target[None, :], axis=1)
    fs = int(np.argmin(dists))
    if dists[fs] > tol_m:
        LOG.warning(
            "Best frame_start match has %.4fm error (> %.4fm tol); alignment may be off",
            float(dists[fs]), float(tol_m),
        )
    return fs


def _rot6d_to_matrix_batch(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:, :3]
    a2 = rot6d[:, 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    u2 = a2 - np.einsum("ti,ti->t", b1, a2)[:, None] * b1
    b2 = u2 / np.maximum(np.linalg.norm(u2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _matrix_to_wxyz_batch(R_mat: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    return np.asarray(quat_xyzw[..., [3, 0, 1, 2]], dtype=np.float32)


def _ik_wrist_dofs(T_world_base: np.ndarray) -> dict:
    """Decompose T_world_base[T, 4, 4] into the 6 URDF wrist DOFs.

    Chain (verified against box_use_01_vector_para.pt to 1e-6):
      T_root_base_dummy = Trans(tx,ty,tz) @ Rz(roll) @ Rx(pitch) @ Ry(-yaw)
    Inverse: roll=ZXY[0], pitch=ZXY[1], yaw=-ZXY[2] (intrinsic).
    """
    trans = T_world_base[:, :3, 3]
    euler = R.from_matrix(T_world_base[:, :3, :3]).as_euler("ZXY")
    return {
        "R_forearm_tx_link_joint": trans[:, 0].astype(np.float32),
        "R_forearm_ty_link_joint": trans[:, 1].astype(np.float32),
        "R_forearm_tz_link_joint": trans[:, 2].astype(np.float32),
        "R_forearm_roll_link_joint": euler[:, 0].astype(np.float32),
        "R_forearm_pitch_link_joint": euler[:, 1].astype(np.float32),
        "R_forearm_yaw_link_joint": (-euler[:, 2]).astype(np.float32),
    }


def _compute_kpt_pos_world(*, T_world_base, pred_q, hand_fk):
    """Per frame: FK on finger DOFs (in base_link frame), then world transform."""
    T = int(pred_q.shape[0])
    out = np.zeros((T, len(DEX_KPT_NAMES), 3), dtype=np.float32)
    for ti in range(T):
        tf_map = hand_fk.link_transforms_from_qpos(pred_q[ti], base_frame=True)
        rot_t = T_world_base[ti, :3, :3]
        t_t = T_world_base[ti, :3, 3]
        for ki, lname in enumerate(DEX_KPT_NAMES):
            tf_local = tf_map.get(lname)
            if tf_local is None:
                raise KeyError(
                    f"Link '{lname}' missing from learned_retargeter FK; known link count={len(tf_map)}."
                )
            link_pos_base = np.asarray(tf_local[:3, 3], dtype=np.float32)
            out[ti, ki] = (rot_t @ link_pos_base + t_t).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hand-config", default=None,
                        help="Override hand_config_path baked into checkpoint.")
    parser.add_argument("--arctic-raw", required=True,
                        help="ARCTIC raw .npz from learned_retargeter project.")
    parser.add_argument("--existing-pt", required=True,
                        help="Existing _vector_para.pt for frame range, left side, demo_data.")
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--predicted-wrist-frame", choices=["base", "origin"], default="base")
    parser.add_argument("--supervision-hand-index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    LOG.info("Loading Retargeter: %s", args.checkpoint)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ret = Retargeter.load(
        ckpt_path=args.checkpoint,
        hand_config_path=args.hand_config,
        device=dev,
        fps=float(args.fps),
    )
    LOG.info(
        "Model: joint_dim=%d window_size=%d object_points=%d device=%s",
        int(ret.joint_dim), int(ret.window_size), int(ret.object_points), dev,
    )

    LOG.info("Loading ARCTIC raw: %s", args.arctic_raw)
    raw = np.load(args.arctic_raw, allow_pickle=True)
    hand_kpts_world = np.asarray(raw["hand_kpts_world"], dtype=np.float32)
    hand_wrist_T_w = np.asarray(raw["hand_wrist_T_world"], dtype=np.float32)
    obj_T_world_full = np.asarray(raw["obj_T_world"], dtype=np.float32)
    obj_canon_full = np.asarray(raw["obj_points"], dtype=np.float32)
    T_raw = int(hand_kpts_world.shape[0])
    LOG.info(
        "Raw: T=%d, O=%d, P=%d/obj",
        T_raw, int(obj_canon_full.shape[1]), int(obj_canon_full.shape[2]),
    )

    LOG.info("Loading existing .pt: %s", args.existing_pt)
    existing = torch.load(args.existing_pt, weights_only=False, map_location="cpu")
    existing_right = existing["retarget_data"]["right"]
    existing_left = existing["retarget_data"]["left"]
    existing_wrist_pose = existing_right["wrist_pose"].numpy()
    T_pt = int(existing_wrist_pose.shape[0])
    LOG.info("Existing .pt: T=%d (left side will be copied verbatim)", T_pt)

    fs = _detect_frame_start(hand_wrist_T_w, existing_wrist_pose[0])
    fe = fs + T_pt
    if fe > T_raw:
        raise RuntimeError(f"frame_end={fe} exceeds raw T={T_raw}")
    LOG.info("Frame range: [%d, %d) (T=%d)", fs, fe, T_pt)

    mano_kpts = hand_kpts_world[fs:fe]
    wrist_world = hand_wrist_T_w[fs:fe]
    bank_pts = obj_canon_full[0]
    obj_world_input = _reconstruct_training_style_obj_world(
        obj_canon_first_frame=bank_pts,
        obj_T_world=obj_T_world_full[fs:fe],
    )
    LOG.info(
        "Inputs: mano_kpts=%s wrist_world=%s obj_world=%s",
        tuple(mano_kpts.shape), tuple(wrist_world.shape), tuple(obj_world_input.shape),
    )

    LOG.info("Running predict_sequence (predicted wrist frame=%s)...", args.predicted_wrist_frame)
    out = ret.predict_sequence(
        mano_kpts=mano_kpts,
        obj_points=obj_world_input,
        wrist_world=wrist_world,
    )
    pred_q = np.asarray(out["pred_q"], dtype=np.float32)
    pred_wrist_pos = np.asarray(out["pred_wrist_pos_world"], dtype=np.float32)
    pred_wrist_rot6d = np.asarray(out["pred_wrist_rot6d_world"], dtype=np.float32)
    LOG.info(
        "Predictions: pred_q=%s pred_wrist_pos=%s",
        tuple(pred_q.shape), tuple(pred_wrist_pos.shape),
    )

    delta_R = _rot6d_to_matrix_batch(pred_wrist_rot6d)
    R_wrist_w = wrist_world[:, :3, :3]
    t_wrist_w = wrist_world[:, :3, 3]
    if args.predicted_wrist_frame == "base":
        R_world_base = (R_wrist_w @ delta_R).astype(np.float32)
        t_world_base = (
            np.einsum("tij,tj->ti", R_wrist_w, pred_wrist_pos) + t_wrist_w
        ).astype(np.float32)
    else:
        raise NotImplementedError(
            "origin-frame prediction not implemented in Phase 1."
        )
    T_world_base = np.tile(np.eye(4, dtype=np.float32)[None, ...], (T_pt, 1, 1))
    T_world_base[:, :3, :3] = R_world_base
    T_world_base[:, :3, 3] = t_world_base

    wrist_dofs = _ik_wrist_dofs(T_world_base)

    LOG.info("Computing kpt_pos via learned FK (T=%d)...", T_pt)
    kpt_pos = _compute_kpt_pos_world(
        T_world_base=T_world_base,
        pred_q=pred_q,
        hand_fk=ret._geometry.hand_fk,
    )

    joint_qpos: dict = {**wrist_dofs}
    for i in range(int(pred_q.shape[1])):
        joint_qpos[f"joint_{i}.0"] = pred_q[:, i].astype(np.float32)

    quat_wxyz = _matrix_to_wxyz_batch(R_world_base)
    wrist_pose = np.concatenate([t_world_base, quat_wxyz], axis=-1).astype(np.float32)

    right_out = {
        "joint_qpos": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in joint_qpos.items()},
        "joint_targets": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in joint_qpos.items()},
        "kpt_pos": torch.as_tensor(kpt_pos, dtype=torch.float32),
        "kpt_names": list(DEX_KPT_NAMES),
        "wrist_pose": torch.as_tensor(wrist_pose, dtype=torch.float32),
        "wrist_link_name": "base_dummy_link",
    }

    output = {
        "retarget_data": {"right": right_out, "left": existing_left},
        "demo_data": existing["demo_data"],
    }
    if "retargeter_results" in existing:
        output["retargeter_results"] = existing["retargeter_results"]

    out_path = Path(args.out_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(out_path))
    LOG.info(
        "Wrote %s  |  right=learned (T=%d, 22 joints, 25 kpts)  left=copied from existing",
        str(out_path), T_pt,
    )


if __name__ == "__main__":
    main()
