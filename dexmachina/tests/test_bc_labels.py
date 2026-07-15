"""Unit tests for BC action labels (dexmachina/rl/bc_dataset.py).

Sim-free: synthetic reference trajectories only. Run from the repo root:
    python -m pytest dexmachina/tests/test_bc_labels.py -q
"""

import numpy as np
import pytest
import torch

from dexmachina.rl.bc_dataset import (
    FINGER_SLICE,
    NDOF,
    WRIST_SLICE,
    ClipData,
    HandRef,
    ObjectRef,
    allegro_dof_names,
    clip_action_labels,
    compose_hybrid_targets,
    hybrid_action_labels,
)

SCALES = (0.1, 1.0)


def linear_ref(T=20, wrist_delta=0.002, finger_lo=0.1, finger_hi=0.4):
    """Reference with constant per-frame wrist delta and linear finger sweep."""
    ref = torch.zeros((T, NDOF))
    t = torch.arange(T, dtype=torch.float32)
    ref[:, WRIST_SLICE] = wrist_delta * t[:, None]
    ref[:, FINGER_SLICE] = finger_lo + (finger_hi - finger_lo) * (t / (T - 1))[:, None]
    limits = torch.zeros((NDOF, 2))
    limits[:, 0], limits[:, 1] = -2.0, 2.0
    return ref, limits


def test_linear_motion_exact_labels():
    T, d = 20, 0.002
    ref, limits = linear_ref(T=T, wrist_delta=d)
    raw, clipped = hybrid_action_labels(ref, limits, SCALES, horizon=1)
    assert raw.shape == (T - 1, NDOF)
    # wrist: delta formula with per-group scales
    expected_trans = d / SCALES[0]
    expected_rot = d / SCALES[1]
    assert torch.allclose(raw[:, 0:3], torch.full((T - 1, 3), expected_trans), atol=1e-6)
    assert torch.allclose(raw[:, 3:6], torch.full((T - 1, 3), expected_rot), atol=1e-6)
    # fingers: absolute limit-normalized NEXT frame
    lo, hi = limits[FINGER_SLICE, 0], limits[FINGER_SLICE, 1]
    expected_fingers = 2.0 * (ref[1:, FINGER_SLICE] - lo) / (hi - lo) - 1.0
    assert torch.allclose(raw[:, FINGER_SLICE], expected_fingers, atol=1e-6)
    assert torch.equal(clipped, raw)  # nothing near the bounds here


def test_horizon_zero_is_kinematic_teacher():
    ref, limits = linear_ref()
    raw, _ = hybrid_action_labels(ref, limits, SCALES, horizon=0)
    assert raw.shape[0] == ref.shape[0]
    assert torch.allclose(raw[:, WRIST_SLICE], torch.zeros_like(raw[:, WRIST_SLICE]))
    lo, hi = limits[FINGER_SLICE, 0], limits[FINGER_SLICE, 1]
    assert torch.allclose(raw[:, FINGER_SLICE], 2.0 * (ref[:, FINGER_SLICE] - lo) / (hi - lo) - 1.0)


def test_horizon_two_scales_delta():
    d = 0.003
    ref, limits = linear_ref(T=15, wrist_delta=d)
    raw, _ = hybrid_action_labels(ref, limits, SCALES, horizon=2)
    assert raw.shape[0] == 13
    assert torch.allclose(raw[:, 0:3], torch.full((13, 3), 2 * d / SCALES[0]), atol=1e-6)


def test_clipping_detected_and_bounded():
    ref, limits = linear_ref(T=10, wrist_delta=0.5)  # 0.5 m/frame >> 0.1 scale
    raw, clipped = hybrid_action_labels(ref, limits, SCALES, horizon=1)
    assert (raw[:, 0:3] > 1.0).all()
    assert torch.equal(clipped[:, 0:3], torch.ones_like(clipped[:, 0:3]))
    assert clipped.abs().max() <= 1.0


def test_short_clip_raises():
    ref, limits = linear_ref(T=3)
    with pytest.raises(ValueError, match="clip too short"):
        hybrid_action_labels(ref, limits, SCALES, horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        hybrid_action_labels(ref, limits, SCALES, horizon=-1)


def test_compose_round_trip():
    ref, limits = linear_ref(T=30, wrist_delta=0.004)
    h = 1
    _, labels = hybrid_action_labels(ref, limits, SCALES, horizon=h)
    targets = compose_hybrid_targets(labels, ref[:-h], limits, SCALES)
    assert torch.allclose(targets, ref[h:], atol=1e-5)


def _synthetic_clip(T=25):
    hands = {}
    for side in ("left", "right"):
        ref, limits = linear_ref(T=T, wrist_delta=0.002 if side == "left" else 0.001)
        hands[side] = HandRef(
            ref_qpos=ref,
            dof_limits=limits,
            kpt_pos=torch.zeros((T, 25, 3)),
            wrist_pose=torch.zeros((T, 7)),
        )
    quat = torch.zeros((T, 4))
    quat[:, 0] = 1.0
    objects = [ObjectRef(name="obj", pos=torch.zeros((T, 3)), quat=quat, arti=torch.zeros((T, 1)))]
    return ClipData(path="synthetic", num_frames=T, hands=hands, objects=objects)


def test_clip_action_labels_full_env_vector():
    clip = _synthetic_clip(T=25)
    labels, diag = clip_action_labels(clip, SCALES, horizon=1)
    assert labels.shape == (24, 2 * NDOF)
    # left wrist trans delta = 0.002 -> 0.02; right = 0.001 -> 0.01
    assert torch.allclose(labels[:, 0:3], torch.full((24, 3), 0.02), atol=1e-6)
    assert torch.allclose(labels[:, NDOF:NDOF + 3], torch.full((24, 3), 0.01), atol=1e-6)
    assert diag.num_samples == 24
    assert diag.clipped_fraction == 0.0
    assert diag.recon_err_max < 1e-5
    assert diag.label_mean.shape == (2 * NDOF,)


def test_dof_name_order_is_bfs_interleaved():
    names = allegro_dof_names("right")
    assert names[:6] == [f"R_forearm_{ax}_link_joint" for ax in ("tx", "ty", "tz", "roll", "pitch", "yaw")]
    assert names[6:10] == ["joint_0.0", "joint_4.0", "joint_8.0", "joint_12.0"]
    assert names[18:] == ["joint_3.0", "joint_7.0", "joint_11.0", "joint_15.0"]
    assert len(names) == NDOF and len(set(names)) == NDOF
