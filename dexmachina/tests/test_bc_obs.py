"""Unit tests for the offline BC observation builder (dexmachina/rl/bc_dataset.py).

The parity test compares against dexmachina/tests/fixtures/e76b2_at3_env_obs_60f.npz,
teacher-forced observations recorded from a LIVE Genesis env (hands + objects
hard-set to the reference each frame, no physics stepping). Regenerate with
rl/verify_bc_obs.py if the obs layout ever changes. Velocity dims are excluded:
set-state leaves sim velocities at zero while the offline builder uses reference
finite differences (the documented teacher-forcing approximation).

Run from the repo root:
    python -m pytest dexmachina/tests/test_bc_obs.py -q
"""

import os

import numpy as np
import pytest
import torch

from dexmachina.rl.bc_dataset import (
    NDOF,
    ClipData,
    HandRef,
    ObjectRef,
    build_bc_arrays,
    build_clip_observations,
    clip_action_labels,
    expected_obs_dim,
    load_clip,
    split_clips,
)

SCALES = (0.1, 1.0)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIP_PT = os.path.join(
    REPO, "dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt"
)
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/e76b2_at3_env_obs_60f.npz")


def synthetic_clip(T=30, n_objects=2, n_kpts=25):
    hands = {}
    for side in ("left", "right"):
        ref = torch.zeros((T, NDOF))
        t = torch.arange(T, dtype=torch.float32)
        ref[:, :6] = 0.002 * t[:, None]
        ref[:, 6:] = 0.1 + 0.01 * t[:, None]
        limits = torch.zeros((NDOF, 2))
        limits[:, 0], limits[:, 1] = -2.0, 2.0
        hands[side] = HandRef(
            ref_qpos=ref,
            dof_limits=limits,
            kpt_pos=torch.randn(T, n_kpts, 3) * 0.01,
            wrist_pose=torch.randn(T, 7) * 0.01,
        )
    objects = []
    for i in range(n_objects):
        angle = 0.01 * torch.arange(T, dtype=torch.float32)  # slow z-rotation
        quat = torch.stack(
            [torch.cos(angle / 2), torch.zeros(T), torch.zeros(T), torch.sin(angle / 2)], dim=1
        )
        pos = torch.zeros((T, 3))
        pos[:, 0] = 0.005 * torch.arange(T, dtype=torch.float32)
        objects.append(ObjectRef(name=f"obj{i}", pos=pos, quat=quat, arti=torch.zeros((T, 1))))
    return ClipData(path="synthetic", num_frames=T, hands=hands, objects=objects)


def build(clip, horizon=1):
    labels, _ = clip_action_labels(clip, SCALES, horizon)
    return labels, build_clip_observations(clip, labels, SCALES, horizon)


def test_obs_dims_two_object_and_shared():
    for n_obj, expected in ((2, 341), (1, 319)):
        clip = synthetic_clip(n_objects=n_obj)
        _, obs = build(clip)
        assert obs.shape == (clip.num_frames - 1, expected)
        assert expected_obs_dim(clip) == expected


def test_dof_target_pos_zero_for_horizon_one():
    # With h=1 the teacher target commanded at t-1 IS ref[t] (when unclipped),
    # so the dof_target_pos block must vanish, matching the env's reset state.
    clip = synthetic_clip()
    _, obs = build(clip, horizon=1)
    for start in (0, 148):  # left, right robot block offsets
        assert obs[:, start:start + NDOF].abs().max() < 1e-6


def test_dof_vel_finite_difference_and_reset_zero():
    clip = synthetic_clip()
    _, obs = build(clip, horizon=1)
    vel = obs[:, 44:66]  # left dof_vel block, scaled by 0.1
    assert vel[0].abs().max() == 0.0
    assert torch.allclose(vel[1:, :6], torch.full_like(vel[1:, :6], 0.002 * 60 * 0.1), atol=1e-5)


def test_object_velocities_and_state_diff():
    clip = synthetic_clip(n_objects=1)
    _, obs = build(clip, horizon=1)
    obj = obs[:, 296:318]
    lin_vel = obj[:, 19:22]
    assert lin_vel[0].abs().max() == 0.0
    assert torch.allclose(lin_vel[1:, 0], torch.full_like(lin_vel[1:, 0], 0.005 * 60 * 2.0), atol=1e-4)
    ang_vel = obj[:, 16:19]
    assert torch.allclose(ang_vel[1:, 2], torch.full_like(ang_vel[1:, 2], 0.01 * 60 * 0.25), atol=1e-3)
    # state_diff pos rows = one-frame demo delta (except at the clamped tail)
    state_diff = obj[:, 8:16]
    assert torch.allclose(state_diff[:-1, 0], torch.full_like(state_diff[:-1, 0], 0.005), atol=1e-5)


def test_phase_feature_range():
    clip = synthetic_clip(T=40)
    _, obs = build(clip, horizon=1)
    phase = obs[:, -1]
    assert phase[0] == -1.0
    assert torch.isclose(phase[-1], torch.tensor(2.0 * 38 / 40 - 1.0))


def test_split_clips_is_trajectory_level_and_deterministic():
    paths = [f"clip{i}" for i in range(5)]
    train, val = split_clips(paths, num_val_clips=1, seed=0)
    train2, val2 = split_clips(paths, num_val_clips=1, seed=0)
    assert (train, val) == (train2, val2)
    assert sorted(train + val) == sorted(paths)
    assert not set(train) & set(val)
    with pytest.raises(ValueError):
        split_clips(paths, num_val_clips=5, seed=0)


@pytest.mark.skipif(not os.path.exists(CLIP_PT), reason="e76b2_at3 clip not in assets")
def test_parity_with_live_env_fixture():
    clip = load_clip(CLIP_PT)
    labels, _ = clip_action_labels(clip, SCALES, horizon=1)
    obs = build_clip_observations(clip, labels, SCALES, horizon=1).numpy()
    sim = np.load(FIXTURE)["obs"]
    n = sim.shape[0]
    assert obs.shape[1] == sim.shape[1] == 341

    velocity_dims = np.zeros(341, dtype=bool)
    for start in (44, 192):          # robot dof_vel blocks
        velocity_dims[start:start + 22] = True
    for start in (312, 334):         # object ang_vel + lin_vel blocks
        velocity_dims[start:start + 6] = True

    diff = np.abs(obs[:n] - sim)
    assert diff[:, ~velocity_dims].max() < 1e-5, "non-velocity obs dims must match the live env"
    # Velocity dims: sim is zero under set-state; offline is the finite-difference
    # teacher value. Bound the (scaled) magnitudes to catch unit errors (x60 etc.).
    assert diff[:, velocity_dims].max() < 2.0


@pytest.mark.skipif(not os.path.exists(CLIP_PT), reason="e76b2_at3 clip not in assets")
def test_build_bc_arrays_real_clip():
    arrays = build_bc_arrays([CLIP_PT], SCALES, horizon=1)
    assert arrays.observations.shape == (768, 341)
    assert arrays.actions.shape == (768, 44)
    assert arrays.actions.abs().max() <= 1.0
    assert arrays.diagnostics[0].clipped_fraction == 0.0
