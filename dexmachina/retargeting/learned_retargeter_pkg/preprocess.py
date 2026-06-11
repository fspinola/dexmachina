"""MANO + object featurization for inference-only retargeting.

Mirrors the per-frame preprocessing in
``learned_retargeter.kinematic.data.build_window_samples_from_sequence`` but
single-frame, single-hand, and without the canonical-sequence / window-extraction
machinery that the training shards need.
"""

from __future__ import annotations

import numpy as np

from ._frames import rotate_to_local, to_local

HAND_TOKENS = 42  # 2 hands x 21 keypoints (model layout is fixed)
HAND_FEAT_DIM = 12  # pos3 + rot6d6 + vel3


def estimate_outward_normals(points_world: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate per-point outward unit normals via local PCA.

    Direct port of the training-time helper so inference produces normals in the
    same convention. See
    ``learned_retargeter.kinematic.data._estimate_outward_normals_from_points``.
    """

    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points [P, 3], got {pts.shape}.")
    p = int(pts.shape[0])
    if p == 0:
        return np.zeros((0, 3), dtype=np.float32)
    k_eff = int(min(max(3, int(k)), p))
    centroid = pts.mean(axis=0)

    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    nn_idx = np.argpartition(d2, kth=k_eff - 1, axis=-1)[:, :k_eff]
    nn_pts = pts[nn_idx]
    nn_pts = nn_pts - nn_pts.mean(axis=1, keepdims=True)
    covs = np.einsum("pki,pkj->pij", nn_pts, nn_pts)
    try:
        _, eigvecs = np.linalg.eigh(covs)
    except np.linalg.LinAlgError:
        return np.zeros((p, 3), dtype=np.float32)
    normals = eigvecs[..., 0]

    outward = pts - centroid
    sign = np.where(np.einsum("pi,pi->p", normals, outward) >= 0.0, 1.0, -1.0)
    normals = normals * sign[:, None]
    mags = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(mags > 1.0e-9, normals / np.maximum(mags, 1.0e-9), 0.0)
    return normals.astype(np.float32)


def sample_object_points(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample ``k`` (point, normal) pairs from a world-frame cloud.

    Matches the resampling strategy used by ``UniformObjectPointSampler`` at
    training time: random indices, with replacement when ``N < k``.
    """

    pts = np.asarray(points_world, dtype=np.float32)
    nrm = np.asarray(normals_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points [P, 3], got {pts.shape}.")
    if nrm.shape != pts.shape:
        raise ValueError(f"normals shape {nrm.shape} must match points {pts.shape}.")
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((int(k), 3), dtype=np.float32), np.zeros((int(k), 3), dtype=np.float32)
    gen = np.random.default_rng(0) if rng is None else rng
    idx = gen.integers(low=0, high=n, size=int(k)) if n < int(k) else gen.choice(n, size=int(k), replace=False)
    return pts[idx].astype(np.float32), nrm[idx].astype(np.float32)


def pack_frame_features(
    *,
    mano_kpts_world: np.ndarray,
    wrist_world: np.ndarray,
    obj_points_world: np.ndarray,
    obj_normals_world: np.ndarray,
    prev_kpts_world: np.ndarray | None,
    fps: float,
    hand_rot6d: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build one frame of model-ready features in supervision-wrist frame.

    Args:
        mano_kpts_world: ``[21, 3]`` MANO keypoints in world.
        wrist_world: ``[4, 4]`` supervision-hand wrist transform (world).
        obj_points_world: ``[K, 3]`` object points (already sampled to K).
        obj_normals_world: ``[K, 3]`` matching outward normals.
        prev_kpts_world: ``[21, 3]`` previous-frame world keypoints for finite-
            difference velocity, or ``None`` on the first frame (yields zero
            velocity, matching the training-side convention).
        fps: sequence frame rate, used to scale velocity.
        hand_rot6d: optional ``[21, 6]`` per-keypoint rot6d. Zeros when absent
            (mirrors ``build_canonical_sequence_from_raw``'s default).

    Returns:
        Dict with ``hand_feats [42, 12]``, ``hand_mask [42]``, ``obj_feats [K, 6]``,
        ``obj_mask [K]`` — all single-frame, single-hand (second hand zeroed).
    """

    kpts = np.asarray(mano_kpts_world, dtype=np.float32)
    if kpts.shape != (21, 3):
        raise ValueError(f"Expected mano_kpts_world [21, 3], got {kpts.shape}.")
    wrist = np.asarray(wrist_world, dtype=np.float32)
    if wrist.shape != (4, 4):
        raise ValueError(f"Expected wrist_world [4, 4], got {wrist.shape}.")

    rot6d = (
        np.zeros((21, 6), dtype=np.float32)
        if hand_rot6d is None
        else np.asarray(hand_rot6d, dtype=np.float32)
    )
    if rot6d.shape != (21, 6):
        raise ValueError(f"Expected hand_rot6d [21, 6], got {rot6d.shape}.")

    if prev_kpts_world is None:
        vel_world = np.zeros_like(kpts, dtype=np.float32)
    else:
        dt = 1.0 / max(float(fps), 1e-6)
        vel_world = ((kpts - np.asarray(prev_kpts_world, dtype=np.float32)) / dt).astype(np.float32)

    pos_wrist = to_local(wrist, kpts).astype(np.float32)
    vel_wrist = rotate_to_local(wrist, vel_world).astype(np.float32)

    hand_feats = np.zeros((HAND_TOKENS, HAND_FEAT_DIM), dtype=np.float32)
    hand_mask = np.zeros((HAND_TOKENS,), dtype=bool)
    hand_feats[:21] = np.concatenate([pos_wrist, rot6d, vel_wrist], axis=-1)
    hand_mask[:21] = True
    # Tokens 21:42 stay zeroed for the absent second hand (mask False -> the
    # model multiplies their features by zero in _fuse_features anyway).

    pts = np.asarray(obj_points_world, dtype=np.float32)
    nrm = np.asarray(obj_normals_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or nrm.shape != pts.shape:
        raise ValueError(
            f"Expected matching obj_points [K, 3] and obj_normals [K, 3], "
            f"got {pts.shape} / {nrm.shape}."
        )
    pts_wrist = to_local(wrist, pts).astype(np.float32)
    nrm_wrist = rotate_to_local(wrist, nrm).astype(np.float32)
    obj_feats = np.concatenate([pts_wrist, nrm_wrist], axis=-1).astype(np.float32)
    obj_mask = np.ones((int(pts.shape[0]),), dtype=bool)

    return {
        "hand_feats": hand_feats,
        "hand_mask": hand_mask,
        "obj_feats": obj_feats,
        "obj_mask": obj_mask,
    }
