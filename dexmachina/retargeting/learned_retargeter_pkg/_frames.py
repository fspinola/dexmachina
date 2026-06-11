"""Frame transform helpers for stage-1 baseline data/model code."""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch_tensor(x: Any) -> bool:
    """Return whether ``x`` behaves like a torch Tensor without importing torch."""

    return hasattr(x, "detach") and hasattr(x, "dim") and hasattr(x, "dtype")


def _check_shapes(transform_world_frame: Any, points: Any) -> None:
    """Validate transform and point trailing dimensions."""

    t_shape = tuple(transform_world_frame.shape)
    p_shape = tuple(points.shape)
    if len(t_shape) < 2 or t_shape[-2:] != (4, 4):
        raise ValueError(f"Expected transform_world_frame [...,4,4], got {t_shape}.")
    if len(p_shape) < 1 or p_shape[-1] != 3:
        raise ValueError(f"Expected points [...,3], got {p_shape}.")


def _einsum_matvec(rot: Any, vec: Any) -> Any:
    """Compute batched matrix-vector product for numpy/torch tensors."""

    if _is_torch_tensor(rot):
        return rot.new_tensor(0.0) + rot.matmul(vec.unsqueeze(-1)).squeeze(-1)
    return np.einsum("...ij,...j->...i", np.asarray(rot), np.asarray(vec))


def to_world(transform_world_frame: Any, points_frame: Any) -> Any:
    """Transform point coordinates from local frame to world frame.

    Args:
        transform_world_frame: ``[...,4,4]`` transform from local frame to world.
        points_frame: ``[...,3]`` points in local frame.

    Returns:
        World-frame points with shape ``[...,3]``.
    """

    _check_shapes(transform_world_frame, points_frame)
    rot = transform_world_frame[..., :3, :3]
    trn = transform_world_frame[..., :3, 3]
    return _einsum_matvec(rot, points_frame) + trn


def to_local(transform_world_frame: Any, points_world: Any) -> Any:
    """Transform point coordinates from world frame to local frame.

    Args:
        transform_world_frame: ``[...,4,4]`` transform from local frame to world.
        points_world: ``[...,3]`` points in world frame.

    Returns:
        Local-frame points with shape ``[...,3]``.
    """

    _check_shapes(transform_world_frame, points_world)
    rot = transform_world_frame[..., :3, :3]
    trn = transform_world_frame[..., :3, 3]
    centered = points_world - trn
    if _is_torch_tensor(rot):
        rot_t = rot.transpose(-1, -2)
    else:
        rot_t = np.swapaxes(np.asarray(rot), -1, -2)
    return _einsum_matvec(rot_t, centered)


def rotate_to_local(transform_world_frame: Any, vectors_world: Any) -> Any:
    """Rotate vectors from world frame to local frame (ignores translation).

    Args:
        transform_world_frame: ``[...,4,4]`` transform from local frame to world.
        vectors_world: ``[...,3]`` vectors in world frame.

    Returns:
        Local-frame vectors with shape ``[...,3]``.
    """

    _check_shapes(transform_world_frame, vectors_world)
    rot = transform_world_frame[..., :3, :3]
    if _is_torch_tensor(rot):
        rot_t = rot.transpose(-1, -2)
    else:
        rot_t = np.swapaxes(np.asarray(rot), -1, -2)
    return _einsum_matvec(rot_t, vectors_world)
