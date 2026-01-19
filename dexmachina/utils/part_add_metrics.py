"""Part-wise ADD-AUC metric (DexMachina paper-style).

The paper describes an ADD-AUC metric (similar to https://arxiv.org/pdf/1711.00199) where ADD is computed for each articulated
object part separately, then averaged across parts, and finally an AUC (up to
some distance threshold) is computed.

This module provides:
- A runtime helper to compute per-step part ADD by comparing the current object
  state to a demo state. (Requires access to the live `ArticulatedObject`.)
- An offline helper to aggregate saved per-step part ADD trajectories into an
    AUC summary (threshold configurable).

Important:
- The repo's `ArticulatedObject` already exposes `part_pos` and `part_quat`
  buffers (via Genesis `entity.get_links_pos/quats`).
- The object config (for ARCTIC assets) provides separate watertight meshes for
  `top` and `bottom` parts (see `get_arctic_object_cfg`).

Implementation choices / assumptions
----------------------------------
1) "Parts" are assumed to be named `top` and `bottom` (as in ARCTIC objects).
2) Points for each part are sampled from the corresponding mesh vertices
   (`ArticulatedObject.sample_mesh_vertices`).
3) Points are assumed to be expressed in the part frame used by Genesis link
   pose. This is consistent with how the assets are packaged in DexMachina.
   If your meshes are in a different frame, you may need a fixed transform.

Contract
--------
Runtime per-step:
- Input: `obj` (ArticulatedObject), `demo_state` (tensor/ndarray with
  [root_pos(3), root_quat(4), dof(1 or more)])
- Output: dict with per-part ADD (meters) and mean across parts.

Offline:
- Input: per-step `part_add_mean` array
- Output: AUC (normalized to [0,1]) up to `cfg.auc_threshold_m`

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from dexmachina.envs.math_utils import matrix_from_quat


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for wxyz quats."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply (wxyz)."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternion q (wxyz).

    q: (..., 4)
    v: (..., 3)
    """
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]


@dataclass(frozen=True)
class PartAddConfig:
    # names match ARCTIC packaging and `ArticulatedObject.link_names`
    part_names: tuple[str, ...] = ("bottom", "top")
    # number of sampled vertices per part
    num_points_per_part: int = 300
    # Percentage of object diameter threshold
    auc_threshold_pct: float = 0.1
    # AUC threshold in meters
    auc_threshold_m: float = 0.03
    # AUC integration resolution in meters
    auc_step_m: float = 0.001
    # deterministic sampling seed
    seed: int = 42


def _ensure_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _transform_points(points_local: torch.Tensor, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) to local points.

    Args:
        points_local: (P, 3)
        pos: (B, 3)
        quat_wxyz: (B, 4)

    Returns:
        points_world: (B, P, 3)
    """
    R = matrix_from_quat(quat_wxyz)  # (B, 3, 3)
    # (B, P, 3) = (B, 3, 3) @ (P, 3)^T
    pts = torch.einsum("bij,pj->bpi", R, points_local)
    return pts + pos[:, None, :]


def _sanitize_quat_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternions and replace non-finite values defensively."""
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / n


def _demo_part_poses_from_demo_state(
    *,
    obj,
    demo_state: torch.Tensor,
    env_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-link SE(3) for the object at `demo_state` without mutating the simulator.

    This avoids the major pitfall of calling `obj.set_object_state()` inside the metric.

    Returns:
        demo_part_pos: (L, 3)
        demo_part_quat: (L, 4) in wxyz
    """

    demo_state = demo_state.to(device=obj.root_pos.device, dtype=torch.float32)
    demo_root_pos = demo_state[:3]
    demo_root_quat = demo_state[3:7]
    demo_qpos = demo_state[7:]

    # Base link poses (from current sim buffers)
    base_part_pos = obj.part_pos[env_idx]  # (L,3)
    base_part_quat = obj.part_quat[env_idx]  # (L,4)
    base_root_pos = obj.root_pos[env_idx]
    base_root_quat = obj.root_quat[env_idx]
    base_qpos = obj.dof_pos[env_idx] if hasattr(obj, "dof_pos") else None

    # Relative transform from base root -> demo root
    q_delta = _quat_mul(demo_root_quat, _quat_conjugate(base_root_quat))
    t_delta = demo_root_pos - _quat_apply(q_delta, base_root_pos)

    demo_part_pos = _quat_apply(q_delta[None, :], base_part_pos) + t_delta[None, :]
    demo_part_quat = _quat_mul(q_delta[None, :], base_part_quat)

    # If we know how to update the articulated link, we correct that one too.
    # (This is important for ARCTIC-style objects where "top" depends on dof_pos.)
    if base_qpos is not None and demo_qpos.numel() == 1 and base_qpos.numel() == 1:
        if "top" in obj.link_names and "bottom" in obj.link_names:
            top_idx = obj.link_names.index("top")

            # relative rotation for the revolute joint about local +Z (assumption for ARCTIC assets)
            dtheta = (demo_qpos[0] - base_qpos[0]).item()
            half = 0.5 * float(dtheta)
            dq_local = torch.tensor([np.cos(half), 0.0, 0.0, np.sin(half)], device=obj.root_pos.device)

            # apply in the bottom frame: q_top_demo = q_bottom_demo * dq_local
            bottom_idx = obj.link_names.index("bottom")
            demo_part_quat[top_idx] = _quat_mul(demo_part_quat[bottom_idx], dq_local)
            # position offset should remain the same for a pure revolute around link origin.
            # If the hinge origin is offset, we'd need the joint anchor; we don't have it here.

    return demo_part_pos, demo_part_quat


def get_part_points_local(obj, cfg: PartAddConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Sample (deterministic) vertices per part from the meshes referenced by object cfg."""
    pts: Dict[str, torch.Tensor] = {}
    for i, part in enumerate(cfg.part_names):
        # different seed per part to avoid identical subsamples
        part_seed = int(cfg.seed + 17 * i)
        v = obj.sample_mesh_vertices(cfg.num_points_per_part, part=part, seed=part_seed)
        pts[part] = v.to(device=device, dtype=torch.float32)
    return pts


@torch.no_grad()
def compute_part_add_step(
    *,
    obj,
    demo_state: torch.Tensor,
    part_points_local: Dict[str, torch.Tensor],
    cfg: PartAddConfig,
    env_idx: int = 0,
    treat_as_rigid: bool = False,
    demo_part_pos: torch.Tensor | None = None,
    demo_part_quat: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute part-wise ADD for the current simulator state vs a demo state.

    This does:
    1) Save current object state.
    2) Read current part poses (policy).
    3) Temporarily set object to demo_state.
    4) Read part poses (demo).
    5) Restore policy state.
    6) Compute ADD per part as mean point distance.

    This is slow (two extra set_state calls per step) but correct and simple.
    For performance, you can precompute demo part poses offline.
    """

    device = obj.root_pos.device

    # Policy part poses (from sim, no mutation)
    pol_part_pos = obj.part_pos[env_idx].clone()  # (L,3)
    pol_part_quat = _sanitize_quat_wxyz(obj.part_quat[env_idx].clone())  # (L,4)

    # Demo part poses:
    # - Preferred: pass cached demo_part_pos/demo_part_quat computed from the simulator
    #   (exact FK, no per-step sim mutation).
    # - Fallback: approximate from demo_state without mutating the simulator.
    if demo_part_pos is None or demo_part_quat is None:
        demo_part_pos, demo_part_quat = _demo_part_poses_from_demo_state(
            obj=obj,
            demo_state=demo_state,
            env_idx=env_idx,
        )
    else:
        # Defensive: normalize even when poses are precomputed.
        demo_part_quat = _sanitize_quat_wxyz(demo_part_quat)
    if treat_as_rigid:
        # Recompute poses ignoring joint DOF contribution.
        demo_state = demo_state.to(device=obj.root_pos.device, dtype=torch.float32)
        demo_root_pos = demo_state[:3]
        demo_root_quat = demo_state[3:7]
        base_part_pos = obj.part_pos[env_idx]
        base_part_quat = obj.part_quat[env_idx]
        base_root_pos = obj.root_pos[env_idx]
        base_root_quat = obj.root_quat[env_idx]

        q_delta = _quat_mul(demo_root_quat, _quat_conjugate(base_root_quat))
        t_delta = demo_root_pos - _quat_apply(q_delta, base_root_pos)
        demo_part_pos = _quat_apply(q_delta[None, :], base_part_pos) + t_delta[None, :]
        demo_part_quat = _quat_mul(q_delta[None, :], base_part_quat)

    out: Dict[str, float] = {}
    adds: list[float] = []

    for part in cfg.part_names:
        if part not in obj.link_names:
            # Keep the output schema stable (caller may expect keys), but mark missing parts.
            out[f"part_add_{part}"] = float("nan")
            adds.append(float("nan"))
            continue
        link_idx = obj.link_names.index(part)
        pts_local = part_points_local[part]

        pol_pts = _transform_points(pts_local, pol_part_pos[link_idx][None], pol_part_quat[link_idx][None])
        demo_pts = _transform_points(pts_local, demo_part_pos[link_idx][None], demo_part_quat[link_idx][None])

        # If something non-finite still slips through (e.g., internal numerical issue), mark as failure.
        if not torch.isfinite(pol_pts).all() or not torch.isfinite(demo_pts).all():
            out[f"part_add_{part}"] = float("nan")
            adds.append(float("nan"))
            continue

        # mean euclidean distance
        add_val = torch.norm(pol_pts - demo_pts, dim=-1).mean().item()
        out[f"part_add_{part}"] = float(add_val)
        adds.append(float(add_val))

    if len(adds) == 0:
        out["part_add_mean"] = float("nan")
    else:
        adds_arr = np.asarray(adds, dtype=np.float64)
        finite = np.isfinite(adds_arr)
        # Avoid RuntimeWarning: Mean of empty slice (e.g., when all parts are missing/non-finite).
        out["part_add_mean"] = float(np.mean(adds_arr[finite])) if finite.any() else float("nan")
    out["num_parts"] = int(len(adds))
    return out


def auc_under_threshold(errors: np.ndarray, *, threshold: float, step: float) -> float:
    """Compute normalized AUC for error CDF up to `threshold`.

    Standard ADD-AUC computation:
    - For thresholds t in [0, threshold], compute success(t)=mean(errors <= t)
    - Integrate success(t) dt and normalize by threshold.

    Returns a value in [0,1].
    """

    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return float("nan")

    ts = np.arange(0.0, threshold + 1e-12, step, dtype=np.float64)
    # success curve
    succ = (errors[None, :] <= ts[:, None]).mean(axis=1)
    # trapezoidal integration
    auc = float(np.trapz(succ, ts) / threshold)
    return auc


def add_auc_from_episode_part_add_mean(part_add_mean: np.ndarray, *, cfg: PartAddConfig) -> Dict[str, Any]:
    """Compute ADD-AUC from per-step averaged part ADD."""
    auc = auc_under_threshold(part_add_mean, threshold=cfg.auc_threshold_m, step=cfg.auc_step_m)
    return {
        "add_auc": float(auc),
        "threshold_m": float(cfg.auc_threshold_m),
        "auc_threshold_pct": float(cfg.auc_threshold_pct),
        "step_m": float(cfg.auc_step_m),
        "num_frames": int(np.isfinite(np.asarray(part_add_mean)).sum()),
    }


def summarize_add_auc(episodes: Iterable[Mapping[str, Any]], *, cfg: PartAddConfig) -> Dict[str, Any]:
    """Aggregate ADD-AUC across episodes.

    Each episode mapping must contain `part_add_mean` as a 1D array.
    """

    per_ep = []
    for ep in episodes:
        per_ep.append(add_auc_from_episode_part_add_mean(np.asarray(ep["part_add_mean"]), cfg=cfg))

    vals = np.array([d["add_auc"] for d in per_ep], dtype=np.float64)
    return {
        "add_auc_mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        "add_auc_std": float(np.nanstd(vals)) if vals.size else float("nan"),
        "num_episodes": int(len(per_ep)),
        "per_episode": per_ep,
    }

