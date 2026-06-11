"""Thin robot-geometry wrapper for stage-1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._hand_fk import HandKinematics
from ._hand_profile import HandProfile, load_hand_profile


@dataclass
class Stage1RobotGeometry:
    """Batched FK + joint-limit helper built on top of :class:`HandKinematics`."""

    hand_profile: HandProfile
    hand_fk: HandKinematics
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    rest_pose: np.ndarray
    fingertip_names: tuple[str, ...]

    @staticmethod
    def from_hand_profile(hand_profile: HandProfile) -> "Stage1RobotGeometry":
        """Build stage-1 geometry wrapper from a validated hand profile."""

        fk = HandKinematics(hand_profile)
        lower, upper = fk.joint_limits_lower_upper(fill_lower=-np.pi, fill_upper=np.pi)
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)

        # Default weak rest pose: midpoint of finite limits, zero when bounds are open.
        finite = np.isfinite(lower) & np.isfinite(upper)
        mid = np.zeros_like(lower, dtype=np.float32)
        mid[finite] = 0.5 * (lower[finite] + upper[finite])

        tip_names = tuple(str(t.name) for t in hand_profile.fingertip_links)
        return Stage1RobotGeometry(
            hand_profile=hand_profile,
            hand_fk=fk,
            joint_lower=lower,
            joint_upper=upper,
            rest_pose=mid,
            fingertip_names=tip_names,
        )

    @staticmethod
    def from_hand_config(
        *,
        hand_name: str | None = None,
        hand_config_path: str | None = None,
        strict_urdf: bool = True,
    ) -> "Stage1RobotGeometry":
        """Convenience builder from hand name and/or explicit config path."""

        profile = load_hand_profile(
            hand_name=hand_name,
            config_path=hand_config_path,
            strict_urdf=bool(strict_urdf),
        )
        if profile is None:
            raise ValueError(
                "Could not resolve hand profile. Provide hand_name or hand_config_path."
            )
        return Stage1RobotGeometry.from_hand_profile(profile)

    @property
    def joint_count(self) -> int:
        """Return robot DoF count."""

        return int(self.joint_lower.shape[0])

    @property
    def fingertip_count(self) -> int:
        """Return number of configured fingertips."""

        return int(len(self.fingertip_names))

    def collision_radii_for_links(self, link_names: tuple[str, ...]) -> np.ndarray:
        """Return collision bounding sphere radii aligned with ``link_names``."""

        radii = self.hand_fk.collision_radii
        return np.array([radii.get(name, 0.0) for name in link_names], dtype=np.float32)

    def link_adjacency_mask(self, link_names: tuple[str, ...]) -> np.ndarray:
        """Return boolean mask ``[L,L]``: True if link pair is adjacent (skip for self-collision)."""

        adj = self.hand_fk.adjacent_links
        n = len(link_names)
        mask = np.zeros((n, n), dtype=bool)
        for i, a in enumerate(link_names):
            for j, b in enumerate(link_names):
                if (a, b) in adj:
                    mask[i, j] = True
        return mask

    def collision_primitives_for_links(
        self,
        link_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flat per-primitive collision sphere data aligned with ``link_names``.

        Returns:
            centers_local: ``[P, 3]`` sphere centers in each owning link's frame.
            radii: ``[P]`` sphere radii.
            link_indices: ``[P]`` index into ``link_names`` per primitive.
        """

        primitives = self.hand_fk.collision_primitives
        centers: list[np.ndarray] = []
        radii: list[float] = []
        link_idx: list[int] = []
        for li, name in enumerate(link_names):
            for c, r in primitives.get(name, []):
                centers.append(np.asarray(c, dtype=np.float64))
                radii.append(float(r))
                link_idx.append(int(li))
        if not centers:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.stack(centers, axis=0).astype(np.float64),
            np.asarray(radii, dtype=np.float64),
            np.asarray(link_idx, dtype=np.int64),
        )

    def scale_raw_to_limits_torch(self, torch: Any, q_raw: Any) -> Any:
        """Map unconstrained raw joint outputs to valid joint ranges."""

        lower = torch.as_tensor(self.joint_lower, dtype=q_raw.dtype, device=q_raw.device)
        upper = torch.as_tensor(self.joint_upper, dtype=q_raw.dtype, device=q_raw.device)
        sigma = torch.sigmoid(q_raw)
        return lower[None, None, :] + sigma * (upper[None, None, :] - lower[None, None, :])

    def clamp_to_limits_torch(self, torch: Any, q: Any) -> Any:
        """Clamp joint vectors to URDF limits."""

        lower = torch.as_tensor(self.joint_lower, dtype=q.dtype, device=q.device)
        upper = torch.as_tensor(self.joint_upper, dtype=q.dtype, device=q.device)
        return torch.maximum(torch.minimum(q, upper), lower)

    def fingertips_from_q_torch(self, torch: Any, q_btj: Any) -> Any:
        """Run batched FK and return fingertip points in retarget-origin frame."""

        if int(q_btj.ndim) != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {tuple(q_btj.shape)}")
        b, t, j = int(q_btj.shape[0]), int(q_btj.shape[1]), int(q_btj.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        flat = q_btj.reshape(b * t, j)
        tips = self.hand_fk.fingertips_from_qpos_batch_torch(torch, flat)
        return tips.reshape(b, t, self.fingertip_count, 3)

    def link_transforms_from_q_torch(
        self,
        torch: Any,
        q_btj: Any,
        *,
        base_frame: bool = True,
        link_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Run differentiable batched FK and return link transforms.

        Args:
            q_btj: joint tensor ``[B,T,J]``.
            base_frame: when true, outputs are relative to base link.
            link_names: optional subset of links.
        Returns:
            Mapping ``link_name -> [B,T,4,4]`` tensor.
        """

        if int(q_btj.ndim) != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {tuple(q_btj.shape)}")
        b, t, j = int(q_btj.shape[0]), int(q_btj.shape[1]), int(q_btj.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        flat = q_btj.reshape(b * t, j)
        tf_map = self.hand_fk.link_transforms_from_qpos_batch_torch(
            torch,
            flat,
            base_frame=bool(base_frame),
            link_names=link_names,
        )
        return {
            str(name): tf.reshape(b, t, 4, 4)
            for name, tf in tf_map.items()
        }

    def link_points_from_q_torch(
        self,
        torch: Any,
        q_btj: Any,
        *,
        base_frame: bool = True,
        link_names: tuple[str, ...] | None = None,
    ) -> tuple[Any, tuple[str, ...]]:
        """Run differentiable batched FK and return link-origin points.

        Returns:
            points: ``[B,T,L,3]``.
            names: tuple of link names aligned with ``L``.
        """

        if int(q_btj.ndim) != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {tuple(q_btj.shape)}")
        b, t, j = int(q_btj.shape[0]), int(q_btj.shape[1]), int(q_btj.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        flat = q_btj.reshape(b * t, j)
        pts_flat, names = self.hand_fk.link_points_from_qpos_batch_torch(
            torch,
            flat,
            base_frame=bool(base_frame),
            link_names=link_names,
        )
        return pts_flat.reshape(b, t, pts_flat.shape[1], 3), names

    def primitive_points_from_q_torch(
        self,
        torch: Any,
        q_btj: Any,
        *,
        base_frame: bool = True,
    ) -> tuple[Any, Any, Any, tuple[str, ...]]:
        """Differentiable FK returning world-frame collision-primitive sphere centers.

        Each link contributes one or more bounding spheres (one per URDF
        ``<collision>`` element, with elongated boxes sphere-chained). Used
        by the self-collision loss so the per-pair margin reflects actual
        link geometry rather than oversized single-sphere-per-link bounds.

        Returns:
            points: ``[B, T, P, 3]`` differentiable primitive centers.
            radii: ``[P]`` (CPU tensor, constant — radii do not depend on q).
            link_indices: ``[P]`` int (CPU tensor) into ``link_names``.
            link_names: tuple aligned with ``link_indices``.
        """

        if int(q_btj.ndim) != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {tuple(q_btj.shape)}")
        b, t, j = int(q_btj.shape[0]), int(q_btj.shape[1]), int(q_btj.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        link_names = self.hand_fk._link_names
        centers_local_np, radii_np, link_indices_np = self.collision_primitives_for_links(link_names)
        p = int(centers_local_np.shape[0])
        radii_t = torch.as_tensor(radii_np, dtype=q_btj.dtype, device=q_btj.device)
        link_idx_t = torch.as_tensor(link_indices_np, dtype=torch.long, device=q_btj.device)
        if p == 0:
            return (
                torch.zeros((b, t, 0, 3), dtype=q_btj.dtype, device=q_btj.device),
                radii_t,
                link_idx_t,
                link_names,
            )
        flat = q_btj.reshape(b * t, j)
        tf_map = self.hand_fk.link_transforms_from_qpos_batch_torch(
            torch,
            flat,
            base_frame=bool(base_frame),
            link_names=link_names,
        )
        centers_local_t = torch.as_tensor(
            centers_local_np, dtype=q_btj.dtype, device=q_btj.device
        )  # [P, 3]
        per_link_pts: list = []
        for pi in range(p):
            tf = tf_map[link_names[int(link_indices_np[pi])]]  # [bt, 4, 4]
            c_local = centers_local_t[pi]  # [3]
            # world_center = R @ c_local + t
            rotated = torch.matmul(tf[:, :3, :3], c_local.reshape(3, 1)).squeeze(-1)
            world_center = rotated + tf[:, :3, 3]
            per_link_pts.append(world_center)
        pts = torch.stack(per_link_pts, dim=1)  # [bt, P, 3]
        return pts.reshape(b, t, p, 3), radii_t, link_idx_t, link_names

    def fingertips_from_q_np(self, q_btj: np.ndarray) -> np.ndarray:
        """Run batched FK and return fingertip points in retarget-origin frame (numpy)."""

        q = np.asarray(q_btj, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {q.shape}")
        b, t, j = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        flat = q.reshape(b * t, j)
        tips = self.hand_fk.fingertips_from_qpos_batch(flat)
        return tips.reshape(b, t, self.fingertip_count, 3).astype(np.float32)

    def link_transforms_from_q_np(
        self,
        q_btj: np.ndarray,
        *,
        base_frame: bool = True,
    ) -> list[dict[str, np.ndarray]]:
        """Return per-sample link transforms for one ``[B,T,J]`` joint tensor."""

        q = np.asarray(q_btj, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {q.shape}")
        flat = q.reshape(-1, q.shape[-1])
        return [
            self.hand_fk.link_transforms_from_qpos(vec, base_frame=bool(base_frame))
            for vec in flat
        ]

    def link_points_from_q_np(
        self,
        q_btj: np.ndarray,
        *,
        base_frame: bool = True,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Run batched FK and return all link-origin points (numpy).

        Returns:
            points: ``[B,T,L,3]`` float64 array of link positions.
            names: tuple of link names aligned with the ``L`` dimension.
        """

        q = np.asarray(q_btj, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {q.shape}")
        b, t, j = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        flat = q.reshape(b * t, j)
        link_names = self.hand_fk._link_names
        all_points = np.zeros((b * t, len(link_names), 3), dtype=np.float64)
        for idx, qvec in enumerate(flat):
            tf_map = self.hand_fk.link_transforms_from_qpos(qvec, base_frame=bool(base_frame))
            for li, name in enumerate(link_names):
                all_points[idx, li] = tf_map[name][:3, 3]
        return all_points.reshape(b, t, len(link_names), 3), link_names

    def primitive_points_from_q_np(
        self,
        q_btj: np.ndarray,
        *,
        base_frame: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        """Run batched FK and return world-frame collision-primitive sphere centers.

        Each link contributes one sphere per ``<collision>`` element, placed at
        the primitive's offset within the link frame (see
        ``collision_primitives_for_links``). This is the recommended input for
        link-aware self-collision and penetration checks; using single
        link-origin points (``link_points_from_q_np``) misses primitives whose
        geometry sits far from the joint origin.

        Returns:
            points: ``[B, T, P, 3]`` float64 array of primitive centers.
            radii: ``[P]`` sphere radii.
            link_indices: ``[P]`` index into ``link_names`` per primitive.
            link_names: tuple of link names aligned with ``link_indices``.
        """

        q = np.asarray(q_btj, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {q.shape}")
        b, t, j = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")
        link_names = self.hand_fk._link_names
        centers_local, radii, link_indices = self.collision_primitives_for_links(link_names)
        p = int(centers_local.shape[0])
        if p == 0:
            return (
                np.zeros((b, t, 0, 3), dtype=np.float64),
                radii,
                link_indices,
                link_names,
            )
        flat = q.reshape(b * t, j)
        out = np.zeros((b * t, p, 3), dtype=np.float64)
        for idx, qvec in enumerate(flat):
            tf_map = self.hand_fk.link_transforms_from_qpos(qvec, base_frame=bool(base_frame))
            for pi in range(p):
                tf = tf_map[link_names[int(link_indices[pi])]]
                out[idx, pi] = tf[:3, :3] @ centers_local[pi] + tf[:3, 3]
        return out.reshape(b, t, p, 3), radii, link_indices, link_names

    def correspondence_points_from_q_np(
        self,
        q_btj: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Compute correspondence points (tips + config extras) with offsets (numpy).

        Returns:
            points: ``[B,T,C,3]`` where C = len(fingertip_links) + len(dexpilot_extra_points).
            human_hand_ids: tuple of human keypoint IDs aligned with C.
        """

        q = np.asarray(q_btj, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"Expected q_btj [B,T,J], got {q.shape}")
        b, t, j = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        if j < self.joint_count:
            raise ValueError(f"Expected at least J={self.joint_count}, got {j}")

        tip_specs = self.hand_profile.fingertip_links
        extra_specs = self.hand_profile.dexpilot_extra_points
        all_specs: list[tuple[str, tuple[float, float, float], int]] = []
        for s in tip_specs:
            all_specs.append((s.link, s.center_offset, s.human_hand_id))
        for s in extra_specs:
            all_specs.append((s.link, s.center_offset, s.human_hand_id))

        c = len(all_specs)
        human_hand_ids = tuple(s[2] for s in all_specs)
        origin_link = self.hand_profile.retarget_origin_link

        flat = q.reshape(b * t, j)
        out = np.zeros((b * t, c, 3), dtype=np.float64)
        for idx, qvec in enumerate(flat):
            tf_map = self.hand_fk.link_transforms_from_qpos(qvec, base_frame=False)
            origin_inv = np.linalg.inv(tf_map[origin_link])
            for ci, (link, offset, _) in enumerate(all_specs):
                link_tf = tf_map[link]
                off = np.asarray(offset, dtype=np.float64).reshape(3)
                world_pt = link_tf[:3, :3] @ off + link_tf[:3, 3]
                local_pt = origin_inv[:3, :3] @ world_pt + origin_inv[:3, 3]
                out[idx, ci] = local_pt
        return out.reshape(b, t, c, 3), human_hand_ids

    def fingertip_name_to_index(self) -> dict[str, int]:
        """Return fingertip-name lookup."""

        return {name: i for i, name in enumerate(self.fingertip_names)}

    def joint_limit_violation_rate_np(self, q_btj: np.ndarray, atol: float = 1e-6) -> float:
        """Compute fraction of joints outside configured limits."""

        q = np.asarray(q_btj, dtype=np.float64)
        lower = np.asarray(self.joint_lower, dtype=np.float64).reshape(1, 1, -1)
        upper = np.asarray(self.joint_upper, dtype=np.float64).reshape(1, 1, -1)
        below = q < (lower - float(atol))
        above = q > (upper + float(atol))
        return float(np.mean(np.logical_or(below, above)))
