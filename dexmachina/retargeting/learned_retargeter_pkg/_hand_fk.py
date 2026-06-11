"""Lightweight URDF-based hand forward kinematics utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

_log = logging.getLogger(__name__)

from ._hand_profile import HandProfile


@dataclass(frozen=True)
class URDFJointSpec:
    """One URDF joint transform specification."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    limit_lower: float | None
    limit_upper: float | None


def _parse_vec3(raw: str | None) -> tuple[float, float, float]:
    """Parse URDF xyz/rpy-like vector attributes with default zeros."""

    if raw is None or not str(raw).strip():
        return (0.0, 0.0, 0.0)
    vals = [float(x) for x in str(raw).strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3-vector, got {vals}")
    return (vals[0], vals[1], vals[2])


def _rotation_x(theta: float) -> np.ndarray:
    """Build 3x3 rotation around x-axis."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rotation_y(theta: float) -> np.ndarray:
    """Build 3x3 rotation around y-axis."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rotation_z(theta: float) -> np.ndarray:
    """Build 3x3 rotation around z-axis."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """Convert URDF roll-pitch-yaw convention to 3x3 matrix."""

    roll, pitch, yaw = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Compute Rodrigues rotation matrix for axis-angle pair."""

    axis_n = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis_n))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis_n / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Pack 3x3 + 3-vector into homogeneous 4x4 transform."""

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


def _joint_origin_transform(spec: URDFJointSpec) -> np.ndarray:
    """Return fixed parent-to-joint transform from URDF origin tag."""

    rot = _rpy_to_matrix(spec.origin_rpy)
    trn = np.asarray(spec.origin_xyz, dtype=np.float64)
    return _transform_from_rt(rot, trn)


def _parse_urdf_joint_specs(urdf_path: Path) -> tuple[dict[str, URDFJointSpec], dict[str, str]]:
    """Parse URDF into joint specs and parent-joint lookup by child link."""

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if str(root.tag).strip().lower() != "robot":
        raise ValueError(f"URDF root tag must be <robot>, got <{root.tag}> in {urdf_path}")

    joint_by_name: dict[str, URDFJointSpec] = {}
    parent_joint_by_child: dict[str, str] = {}
    for elem in root.findall("joint"):
        attrib = dict(elem.attrib or {})
        name = str(attrib.get("name", "")).strip()
        if not name:
            raise ValueError(f"URDF joint without name in {urdf_path}")
        joint_type = str(attrib.get("type", "fixed")).strip().lower()

        parent_elem = elem.find("parent")
        child_elem = elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"URDF joint '{name}' missing parent/child tag in {urdf_path}")
        parent_link = str(parent_elem.attrib.get("link", "")).strip()
        child_link = str(child_elem.attrib.get("link", "")).strip()
        if not parent_link or not child_link:
            raise ValueError(f"URDF joint '{name}' has invalid parent/child link in {urdf_path}")

        origin_elem = elem.find("origin")
        origin_xyz = _parse_vec3(None if origin_elem is None else origin_elem.attrib.get("xyz"))
        origin_rpy = _parse_vec3(None if origin_elem is None else origin_elem.attrib.get("rpy"))

        axis_elem = elem.find("axis")
        axis_xyz = _parse_vec3(None if axis_elem is None else axis_elem.attrib.get("xyz"))

        lower: float | None = None
        upper: float | None = None
        limit_elem = elem.find("limit")
        if limit_elem is not None:
            if "lower" in limit_elem.attrib:
                lower = float(limit_elem.attrib["lower"])
            if "upper" in limit_elem.attrib:
                upper = float(limit_elem.attrib["upper"])

        spec = URDFJointSpec(
            name=name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis_xyz=axis_xyz,
            limit_lower=lower,
            limit_upper=upper,
        )
        if name in joint_by_name:
            raise ValueError(f"Duplicate URDF joint name '{name}' in {urdf_path}")
        joint_by_name[name] = spec

        if child_link in parent_joint_by_child:
            raise ValueError(
                f"URDF child link '{child_link}' has multiple parent joints in {urdf_path}"
            )
        parent_joint_by_child[child_link] = name

    return joint_by_name, parent_joint_by_child


def _obj_mesh_bounding_radius(mesh_path: Path, scale: tuple[float, float, float]) -> float:
    """Compute bounding sphere radius of an OBJ mesh from its origin."""

    try:
        max_r2 = 0.0
        with open(mesh_path) as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                x = float(parts[1]) * scale[0]
                y = float(parts[2]) * scale[1]
                z = float(parts[3]) * scale[2]
                max_r2 = max(max_r2, x * x + y * y + z * z)
        return math.sqrt(max_r2)
    except Exception:
        _log.warning("Could not load collision mesh %s, using radius 0.", mesh_path)
        return 0.0


def _obj_mesh_aabb_bounding_sphere(
    mesh_path: Path,
    scale: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """AABB-center bounding sphere of an OBJ mesh, in the mesh's own frame.

    Returns (center, radius) where center is the AABB midpoint and radius is
    the max vertex distance from that center. Centering on the AABB midpoint
    (instead of the mesh's local origin) yields a much tighter sphere when
    the geometry is offset from its frame origin.
    """
    try:
        verts: list[tuple[float, float, float]] = []
        with open(mesh_path) as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                verts.append((
                    float(parts[1]) * scale[0],
                    float(parts[2]) * scale[1],
                    float(parts[3]) * scale[2],
                ))
        if not verts:
            return np.zeros(3, dtype=np.float64), 0.0
        v = np.asarray(verts, dtype=np.float64)
        center = 0.5 * (v.min(axis=0) + v.max(axis=0))
        radius = float(np.max(np.linalg.norm(v - center, axis=1)))
        return center.astype(np.float64), radius
    except Exception:
        _log.warning("Could not load collision mesh %s, using radius 0.", mesh_path)
        return np.zeros(3, dtype=np.float64), 0.0


def _collision_primitive_radius(geom_elem: Any, urdf_dir: Path) -> float:
    """Bounding sphere radius of one URDF collision geometry primitive."""

    box = geom_elem.find("box")
    if box is not None:
        size = _parse_vec3(box.attrib.get("size"))
        return math.sqrt(sum((s / 2.0) ** 2 for s in size))
    sphere = geom_elem.find("sphere")
    if sphere is not None:
        return float(sphere.attrib.get("radius", "0"))
    cylinder = geom_elem.find("cylinder")
    if cylinder is not None:
        r = float(cylinder.attrib.get("radius", "0"))
        l = float(cylinder.attrib.get("length", "0"))
        return math.sqrt(r * r + (l / 2.0) ** 2)
    mesh = geom_elem.find("mesh")
    if mesh is not None:
        fname = str(mesh.attrib.get("filename", "")).strip()
        scale_str = mesh.attrib.get("scale")
        scale = _parse_vec3(scale_str) if scale_str else (1.0, 1.0, 1.0)
        mesh_path = urdf_dir / fname
        if mesh_path.suffix.lower() == ".obj" and mesh_path.exists():
            return _obj_mesh_bounding_radius(mesh_path, scale)
        _log.debug("Skipping non-OBJ or missing collision mesh: %s", mesh_path)
        return 0.0
    return 0.0


def _parse_urdf_collision_radii(urdf_path: Path) -> dict[str, float]:
    """Parse URDF and return per-link bounding sphere radius from collision geometry.

    Each link's radius is the max over its collision elements of
    ``||offset|| + primitive_bounding_radius``, giving a sphere centered
    at the link origin that encompasses all collision geometry.

    Note: this single-sphere-per-link model is too coarse for collision
    checking (a link's bounding sphere centered at the joint origin can be
    arbitrarily large when the geometry is offset from the joint). Prefer
    ``_parse_urdf_collision_primitives`` for self-collision / penetration
    checks. This function is kept for legacy callers that still use the
    per-link radius (e.g. training-side soft loss buffers).
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    radii: dict[str, float] = {}
    for link_elem in root.findall("link"):
        link_name = str(link_elem.attrib.get("name", "")).strip()
        if not link_name:
            continue
        max_radius = 0.0
        for coll in link_elem.findall("collision"):
            origin = coll.find("origin")
            offset = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
            offset_dist = math.sqrt(sum(x * x for x in offset))
            geom = coll.find("geometry")
            if geom is None:
                continue
            prim_r = _collision_primitive_radius(geom, urdf_dir)
            max_radius = max(max_radius, offset_dist + prim_r)
        radii[link_name] = max_radius
    return radii


def _box_sphere_chain(
    size: tuple[float, float, float],
) -> list[tuple[np.ndarray, float]]:
    """Approximate an axis-aligned box with a chain of bounding spheres.

    A single bounding sphere has half-diagonal radius (√3 × half-extent for a
    cube), which inflates elongated boxes substantially. Splitting along the
    longest axis into N equal slabs and bounding each slab gives a much tighter
    fit. N is chosen so the slab length is comparable to the cross-section
    width, keeping per-sphere inflation under √2.

    PAPER NOTE: this sphere-chain construction (and the per-primitive sphere
    model in ``_parse_urdf_collision_primitives``) underlies the penetration
    and self-collision metrics described in ``evaluate_oakink_grouped.py``
    (search "PAPER NOTE"). Mention the construction and the conservative-
    bound caveat in the methods section.

    Args:
        size: box dimensions ``(sx, sy, sz)``.

    Returns:
        List of ``(center, radius)`` spheres in the box's local frame
        (i.e. centered at the box origin).
    """

    sx, sy, sz = (max(0.0, float(s)) for s in size)
    extents = (sx, sy, sz)
    if max(extents) <= 0.0:
        return []
    long_axis = int(np.argmax(extents))
    long_extent = extents[long_axis]
    cross = [extents[i] for i in range(3) if i != long_axis]
    cross_max = max(cross) if cross else long_extent
    n = max(1, int(math.ceil(long_extent / max(cross_max, 1.0e-9))))
    slab_extent = long_extent / float(n)
    half_cross_x = cross[0] * 0.5
    half_cross_y = cross[1] * 0.5
    radius = math.sqrt((slab_extent / 2.0) ** 2 + half_cross_x ** 2 + half_cross_y ** 2)
    spheres: list[tuple[np.ndarray, float]] = []
    for i in range(n):
        center = np.zeros(3, dtype=np.float64)
        center[long_axis] = -long_extent / 2.0 + slab_extent * (i + 0.5)
        spheres.append((center, float(radius)))
    return spheres


def _collision_primitive_spheres(
    geom_elem: Any,
    urdf_dir: Path,
) -> list[tuple[np.ndarray, float]]:
    """Return a list of bounding spheres approximating one URDF collision primitive.

    Boxes are split into a sphere chain along their longest axis; spheres,
    cylinders, and meshes are returned as a single sphere.
    """

    box = geom_elem.find("box")
    if box is not None:
        size = _parse_vec3(box.attrib.get("size"))
        return _box_sphere_chain(size)
    sphere = geom_elem.find("sphere")
    if sphere is not None:
        r = float(sphere.attrib.get("radius", "0"))
        if r <= 0.0:
            return []
        return [(np.zeros(3, dtype=np.float64), r)]
    cylinder = geom_elem.find("cylinder")
    if cylinder is not None:
        r = float(cylinder.attrib.get("radius", "0"))
        length = float(cylinder.attrib.get("length", "0"))
        rad = math.sqrt(r * r + (length / 2.0) ** 2)
        if rad <= 0.0:
            return []
        return [(np.zeros(3, dtype=np.float64), rad)]
    mesh = geom_elem.find("mesh")
    if mesh is not None:
        fname = str(mesh.attrib.get("filename", "")).strip()
        scale_str = mesh.attrib.get("scale")
        scale = _parse_vec3(scale_str) if scale_str else (1.0, 1.0, 1.0)
        mesh_path = urdf_dir / fname
        if mesh_path.suffix.lower() == ".obj" and mesh_path.exists():
            c, r = _obj_mesh_aabb_bounding_sphere(mesh_path, scale)
            if r <= 0.0:
                return []
            return [(c, r)]
        _log.debug("Skipping non-OBJ or missing collision mesh: %s", mesh_path)
    return []


def _parse_urdf_collision_primitives(
    urdf_path: Path,
) -> dict[str, list[tuple[np.ndarray, float]]]:
    """Return per-link list of ``(center_in_link_frame, radius)`` bounding spheres.

    One sphere is generated per ``<collision>`` element. The sphere is placed
    at the primitive geometry's actual position in the link frame (i.e. the
    ``<origin>`` transform composed with the geometry's local sphere center),
    not at the joint origin. This yields a multi-sphere approximation that
    closely tracks the link's collision shape and avoids the over-conservative
    single-sphere-at-joint-origin model used by ``_parse_urdf_collision_radii``.
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    primitives: dict[str, list[tuple[np.ndarray, float]]] = {}
    for link_elem in root.findall("link"):
        link_name = str(link_elem.attrib.get("name", "")).strip()
        if not link_name:
            continue
        link_primitives: list[tuple[np.ndarray, float]] = []
        for coll in link_elem.findall("collision"):
            origin = coll.find("origin")
            xyz = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
            rpy = _parse_vec3(origin.attrib.get("rpy") if origin is not None else None)
            geom = coll.find("geometry")
            if geom is None:
                continue
            rot = _rpy_to_matrix(rpy)
            offset = np.asarray(xyz, dtype=np.float64)
            for shape_center, radius in _collision_primitive_spheres(geom, urdf_dir):
                center = rot @ shape_center + offset
                link_primitives.append((center.astype(np.float64), float(radius)))
        primitives[link_name] = link_primitives
    return primitives


class HandKinematics:
    """Numerical hand forward kinematics from legacy hand profile + URDF."""

    def __init__(self, hand_profile: HandProfile):
        """Build FK runtime state from one validated hand profile."""

        self.profile = hand_profile
        joint_specs, parent_joint_by_child = _parse_urdf_joint_specs(hand_profile.urdf_path)
        self._joint_specs = joint_specs
        self._parent_joint_by_child = parent_joint_by_child
        self._joint_origin_tf = {
            name: _joint_origin_transform(spec) for name, spec in joint_specs.items()
        }
        self._joint_index_by_name = {
            str(name): int(idx) for idx, name in enumerate(hand_profile.joint_order)
        }
        link_names: set[str] = {
            str(self.profile.base_link),
            str(self.profile.retarget_origin_link),
        }
        for spec in joint_specs.values():
            link_names.add(str(spec.parent_link))
            link_names.add(str(spec.child_link))
        self._link_names = tuple(sorted(link_names))

        for joint_name in hand_profile.joint_order:
            if joint_name not in self._joint_specs:
                raise ValueError(
                    f"Hand profile '{hand_profile.name}' joint '{joint_name}' "
                    f"is not present in URDF {hand_profile.urdf_path}."
                )

        self._base_joint_chain = self._joint_chain_to_link(self.profile.base_link)
        self._origin_joint_chain = self._joint_chain_to_link(self.profile.retarget_origin_link)
        self._fingertip_joint_chains = tuple(
            self._joint_chain_to_link(tip.link) for tip in self.profile.fingertip_links
        )
        self._link_joint_chains = {
            str(link_name): self._joint_chain_to_link(str(link_name))
            for link_name in self._link_names
        }

        # Per-link collision bounding sphere radii from URDF collision geometry
        # (single-sphere-at-joint-origin; coarse, kept for legacy callers).
        self._collision_radii = _parse_urdf_collision_radii(hand_profile.urdf_path)
        # Per-link multi-sphere primitives: one sphere per ``<collision>`` element,
        # placed at the primitive's actual offset within the link frame.
        self._collision_primitives = _parse_urdf_collision_primitives(hand_profile.urdf_path)

        # Link adjacency (parent-child pairs connected by a joint).
        self._adjacent_links: set[tuple[str, str]] = set()
        for spec in joint_specs.values():
            self._adjacent_links.add((spec.parent_link, spec.child_link))
            self._adjacent_links.add((spec.child_link, spec.parent_link))

    @property
    def collision_radii(self) -> dict[str, float]:
        """Per-link bounding sphere radii from URDF collision geometry."""
        return dict(self._collision_radii)

    @property
    def collision_primitives(self) -> dict[str, list[tuple[np.ndarray, float]]]:
        """Per-link bounding spheres of each ``<collision>`` primitive.

        Each entry is ``(center_in_link_frame, radius)``. A link with no
        collision geometry maps to an empty list.
        """
        return {
            name: [(c.copy(), float(r)) for (c, r) in prims]
            for name, prims in self._collision_primitives.items()
        }

    @property
    def adjacent_links(self) -> set[tuple[str, str]]:
        """Set of (link_a, link_b) pairs directly connected by a joint."""
        return set(self._adjacent_links)

    @property
    def joint_count(self) -> int:
        """Return expected controlled joint dimension for qpos input."""

        return int(len(self.profile.joint_order))

    def _clip_joint(self, joint_name: str, value: float) -> float:
        """Clamp one joint value to URDF limits when limits are defined."""

        spec = self._joint_specs.get(joint_name)
        if spec is None:
            return float(value)
        out = float(value)
        if spec.limit_lower is not None:
            out = max(out, float(spec.limit_lower))
        if spec.limit_upper is not None:
            out = min(out, float(spec.limit_upper))
        return out

    def joint_limits_lower_upper(
        self,
        *,
        fill_lower: float = -np.inf,
        fill_upper: float = np.inf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint lower/upper limit vectors in ``joint_order``."""

        lower = np.full((self.joint_count,), float(fill_lower), dtype=np.float64)
        upper = np.full((self.joint_count,), float(fill_upper), dtype=np.float64)
        for idx, joint_name in enumerate(self.profile.joint_order):
            spec = self._joint_specs.get(str(joint_name))
            if spec is None:
                continue
            if spec.limit_lower is not None:
                lower[idx] = float(spec.limit_lower)
            if spec.limit_upper is not None:
                upper[idx] = float(spec.limit_upper)
        return lower, upper

    def _normalize_q(self, qpos: np.ndarray) -> dict[str, float]:
        """Map raw qpos vector to joint-name dictionary with mimic enforcement."""

        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.shape[0] < self.joint_count:
            raise ValueError(
                f"Expected qpos with at least {self.joint_count} entries, got {q.shape[0]}."
            )
        q = np.asarray(q[: self.joint_count], dtype=np.float64)
        out: dict[str, float] = {
            name: self._clip_joint(name, float(q[idx]))
            for idx, name in enumerate(self.profile.joint_order)
        }

        # Legacy mimic convention: follower = multiplier * master + offset.
        for mimic in self.profile.mimic_joints:
            master_val = out[mimic.master_joint]
            follower_val = float(mimic.multiplier) * float(master_val) + float(mimic.offset)
            out[mimic.follower_joint] = self._clip_joint(mimic.follower_joint, follower_val)
        return out

    def _joint_motion_transform(self, spec: URDFJointSpec, value: float) -> np.ndarray:
        """Build joint motion transform in joint local frame."""

        t = str(spec.joint_type).strip().lower()
        if t in {"fixed"}:
            return np.eye(4, dtype=np.float64)
        if t in {"revolute", "continuous"}:
            axis = np.asarray(spec.axis_xyz, dtype=np.float64)
            rot = _axis_angle_to_matrix(axis, float(value))
            return _transform_from_rt(rot, np.zeros((3,), dtype=np.float64))
        if t == "prismatic":
            axis = np.asarray(spec.axis_xyz, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axis = axis / norm
            trn = axis * float(value)
            return _transform_from_rt(np.eye(3, dtype=np.float64), trn)
        raise ValueError(
            f"Unsupported URDF joint type '{spec.joint_type}' for joint '{spec.name}'. "
            "Supported: fixed, revolute, continuous, prismatic."
        )

    def _joint_chain_to_link(self, link_name: str) -> tuple[str, ...]:
        """Return ordered joint-name chain from root to ``link_name``."""

        chain: list[str] = []
        seen_links: set[str] = set()
        current = str(link_name)
        while True:
            if current in seen_links:
                raise ValueError(
                    f"Cycle detected while tracing link chain for '{link_name}' in "
                    f"{self.profile.urdf_path}."
                )
            seen_links.add(current)
            parent_joint = self._parent_joint_by_child.get(current)
            if parent_joint is None:
                break
            chain.append(parent_joint)
            current = str(self._joint_specs[parent_joint].parent_link)
        chain.reverse()
        return tuple(chain)

    @staticmethod
    def _skew_torch(torch: Any, vec: Any) -> Any:
        """Build batched skew matrix [B,3,3] from [B,3] vectors."""

        x = vec[:, 0]
        y = vec[:, 1]
        z = vec[:, 2]
        zeros = torch.zeros_like(x)
        row0 = torch.stack([zeros, -z, y], dim=1)
        row1 = torch.stack([z, zeros, -x], dim=1)
        row2 = torch.stack([-y, x, zeros], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def _joint_motion_transform_torch(
        self,
        torch: Any,
        spec: URDFJointSpec,
        value: Any,
    ) -> Any:
        """Build batched joint local transform [B,4,4] from joint values [B]."""

        b = int(value.shape[0])
        dtype = value.dtype
        device = value.device
        out = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
        t = str(spec.joint_type).strip().lower()

        if t in {"fixed"}:
            return out

        axis_np = np.asarray(spec.axis_xyz, dtype=np.float64).reshape(3)
        axis_t = torch.as_tensor(axis_np, device=device, dtype=dtype).reshape(1, 3).repeat(b, 1)
        axis_norm = torch.linalg.norm(axis_t, dim=1, keepdim=True)
        safe = axis_norm > 1.0e-12
        axis_unit = torch.where(safe, axis_t / axis_norm.clamp_min(1.0e-12), axis_t)

        if t in {"revolute", "continuous"}:
            k = self._skew_torch(torch, axis_unit)
            eye3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
            angle = value.reshape(b, 1, 1)
            sin = torch.sin(angle)
            cos = torch.cos(angle)
            rot = eye3 + sin * k + (1.0 - cos) * (k @ k)
            out[:, :3, :3] = rot
            return out

        if t == "prismatic":
            trn = axis_unit * value.reshape(b, 1)
            out[:, :3, 3] = trn
            return out

        raise ValueError(
            f"Unsupported URDF joint type '{spec.joint_type}' for joint '{spec.name}'. "
            "Supported: fixed, revolute, continuous, prismatic."
        )

    def _chain_transform_torch(
        self,
        torch: Any,
        qmap: dict[str, Any],
        chain: tuple[str, ...],
        *,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """Compose batched root-to-link transforms for one precomputed chain."""

        tf = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        for joint_name in chain:
            spec = self._joint_specs[joint_name]
            origin_tf = torch.as_tensor(
                self._joint_origin_tf[joint_name],
                device=device,
                dtype=dtype,
            ).unsqueeze(0)
            motion_val = qmap.get(joint_name, None)
            if motion_val is None:
                motion_val = torch.zeros((batch_size,), device=device, dtype=dtype)
            motion_tf = self._joint_motion_transform_torch(torch, spec, motion_val)
            tf = tf @ origin_tf @ motion_tf
        return tf

    def _link_transform_world(
        self,
        link_name: str,
        qmap: dict[str, float],
        cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute world transform for one link with memoized recursion."""

        cached = cache.get(link_name)
        if cached is not None:
            return cached

        parent_joint_name = self._parent_joint_by_child.get(link_name)
        if parent_joint_name is None:
            identity = np.eye(4, dtype=np.float64)
            cache[link_name] = identity
            return identity

        spec = self._joint_specs[parent_joint_name]
        parent_tf = self._link_transform_world(spec.parent_link, qmap, cache)
        origin_tf = self._joint_origin_tf[parent_joint_name]
        motion_tf = self._joint_motion_transform(spec, qmap.get(spec.name, 0.0))
        out = parent_tf @ origin_tf @ motion_tf
        cache[link_name] = out
        return out

    def fingertips_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Compute fingertip points [F,3] in retarget-origin (wrist) frame from qpos."""

        qmap = self._normalize_q(qpos)
        link_cache: dict[str, np.ndarray] = {}
        origin_tf = self._link_transform_world(self.profile.retarget_origin_link, qmap, link_cache)
        origin_inv = np.linalg.inv(origin_tf)

        points: list[np.ndarray] = []
        for tip in self.profile.fingertip_links:
            link_tf = self._link_transform_world(tip.link, qmap, link_cache)
            offset = np.asarray(tip.center_offset, dtype=np.float64).reshape(3)
            world_pt = link_tf[:3, :3] @ offset + link_tf[:3, 3]
            origin_pt = origin_inv[:3, :3] @ world_pt + origin_inv[:3, 3]
            points.append(origin_pt.astype(np.float64))

        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(points, axis=0).astype(np.float32)

    def fingertips_from_qpos_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        """Compute fingertip points [N,F,3] from qpos batch [N,D]."""

        q = np.asarray(qpos_batch, dtype=np.float64)
        if q.ndim != 2:
            raise ValueError(f"Expected qpos_batch [N,D], got {q.shape}")
        return np.stack([self.fingertips_from_qpos(q[i]) for i in range(q.shape[0])], axis=0)

    def link_transforms_from_qpos(
        self,
        qpos: np.ndarray,
        *,
        base_frame: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute 4x4 transforms for all known links from one qpos vector."""

        qmap = self._normalize_q(qpos)
        cache: dict[str, np.ndarray] = {}
        base_tf = self._link_transform_world(self.profile.base_link, qmap, cache)
        base_inv = np.linalg.inv(base_tf) if bool(base_frame) else np.eye(4, dtype=np.float64)

        out: dict[str, np.ndarray] = {}
        for link_name in self._link_names:
            world_tf = self._link_transform_world(link_name, qmap, cache)
            out[str(link_name)] = base_inv @ world_tf
        return out

    def _qmap_from_qpos_batch_torch(self, torch: Any, qpos_batch: Any) -> tuple[Any, Any, Any, int]:
        """Build clipped/mimic-resolved q-map for one batched q tensor."""

        q = qpos_batch
        if int(q.ndim) != 2:
            raise ValueError(f"Expected qpos_batch [N,D], got {tuple(q.shape)}")
        if int(q.shape[1]) < self.joint_count:
            raise ValueError(
                f"Expected qpos with at least {self.joint_count} joints, got {int(q.shape[1])}."
            )
        q = q[:, : self.joint_count]
        b = int(q.shape[0])
        q_eff = q.clone()

        lower, upper = self.joint_limits_lower_upper()
        lower_t = torch.as_tensor(lower, device=q.device, dtype=q.dtype)
        upper_t = torch.as_tensor(upper, device=q.device, dtype=q.dtype)
        q_eff = torch.maximum(q_eff, lower_t[None, :])
        q_eff = torch.minimum(q_eff, upper_t[None, :])

        for mimic in self.profile.mimic_joints:
            master_idx = self._joint_index_by_name[mimic.master_joint]
            follower_idx = self._joint_index_by_name[mimic.follower_joint]
            follower = q_eff[:, master_idx] * float(mimic.multiplier) + float(mimic.offset)
            follower = torch.maximum(follower, lower_t[follower_idx])
            follower = torch.minimum(follower, upper_t[follower_idx])
            q_eff[:, follower_idx] = follower

        qmap: dict[str, Any] = {}
        for idx, joint_name in enumerate(self.profile.joint_order):
            qmap[str(joint_name)] = q_eff[:, idx]
        return qmap, q_eff, lower_t, b

    def fingertips_from_qpos_batch_torch(self, torch: Any, qpos_batch: Any) -> Any:
        """Compute differentiable fingertip points [N,F,3] in retarget-origin frame."""

        qmap, _, _, b = self._qmap_from_qpos_batch_torch(torch, qpos_batch)
        q = qpos_batch

        origin_tf = self._chain_transform_torch(
            torch,
            qmap,
            self._origin_joint_chain,
            batch_size=b,
            device=q.device,
            dtype=q.dtype,
        )
        origin_inv = torch.linalg.inv(origin_tf)

        points: list[Any] = []
        for tip, chain in zip(
            self.profile.fingertip_links, self._fingertip_joint_chains, strict=True
        ):
            link_tf = self._chain_transform_torch(
                torch,
                qmap,
                chain,
                batch_size=b,
                device=q.device,
                dtype=q.dtype,
            )
            offset = torch.as_tensor(
                np.asarray(tip.center_offset, dtype=np.float64),
                device=q.device,
                dtype=q.dtype,
            ).reshape(1, 3, 1)
            world_pt = (link_tf[:, :3, :3] @ offset).squeeze(-1) + link_tf[:, :3, 3]
            origin_pt = (origin_inv[:, :3, :3] @ world_pt[:, :, None]).squeeze(-1) + origin_inv[:, :3, 3]
            points.append(origin_pt)

        if len(points) == 0:
            return torch.zeros((b, 0, 3), device=q.device, dtype=q.dtype)
        return torch.stack(points, dim=1)

    def link_transforms_from_qpos_batch_torch(
        self,
        torch: Any,
        qpos_batch: Any,
        *,
        base_frame: bool = True,
        link_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Compute differentiable link transforms for one q batch.

        Args:
            qpos_batch: ``[N,D]`` joint tensor.
            base_frame: when true, transforms are relative to the profile base link.
            link_names: optional subset of links; defaults to all known links.
        Returns:
            Mapping ``link_name -> [N,4,4]`` transforms.
        """

        qmap, _, _, b = self._qmap_from_qpos_batch_torch(torch, qpos_batch)
        q = qpos_batch
        selected = self._link_names if link_names is None else tuple(str(x) for x in link_names)

        base_tf = self._chain_transform_torch(
            torch,
            qmap,
            self._base_joint_chain,
            batch_size=b,
            device=q.device,
            dtype=q.dtype,
        )
        base_inv = (
            torch.linalg.inv(base_tf)
            if bool(base_frame)
            else torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(b, 1, 1)
        )

        out: dict[str, Any] = {}
        for link_name in selected:
            chain = self._link_joint_chains.get(str(link_name), None)
            if chain is None:
                raise KeyError(f"Unknown link name '{link_name}'.")
            link_tf = self._chain_transform_torch(
                torch,
                qmap,
                chain,
                batch_size=b,
                device=q.device,
                dtype=q.dtype,
            )
            out[str(link_name)] = base_inv @ link_tf
        return out

    def link_points_from_qpos_batch_torch(
        self,
        torch: Any,
        qpos_batch: Any,
        *,
        base_frame: bool = True,
        link_names: tuple[str, ...] | None = None,
    ) -> tuple[Any, tuple[str, ...]]:
        """Return batched link-origin points from differentiable transforms.

        Returns:
            points: ``[N,L,3]``.
            names: tuple of link names aligned with the point dimension.
        """

        selected = self._link_names if link_names is None else tuple(str(x) for x in link_names)
        tf_map = self.link_transforms_from_qpos_batch_torch(
            torch,
            qpos_batch,
            base_frame=bool(base_frame),
            link_names=selected,
        )
        pts = [tf_map[name][:, :3, 3] for name in selected]
        if len(pts) <= 0:
            q = qpos_batch
            return torch.zeros((int(q.shape[0]), 0, 3), device=q.device, dtype=q.dtype), tuple()
        return torch.stack(pts, dim=1), selected


def build_hand_kinematics(hand_profile: HandProfile | None) -> HandKinematics | None:
    """Build FK runtime helper for one hand profile when available."""

    if hand_profile is None:
        return None
    return HandKinematics(hand_profile)
