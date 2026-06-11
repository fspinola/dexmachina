"""Hand-profile loading and validation for stage-1 robot geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class FingertipSpec:
    """One fingertip mapping entry from hand config."""

    name: str
    link: str
    joint_names: tuple[str, ...]
    center_offset: tuple[float, float, float]
    human_hand_id: int


@dataclass(frozen=True)
class HandPointSpec:
    """One generic hand-point mapping entry from hand config."""

    name: str
    link: str
    center_offset: tuple[float, float, float]
    human_hand_id: int


@dataclass(frozen=True)
class MimicJointSpec:
    """One mimic-joint rule from hand configuration."""

    follower_joint: str
    master_joint: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class HandProfile:
    """Validated hand profile used by stage-1 runtime mapping."""

    name: str
    config_path: Path
    urdf_path: Path
    base_link: str
    retarget_origin_link: str
    retarget_origin_offset: tuple[float, float, float]
    joint_order: tuple[str, ...]
    fingertip_links: tuple[FingertipSpec, ...]
    dexpilot_extra_points: tuple[HandPointSpec, ...]
    mimic_joints: tuple[MimicJointSpec, ...]
    joint_index_by_name: dict[str, int]
    fingertip_joint_indices: tuple[tuple[int, ...], ...]
    default_distal_joint_indices: tuple[int, ...]
    default_prev_joint_indices: tuple[int, ...]

    @property
    def joint_count(self) -> int:
        """Return number of controlled robot joints."""

        return int(len(self.joint_order))

    @property
    def fingertip_human_ids(self) -> tuple[int, ...]:
        """Return configured human keypoint ids for fingertips in method order."""

        return tuple(int(x.human_hand_id) for x in self.fingertip_links)


def _default_project_root() -> Path:
    """Return repository root for resolving config and asset references."""

    return Path(__file__).resolve().parents[2]


def _resolve_hand_config_path(
    *,
    hand_name: str | None,
    config_path: str | None,
    project_root: Path,
) -> Path | None:
    """Resolve hand config path from explicit path or hand name."""

    if config_path is not None and str(config_path).strip():
        cfg_path = Path(str(config_path)).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (project_root / cfg_path).resolve()
        return cfg_path

    if hand_name is None or not str(hand_name).strip():
        return None

    return (project_root / "configs" / "hands" / f"{hand_name}.json").resolve()


def _resolve_urdf_path(raw_path: str, *, config_path: Path, project_root: Path) -> Path:
    """Resolve URDF path from config value."""

    urdf = Path(str(raw_path)).expanduser()
    if urdf.is_absolute():
        return urdf.resolve()

    from_project = (project_root / urdf).resolve()
    if from_project.exists():
        return from_project
    return (config_path.parent / urdf).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    """Load hand config JSON and validate root type."""

    import json

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Hand config root must be a JSON object: {path}")
    return dict(payload)


def _as_offset3(value: Any, field_name: str) -> tuple[float, float, float]:
    """Validate and cast offset vectors to 3 floats."""

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{field_name} must have exactly 3 values, got shape {arr.shape}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _parse_urdf_symbols(path: Path) -> tuple[set[str], set[str]]:
    """Parse URDF and return link and joint name sets."""

    tree = ET.parse(path)
    root = tree.getroot()
    if str(root.tag).strip().lower() != "robot":
        raise ValueError(f"URDF root tag must be <robot>, got <{root.tag}> in {path}.")
    links = {
        str(elem.attrib["name"])
        for elem in root.findall("link")
        if isinstance(elem.attrib, dict) and "name" in elem.attrib
    }
    joints = {
        str(elem.attrib["name"])
        for elem in root.findall("joint")
        if isinstance(elem.attrib, dict) and "name" in elem.attrib
    }
    return links, joints


def _validate_with_urdf(
    *,
    profile_name: str,
    urdf_path: Path,
    base_link: str,
    origin_link: str,
    joint_order: tuple[str, ...],
    fingertips: tuple[FingertipSpec, ...],
    extra_points: tuple[HandPointSpec, ...] = tuple(),
) -> None:
    """Validate config references against URDF symbols."""

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found for '{profile_name}': {urdf_path}")
    links, joints = _parse_urdf_symbols(urdf_path)

    required_links = {base_link, origin_link}
    required_links.update(str(t.link) for t in fingertips)
    required_links.update(str(p.link) for p in extra_points)
    missing_links = sorted(x for x in required_links if x not in links)
    if missing_links:
        raise ValueError(
            f"Hand config '{profile_name}' references links missing in URDF {urdf_path}: "
            f"{missing_links}"
        )

    required_joints = set(joint_order)
    for tip in fingertips:
        required_joints.update(tip.joint_names)
    missing_joints = sorted(x for x in required_joints if x not in joints)
    if missing_joints:
        raise ValueError(
            f"Hand config '{profile_name}' references joints missing in URDF {urdf_path}: "
            f"{missing_joints}"
        )


def load_hand_profile(
    *,
    hand_name: str | None,
    config_path: str | None = None,
    strict_urdf: bool = True,
    project_root: str | Path | None = None,
) -> HandProfile | None:
    """Load and validate hand profile from hand JSON config.

    Returns:
        HandProfile when a hand config is requested, otherwise ``None``.
    """

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _default_project_root()
    )
    cfg_path = _resolve_hand_config_path(
        hand_name=(None if hand_name is None else str(hand_name)),
        config_path=config_path,
        project_root=root,
    )
    if cfg_path is None:
        return None
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hand config not found: {cfg_path}")

    payload = _load_json(cfg_path)
    name = str(payload.get("name", hand_name or cfg_path.stem))
    urdf_path = _resolve_urdf_path(
        str(payload.get("urdf_path", "")),
        config_path=cfg_path,
        project_root=root,
    )
    base_link = str(payload.get("base_link", "")).strip()
    origin_link = str(payload.get("retarget_origin_link", base_link)).strip()
    if not base_link:
        raise ValueError(f"Hand config '{name}' missing required field 'base_link'.")
    if not origin_link:
        raise ValueError(f"Hand config '{name}' missing required field 'retarget_origin_link'.")
    origin_offset = _as_offset3(
        payload.get("retarget_origin_offset", [0.0, 0.0, 0.0]),
        "retarget_origin_offset",
    )

    joint_order_raw = payload.get("joint_order", [])
    if not isinstance(joint_order_raw, list) or len(joint_order_raw) == 0:
        raise ValueError(f"Hand config '{name}' field 'joint_order' must be a non-empty list.")
    joint_order = tuple(str(x) for x in joint_order_raw)
    if len(set(joint_order)) != len(joint_order):
        raise ValueError(f"Hand config '{name}' contains duplicated joint names in joint_order.")
    joint_index_by_name = {joint: idx for idx, joint in enumerate(joint_order)}

    fingertips_raw = payload.get("fingertip_link", [])
    if not isinstance(fingertips_raw, list) or len(fingertips_raw) == 0:
        raise ValueError(f"Hand config '{name}' field 'fingertip_link' must be a non-empty list.")

    fingertips: list[FingertipSpec] = []
    extra_points: list[HandPointSpec] = []
    fingertip_joint_indices: list[tuple[int, ...]] = []
    distal_indices: list[int] = []
    prev_indices: list[int] = []
    human_ids: set[int] = set()

    for idx, entry in enumerate(fingertips_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Hand config '{name}' fingertip[{idx}] must be a JSON object.")
        tip_name = str(entry.get("name", f"tip_{idx}"))
        tip_link = str(entry.get("link", "")).strip()
        if not tip_link:
            raise ValueError(f"Hand config '{name}' fingertip '{tip_name}' missing field 'link'.")
        joint_names_raw = entry.get("joint", [])
        if not isinstance(joint_names_raw, list) or len(joint_names_raw) == 0:
            raise ValueError(
                f"Hand config '{name}' fingertip '{tip_name}' field 'joint' must be non-empty list."
            )
        tip_joint_names = tuple(str(x) for x in joint_names_raw)
        try:
            tip_joint_idx = tuple(int(joint_index_by_name[x]) for x in tip_joint_names)
        except KeyError as missing:
            raise ValueError(
                f"Hand config '{name}' fingertip '{tip_name}' references unknown joint: {missing}"
            ) from missing

        human_hand_id = int(entry.get("human_hand_id"))
        if human_hand_id in human_ids:
            raise ValueError(
                f"Hand config '{name}' has duplicated human_hand_id={human_hand_id} "
                "across fingertips."
            )
        human_ids.add(human_hand_id)

        center_offset = _as_offset3(entry.get("center_offset", [0.0, 0.0, 0.0]), "center_offset")
        tip_spec = FingertipSpec(
            name=tip_name,
            link=tip_link,
            joint_names=tip_joint_names,
            center_offset=center_offset,
            human_hand_id=human_hand_id,
        )
        fingertips.append(tip_spec)
        fingertip_joint_indices.append(tip_joint_idx)

        if len(tip_joint_idx) >= 2:
            prev_indices.append(int(tip_joint_idx[-2]))
            distal_indices.append(int(tip_joint_idx[-1]))

    mimic_raw = payload.get("mimic_joints", [])
    if mimic_raw is None:
        mimic_raw = []
    if not isinstance(mimic_raw, list):
        raise ValueError(f"Hand config '{name}' field 'mimic_joints' must be a list.")
    mimic_joints: list[MimicJointSpec] = []
    for idx, entry in enumerate(mimic_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Hand config '{name}' mimic_joints[{idx}] must be a JSON object.")
        follower = str(entry.get("joint", "")).strip()
        master = str(entry.get("mimic", "")).strip()
        if follower not in joint_index_by_name:
            raise ValueError(
                f"Hand config '{name}' mimic follower joint '{follower}' not in joint_order."
            )
        if master not in joint_index_by_name:
            raise ValueError(f"Hand config '{name}' mimic master joint '{master}' not in joint_order.")
        mimic_joints.append(
            MimicJointSpec(
                follower_joint=follower,
                master_joint=master,
                multiplier=float(entry.get("multiplier", 1.0)),
                offset=float(entry.get("offset", 0.0)),
            )
        )

    extra_raw = payload.get("dexpilot_extra_point", [])
    if extra_raw is None:
        extra_raw = []
    if not isinstance(extra_raw, list):
        raise ValueError(f"Hand config '{name}' field 'dexpilot_extra_point' must be a list.")
    for idx, entry in enumerate(extra_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Hand config '{name}' dexpilot_extra_point[{idx}] must be a JSON object."
            )
        point_name = str(entry.get("name", f"extra_{idx}"))
        point_link = str(entry.get("link", "")).strip()
        if not point_link:
            raise ValueError(
                f"Hand config '{name}' dexpilot_extra_point '{point_name}' missing field 'link'."
            )
        human_hand_id = int(entry.get("human_hand_id"))
        if human_hand_id in human_ids:
            raise ValueError(
                f"Hand config '{name}' has duplicated human_hand_id={human_hand_id} "
                "across fingertip_link and dexpilot_extra_point."
            )
        human_ids.add(human_hand_id)
        center_offset = _as_offset3(
            entry.get("center_offset", [0.0, 0.0, 0.0]),
            "center_offset",
        )
        extra_points.append(
            HandPointSpec(
                name=point_name,
                link=point_link,
                center_offset=center_offset,
                human_hand_id=human_hand_id,
            )
        )

    fingertips_tuple = tuple(fingertips)
    extra_points_tuple = tuple(extra_points)
    if bool(strict_urdf):
        _validate_with_urdf(
            profile_name=name,
            urdf_path=urdf_path,
            base_link=base_link,
            origin_link=origin_link,
            joint_order=joint_order,
            fingertips=fingertips_tuple,
            extra_points=extra_points_tuple,
        )

    return HandProfile(
        name=name,
        config_path=cfg_path,
        urdf_path=urdf_path,
        base_link=base_link,
        retarget_origin_link=origin_link,
        retarget_origin_offset=origin_offset,
        joint_order=joint_order,
        fingertip_links=fingertips_tuple,
        dexpilot_extra_points=extra_points_tuple,
        mimic_joints=tuple(mimic_joints),
        joint_index_by_name=joint_index_by_name,
        fingertip_joint_indices=tuple(fingertip_joint_indices),
        default_distal_joint_indices=tuple(distal_indices),
        default_prev_joint_indices=tuple(prev_indices),
    )
