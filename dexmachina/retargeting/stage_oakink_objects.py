"""Stage OakInk-v2 rigid objects as DexMachina/Genesis assets.

DexMachina's ``ArticulatedObject`` is ARCTIC-shaped: it asserts the URDF has exactly
one movable (revolute/prismatic) joint (``dexmachina/envs/object.py``). OakInk objects
are rigid single bodies, so each staged URDF gets the real mesh on a ``base`` link plus a
**frozen dummy revolute joint** (limits ``[0, 0]``) to a massless ``tip_dummy`` child.
This satisfies the assert with no env change: with ``actuated=False`` and a demo
articulation of 0, the joint never moves and the articulation reward is ``exp(0)=1`` (a
no-op). See ``get_oakink_object_cfg`` (constructors) for the matching object config.

Source meshes: ManipTrans's ``coacd_object_preview/align_ds/<id>/scan.ply`` (meters).
Output: ``dexmachina/assets/oakink/<sanitized_id>/`` with ``visual.ply``, ``collision.obj``
(convex hull, or COACD parts with ``--coacd``), ``object.urdf``, and a top-level
``oakink_objects_manifest.json`` mapping raw OakInk ids to sanitized asset names.

Run from an env with trimesh (e.g. the learned_retargeter uv env):
    uv run python -m dexmachina.retargeting.stage_oakink_objects \
        --align-ds /home/fspinola/ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds \
        --out /nas/home2/f/fspinola/Documents/dexmachina/dexmachina/assets/oakink
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import trimesh

_LOG = logging.getLogger("stage_oakink_objects")

# Object masses (kg) measured in ManipTrans (main/dataset/oakink2_dataset_utils.py).
# Everything else is density-estimated and clamped to a plausible household range.
OAKINK_OBJ_MASS = {
    "O02@0015@00002": 0.101,
    "O02@0015@00001": 0.027,
    "C12001": 0.114,
    "O02@0030@00002": 0.0144,
    "O02@0033@00002": 0.014,
    "O02@0011@00003": 0.12,
    "O02@0206@00002": 0.163,
}
_DEFAULT_DENSITY = 300.0  # kg/m^3, light plastic/wood; only used when no measured mass
_MASS_CLAMP = (0.02, 1.0)  # kg
_OBJ_COLOR = (1.0, 0.4235, 0.0392, 1.0)


def sanitize_obj_id(obj_id: str) -> str:
    """OakInk ids contain '@'; map to a filesystem/URDF/clip-safe asset name.

    DexMachina clip strings split on '-', so '@' -> '_' and never introduce '-'.
    """
    return obj_id.replace("@", "_")


def _mass_inertia(hull: trimesh.Trimesh, obj_id: str) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, com[3], inertia[3,3]) about the COM, for the URDF inertial block.

    Uses the watertight convex hull for robust mass properties; scales the unit-density
    inertia to the target mass (inertia scales linearly with mass at fixed geometry).
    """
    com = np.asarray(hull.center_mass, dtype=np.float64)
    volume = float(hull.volume)
    if obj_id in OAKINK_OBJ_MASS:
        mass = float(OAKINK_OBJ_MASS[obj_id])
    else:
        mass = float(np.clip(_DEFAULT_DENSITY * max(volume, 1e-9), *_MASS_CLAMP))
    # trimesh.moment_inertia is about the COM at the mesh's current (unit) density,
    # i.e. scaled by the unit-density mass (== volume). Rescale to the target mass.
    unit_mass = max(volume, 1e-9)
    inertia = np.asarray(hull.moment_inertia, dtype=np.float64) * (mass / unit_mass)
    # Guard against degenerate/near-singular inertia for tiny/flat meshes.
    inertia = inertia + np.eye(3) * (mass * 1e-6)
    return mass, com, inertia


def _coacd_collision(mesh: trimesh.Trimesh, out_dir: Path) -> list[str]:
    """Write COACD convex parts as collision_<k>.obj; return their filenames.

    Optional refinement for concave objects (cups/bowls) where a single convex hull
    would fill the cavity and break grasps. Requires the ``coacd`` package.
    """
    import coacd  # local import: only needed with --coacd

    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cmesh)  # list of (verts, faces)
    names = []
    for k, (verts, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
        fname = f"collision_{k}.obj"
        part.export(out_dir / fname)
        names.append(fname)
    return names


def _write_urdf(
    urdf_path: Path,
    name: str,
    visual_fname: str,
    collision_fnames: list[str],
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
) -> None:
    cx, cy, cz = com
    ixx, ixy, ixz = inertia[0]
    _, iyy, iyz = inertia[1]
    _, _, izz = inertia[2]
    collisions = "\n".join(
        f"""    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{c}" scale="1 1 1"/></geometry>
    </collision>"""
        for c in collision_fnames
    )
    urdf = f"""<?xml version="1.0" ?>
<robot name="{name}">
  <material name="obj_color">
    <color rgba="{_OBJ_COLOR[0]} {_OBJ_COLOR[1]} {_OBJ_COLOR[2]} {_OBJ_COLOR[3]}"/>
  </material>
  <link name="base">
    <inertial>
      <origin rpy="0 0 0" xyz="{cx} {cy} {cz}"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{visual_fname}" scale="1 1 1"/></geometry>
      <material name="obj_color"/>
    </visual>
{collisions}
  </link>
  <!-- Massless child + frozen revolute joint: satisfies DexMachina's single-movable-joint
       assert while keeping the object rigid (limits [0,0], actuated=False, demo arti=0). -->
  <link name="tip_dummy">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1e-06"/>
      <inertia ixx="1e-09" ixy="0" ixz="0" iyy="1e-09" iyz="0" izz="1e-09"/>
    </inertial>
  </link>
  <joint name="rotation" type="revolute">
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <parent link="base"/>
    <child link="tip_dummy"/>
    <dynamics damping="0.01" friction="0.0"/>
    <limit effort="50" velocity="50" lower="0" upper="0"/>
  </joint>
</robot>
"""
    urdf_path.write_text(urdf)


def _find_mesh(obj_dir: Path) -> Path | None:
    """The single .ply/.obj mesh in an OakInk object dir (name varies: scan.ply, bowl.ply, ...).

    Mirrors ManipTrans's load_obj_map (oakink2_dataset_utils.py): exactly one mesh per dir.
    """
    candidates = sorted(p for p in obj_dir.iterdir() if p.suffix in (".ply", ".obj"))
    if len(candidates) != 1:
        _LOG.warning("skip %s: expected exactly one mesh, found %s", obj_dir.name, [p.name for p in candidates])
        return None
    return candidates[0]


def stage_object(obj_id: str, align_ds: Path, out_root: Path, use_coacd: bool) -> dict | None:
    obj_dir = align_ds / obj_id
    if not obj_dir.is_dir():
        _LOG.warning("skip %s: %s missing", obj_id, obj_dir)
        return None
    src_mesh = _find_mesh(obj_dir)
    if src_mesh is None:
        return None
    name = sanitize_obj_id(obj_id)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(src_mesh), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        _LOG.warning("skip %s: unreadable/empty mesh", obj_id)
        return None
    hull = mesh.convex_hull

    # Visual: keep the original scan; Collision: convex hull (default) or COACD parts.
    mesh.export(out_dir / "visual.ply")
    if use_coacd:
        try:
            collision_fnames = _coacd_collision(mesh, out_dir)
        except Exception as exc:  # noqa: BLE001 - fall back loudly, don't silently ship a bad hull
            _LOG.warning("COACD failed for %s (%s); falling back to convex hull", obj_id, exc)
            hull.export(out_dir / "collision.obj")
            collision_fnames = ["collision.obj"]
    else:
        hull.export(out_dir / "collision.obj")
        collision_fnames = ["collision.obj"]

    mass, com, inertia = _mass_inertia(hull, obj_id)
    _write_urdf(out_dir / "object.urdf", name, "visual.ply", collision_fnames, mass, com, inertia)
    _LOG.info("staged %s -> %s/ (mass=%.4f kg, %d collision part(s))",
              obj_id, name, mass, len(collision_fnames))
    return {
        "obj_id": obj_id,
        "name": name,
        "urdf": str((out_dir / "object.urdf").relative_to(out_root)),
        "mass": mass,
        "n_collision_parts": len(collision_fnames),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--align-ds", type=Path,
                   default=Path("/home/fspinola/ManipTrans/data/OakInk-v2/coacd_object_preview/align_ds"),
                   help="OakInk align_ds dir with <obj_id>/scan.ply")
    p.add_argument("--out", type=Path,
                   default=Path("/nas/home2/f/fspinola/Documents/dexmachina/dexmachina/assets/oakink"),
                   help="Output assets dir (dexmachina/assets/oakink)")
    p.add_argument("--ids", nargs="*", default=None,
                   help="Specific OakInk object ids to stage (default: all under --align-ds)")
    p.add_argument("--coacd", action="store_true",
                   help="Convex-decompose collision meshes (recommended for concave objects; needs coacd)")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args()
    ids = args.ids if args.ids else sorted(p.name for p in args.align_ds.iterdir() if p.is_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for obj_id in ids:
        entry = stage_object(obj_id, args.align_ds, args.out, args.coacd)
        if entry is not None:
            manifest.append(entry)
    (args.out / "oakink_objects_manifest.json").write_text(json.dumps(manifest, indent=2))
    _LOG.info("staged %d/%d objects; manifest at %s/oakink_objects_manifest.json",
              len(manifest), len(ids), args.out)


if __name__ == "__main__":
    main()
