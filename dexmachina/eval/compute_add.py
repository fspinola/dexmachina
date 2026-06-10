import argparse
import json
import os
import re
from collections import defaultdict
from os.path import join
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch
import genesis as gs

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.eval.utils import (
    ensure_dir,
    extract_clip,
    find_config_path,
    infer_object_name_from_clip,
    list_eval_files,
    load_config,
)


def create_scene(obj_name: str, device: torch.device, num_envs: int):
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=2, gravity=(0, 0, 0)),
        vis_options=gs.options.VisOptions(
            n_rendered_envs=1,
            show_world_frame=False,
            visualize_contact=False,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=1 / 60,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.5, 1.8),
            camera_lookat=(0.0, -0.15, 1.0),
            camera_fov=30,
        ),
        use_visualizer=False,
        show_viewer=False,
    )
    scene = gs.Scene(**scene_cfg)
    obj_cfg = get_arctic_object_cfg(name=obj_name, convexify=True)
    obj = ArticulatedObject(
        obj_cfg, device=device, scene=scene, num_envs=num_envs, disable_collision=True
    )
    scene.build(n_envs=num_envs)
    obj.post_scene_build_setup()
    return scene, obj


def load_part_verts(
    obj: ArticulatedObject,
    obj_name: str,
    device: torch.device,
    num_samples: int = 500,
    cache_dir: str = "dexmachina/eval/sub_verts",
    output_cache: bool = False,
) -> Dict[str, torch.Tensor]:
    part_verts = {}
    for part in ["top", "bottom"]:
        if output_cache:
            vdir = join(cache_dir, obj_name)
            ensure_dir(vdir)
            vname = join(vdir, f"{part}_{num_samples}.npy")
            if not os.path.exists(vname):
                verts = obj.sample_mesh_vertices(part=part, num_samples=num_samples)
                np.save(vname, verts.cpu().numpy())
            else:
                verts = np.load(vname)
        else:
            verts = obj.sample_mesh_vertices(part=part, num_samples=num_samples)
        part_verts[part] = torch.tensor(
            verts[None], dtype=torch.float32, device=device
        )
    return part_verts


def get_all_add(
    obj: ArticulatedObject,
    part_verts: Dict[str, torch.Tensor],
    demo_states: torch.Tensor,
    obj_states: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    demo_states: (num_frames, 8)
    obj_states: (num_frames, num_eval_envs, 8)
    """
    obj.reset()
    obj.set_object_state(
        root_pos=demo_states[:, :3],
        root_quat=demo_states[:, 3:7],
        joint_qpos=demo_states[:, 7:],
    )
    demo_verts = {
        part: obj.transform_part_vertices(part_verts[part], part)
        for part in part_verts.keys()
    }
    dists = defaultdict(list)
    n_envs = obj_states.shape[1]
    for i in range(n_envs):
        states = obj_states[:, i, :]
        obj.set_object_state(
            root_pos=states[:, :3],
            root_quat=states[:, 3:7],
            joint_qpos=states[:, 7:],
        )
        obj_verts = {
            part: obj.transform_part_vertices(part_verts[part], part)
            for part in part_verts.keys()
        }
        for part in part_verts.keys():
            dist = torch.norm(demo_verts[part] - obj_verts[part], dim=-1)
            dist = torch.mean(dist, dim=-1)
            dists[part].append(dist.cpu().numpy())
    return {part: np.array(dists[part]) for part in dists.keys()}


def compute_auc(mean_add: np.ndarray, thresholds):
    accuracies = []
    for thres in thresholds:
        acc = np.mean(mean_add < thres)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    x_values = np.linspace(0, 1, len(thresholds))
    return float(np.trapz(accuracies, x=x_values))


def compute_add_stats(add_data: Dict[str, np.ndarray], thresholds) -> Dict[str, Dict]:
    mean_add = {part: float(np.mean(add_data[part])) for part in add_data.keys()}
    std_add = {part: float(np.std(add_data[part])) for part in add_data.keys()}
    auc = {part: compute_auc(add_data[part], thresholds) for part in add_data.keys()}
    return dict(
        mean_add={
            **mean_add,
            "overall": float(np.mean(list(mean_add.values()))),
        },
        std_add={
            **std_add,
            "overall": float(np.mean(list(std_add.values()))),
        },
        auc={
            **auc,
            "overall": float(np.mean(list(auc.values()))),
        },
        thresholds=list(thresholds),
    )


def list_arctic_objects() -> Set[str]:
    """Names of the ARCTIC objects available on disk (one sub-dir per object)."""
    arctic_dir = str(get_asset_path("arctic"))
    if not os.path.isdir(arctic_dir):
        return set()
    return {
        name
        for name in os.listdir(arctic_dir)
        if os.path.isdir(os.path.join(arctic_dir, name))
    }


def infer_object_name_from_path(eval_path: str, valid_objects: Set[str]) -> Optional[str]:
    """Recover the object name from the run directory when no config.yaml exists.

    Run dirs encode the clip, e.g.
        .../allegro-20012026_box30-230-s01-u01_B12000_.../..._eval/eval_ep0.npy
    so the leading letters of a clip token give the object ("box"). Every token
    is validated against the known ARCTIC objects, so experiment tags such as
    "graphnc", "hybrid", "allegro" or "inspire" are skipped automatically and
    the same logic works for the graph, allegro and inspire runs alike.
    """
    path = Path(eval_path).resolve()
    for parent in list(path.parents):
        for token in re.split(r"[_\-]", parent.name):
            match = re.match(r"([a-zA-Z]+)", token)
            if match and match.group(1) in valid_objects:
                return match.group(1)
    return None


def infer_object_name(eval_path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    cfg_path = find_config_path(eval_path)
    if cfg_path:
        cfg = load_config(cfg_path)
        clip = extract_clip(cfg)
        obj_name = infer_object_name_from_clip(clip)
        if obj_name:
            return obj_name
    # No config.yaml (or no clip inside it): fall back to the run directory name,
    # which encodes the clip for the allegro/inspire (and graph) eval runs.
    obj_name = infer_object_name_from_path(eval_path, list_arctic_objects())
    if obj_name:
        return obj_name
    raise ValueError(
        f"Could not find config.yaml or infer object name from path for {eval_path}"
    )


def compute_for_eval(
    eval_path: str,
    obj_name: Optional[str],
    out_name: str,
    stats_name: str,
    overwrite: bool,
    output_cache: bool,
    cache_dir: str,
):
    eval_data = np.load(eval_path, allow_pickle=True).item()
    demo_states = eval_data["demo_state"]
    obj_states = eval_data["obj_state"]
    if obj_states.ndim == 2:
        obj_states = obj_states[:, None, :]
    if demo_states.ndim == 3 and demo_states.shape[1] == 1:
        demo_states = demo_states[:, 0, :]

    obj_name = infer_object_name(eval_path, obj_name)
    num_frames = demo_states.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene, obj = create_scene(obj_name, device, num_envs=num_frames)
    part_verts = load_part_verts(
        obj, obj_name, device, output_cache=output_cache, cache_dir=cache_dir
    )
    demo_states_t = torch.tensor(demo_states, dtype=torch.float32, device=device)
    obj_states_t = torch.tensor(obj_states, dtype=torch.float32, device=device)

    add_data = get_all_add(obj, part_verts, demo_states_t, obj_states_t)
    out_dir = os.path.dirname(eval_path)
    out_path = join(out_dir, out_name)
    stats_path = join(out_dir, stats_name)

    if os.path.exists(out_path) and not overwrite:
        print(f"Using existing {out_path}")
        add_data = np.load(out_path, allow_p