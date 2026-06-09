import argparse
import json
import os
from collections import defaultdict
from os.path import join
from typing import Dict, Optional

import numpy as np
import torch
import genesis as gs

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


def infer_object_name(eval_path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    cfg_path = find_config_path(eval_path)
    if not cfg_path:
        raise ValueError(f"Could not find config.yaml for {eval_path}")
    cfg = load_config(cfg_path)
    clip = extract_clip(cfg)
    obj_name = infer_object_name_from_clip(clip)
    if not obj_name:
        raise ValueError(f"Could not infer object name from clip in {cfg_path}")
    return obj_name


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
        add_data = np.load(out_path, allow_pickle=True).item()
    else:
        np.save(out_path, add_data)
        print(f"Wrote {out_path}")

    if not os.path.exists(stats_path) or overwrite:
        thresholds = list(np.arange(0.01, 0.1, 0.01))
        stats = compute_add_stats(add_data, thresholds)
        stats["eval_path"] = eval_path
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote eval stats to: \n{stats_path}")
    
    del scene
    del obj 
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Eval file or root directory")
    parser.add_argument("--pattern", default="**/eval_ep*.npy")
    parser.add_argument("--obj_name", default=None)
    parser.add_argument("--out_name", default="add.npy")
    parser.add_argument("--stats_name", default="add_stats.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_cache", action="store_true")
    parser.add_argument("--cache_dir", default="dexmachina/eval/sub_verts")
    args = parser.parse_args()

    eval_files = list_eval_files(args.input, args.pattern)
    if not eval_files:
        raise ValueError(f"No eval files found for {args.input}")
    print("Found eval files:")
    print("\n\n".join(eval_files))

    gs.init(backend=gs.gpu, logging_level="warning") # NOTE: init only once
    for eval_path in sorted(eval_files):
        compute_for_eval(
            eval_path=eval_path,
            obj_name=args.obj_name,
            out_name=args.out_name,
            stats_name=args.stats_name,
            overwrite=args.overwrite,
            output_cache=args.output_cache,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
