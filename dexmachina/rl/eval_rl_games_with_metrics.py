import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
from datetime import datetime 
import csv

from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv

  
from collections import defaultdict
import shutil
import moviepy
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from dexmachina.rl.part_add_metrics import (
    PartAddConfig,
    add_auc_from_episode_part_add_mean,
    compute_part_add_step,
    get_part_points_local,
)


def _to_jsonable(obj):
    """Best-effort conversion of common numeric types to JSON-serializable Python types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        if obj.shape == ():
            return _to_jsonable(obj.item())
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return _to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, (dict,)):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_episode_metrics_sidecar(eval_data: dict, out_npy_path: str, eps: int, metric_key: str):
    """Save only *computed* episode-level metrics next to the .npy file.

    Writes two files:
      - metrics_ep{eps}.json
      - metrics_ep{eps}.csv (single-row)
    """
    out_dir = os.path.dirname(out_npy_path)

    # Keep this small: only aggregated episode-level fields we compute in this script.
    metrics = {
        "episode": int(eps),
        "metric_key": str(metric_key),
        "add_auc": _to_jsonable(eval_data.get("add_auc")),
        "success_at_thresh": _to_jsonable(eval_data.get("success_at_thresh")),
        "add_invalid_count": _to_jsonable(eval_data.get("_add_invalid_count", np.array([0], dtype=np.int32))),
    }
    # Optional extras: keep details if present (still JSON-friendly).
    if "add_auc_details" in eval_data:
        metrics["add_auc_details"] = _to_jsonable(eval_data["add_auc_details"])

    json_path = os.path.join(out_dir, f"metrics_ep{eps}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, indent=2, sort_keys=True)

    csv_path = os.path.join(out_dir, f"metrics_ep{eps}.csv")
    # CSV: flatten a few scalar fields (single row). Keep the detailed dict in JSON only.
    csv_row = {
        "episode": metrics["episode"],
        "metric_key": metrics["metric_key"],
        "thresh": metrics["add_auc_details"].get("threshold_m", None),
        "add_auc": float(np.asarray(eval_data.get("add_auc", np.array([np.nan]))).reshape(-1)[0]),
        "success_at_thresh": float(np.asarray(eval_data.get("success_at_thresh", np.array([np.nan]))).reshape(-1)[0]),
        "add_invalid_count": int(np.asarray(eval_data.get("_add_invalid_count", np.array([0]))).reshape(-1)[0]),
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)
 

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()


def estimate_object_diameter_m(obj, *, num_points: int = 4000, seed: int = 0) -> float:
    """Estimate object diameter (max distance between sampled surface points) in meters.

    This is a cheap proxy using random vertices sampled from each part mesh.
    Assumes `obj.sample_mesh_vertices(N, part=..., seed=...)` returns points in a
    consistent object frame.
    """
    pts_all = []
    # Prefer ADD parts if present; otherwise fall back to whatever is available.
    parts = ("bottom", "top")
    n_per = max(1, num_points // max(1, len(parts)))
    for i, part in enumerate(parts):
        try:
            v = obj.sample_mesh_vertices(n_per, part=part, seed=int(seed + 17 * i))
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 2 and v.shape[1] == 3 and v.size > 0:
            pts_all.append(v)
    if not pts_all:
        return float("nan")
    pts = np.concatenate(pts_all, axis=0)
    # bounding-box diagonal as a stable diameter proxy
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(np.linalg.norm(maxs - mins))

def eval_one_episode(env, agent, obj_state_tensor, print_rew=False, record_video=False, show_reference=False):
    obs = env.reset() 
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    joint_target_left = left_hand.residual_qpos
    joint_target_right = right_hand.residual_qpos
    
    eval_data = defaultdict(list)

    # IMPORTANT: Metric computation must not perturb the policy rollout.
    # Always compute metrics against env 0 (the actual rollout), and only use the last env
    # for visualization when --show_reference/-ref is on.
    eval_env_idx = 0

    add_cfg = PartAddConfig()
    part_points_local = get_part_points_local(obj, add_cfg, device=device)
    if show_reference:
        # store as a list so the final np.stack works consistently
        eval_data["_debug_ref_env_idx"].append(np.int32(eval_env_idx))
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
            # Guard: env_step can reach the trajectory length at the terminal step.
            # Clamp so we always use a valid demo frame.
            env_step = int(env_step)
            env_step = min(env_step, int(obj_state_tensor.shape[0]) - 1)

            demo_state = obj_state_tensor[env_step]
                    
            if show_reference: # visualize the demo traj and set zero action
                obj.set_object_state(
                    root_pos=demo_state[:3][None],
                    root_quat=demo_state[3:7][None],
                    joint_qpos=demo_state[7:][None],
                    env_idxs=torch.tensor([num_envs-1], dtype=torch.int32, device=device),
                )
                actions[-1, :] = -1.0 
                for robot, joints in zip([left_hand, right_hand], [joint_target_left, joint_target_right]):
                    robot.set_joint_position(
                        joint_targets=joints[env_step][None],
                        env_idxs=[num_envs-1],
                    ) 

                # Strict GT sanity check: right after forcing to demo_state, policy and demo should match.
                # NON-MUTATING: compute demo poses inside compute_part_add_step (do not use cached FK).
                add_out_pre = compute_part_add_step(
                    obj=obj,
                    demo_state=demo_state,
                    part_points_local=part_points_local,
                    cfg=add_cfg,
                    env_idx=eval_env_idx,
                )
                eval_data["part_add_mean_pre"].append(np.array(add_out_pre["part_add_mean"], dtype=np.float32))
                if "part_add_bottom" in add_out_pre:
                    eval_data["part_add_bottom_pre"].append(np.array(add_out_pre["part_add_bottom"], dtype=np.float32))
                if "part_add_top" in add_out_pre:
                    eval_data["part_add_top_pre"].append(np.array(add_out_pre["part_add_top"], dtype=np.float32))
            obs, rew, dones, infos = env.step(actions) 
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            # print(f"Step {env_step}: Obj pos: {obj_pos.cpu().numpy()}")
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())

            # Part-wise ADD (compute per-part ADD and mean across parts)
            # NOTE: do NOT pass demo_part_pos/demo_part_quat so the metric stays non-mutating.
            add_out = compute_part_add_step(
                obj=obj,
                demo_state=demo_state,
                part_points_local=part_points_local,
                cfg=add_cfg,
                env_idx=eval_env_idx,
            )

            # If ADD becomes non-finite, treat it as a failure frame (>3cm) for logging/AUC.
            # (We still keep the NaN debug prints so we can investigate root cause.)
            if not np.isfinite(add_out.get("part_add_mean", np.nan)):
                add_out["part_add_mean"] = float(add_cfg.auc_threshold_m + 1.0)
                for part in add_cfg.part_names:
                    k = f"part_add_{part}"
                    if k in add_out and (not np.isfinite(add_out[k])):
                        add_out[k] = float(add_cfg.auc_threshold_m + 1.0)

            for k, v in add_out.items():
                eval_data[k].append(np.array(v, dtype=np.float32))

            rew_dict = uenv.rew_dict
            for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                eval_data[key].append(rew_dict[key].cpu().numpy())
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0 
        
        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes 
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    frames = uenv.get_recorded_frames()
    return frames, eval_data


def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint

    # ADD-AUC settings (paper-style part-wise ADD then AUC)
    parser.add_argument('--add_num_points', type=int, default=300, help='Sampled mesh vertices per part')
    # If provided, overrides the adaptive threshold below.
    parser.add_argument('--add_auc_thresh', type=float, default=None, help='AUC threshold in meters (overrides adaptive threshold)')
    parser.add_argument('--add_auc_thresh_frac_diam', type=float, default=0.10, help='AUC threshold as fraction of estimated object diameter (default 0.10) similar to https://arxiv.org/pdf/1711.00199')
    parser.add_argument('--add_auc_step', type=float, default=0.001, help='AUC integration step in meters')
    
    args = parser.parse_args()

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    os.makedirs(ckpt_data_folder, exist_ok=True) 

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    # load to pkl
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        # env_kwargs['camera_res'] = (2048, 2048)
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        ) 
    for name, cfg in env_kwargs['object_cfgs'].items():
        print("Setting eval time obj gains to 0.0")
        cfg['actuated'] = False

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    print("Removing any curriculum config during eval")
    env_kwargs.pop("curriculum_cfg") 
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')
    env = BaseEnv(
         **env_kwargs
    )
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data)

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped 
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint) 
    agent.restore(resume_path)
    agent.reset()

    # Adaptive ADD-AUC threshold: 10% of object diameter by default.
    uenv = env.unwrapped
    try:
        obj0 = uenv.objects[uenv.object_names[0]]
        obj_diam_m = estimate_object_diameter_m(obj0)
    except Exception:
        obj_diam_m = float("nan")
    if args.add_auc_thresh is not None:
        add_auc_thresh_m = float(args.add_auc_thresh)
        add_auc_thresh_src = "--add_auc_thresh"
    else:
        if np.isfinite(obj_diam_m) and obj_diam_m > 0:
            add_auc_thresh_m = float(args.add_auc_thresh_frac_diam) * float(obj_diam_m)
            add_auc_thresh_src = f"{args.add_auc_thresh_frac_diam:.3f}*diam"
        else:
            # Fallback to old/default behavior if diameter can't be estimated.
            add_auc_thresh_m = 0.03
            add_auc_thresh_src = "fallback_0.03m"
    print(f"ADD-AUC threshold: {add_auc_thresh_m:.6f} m (source={add_auc_thresh_src}, est_diam={obj_diam_m})")

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference
            )

        # Episode-level ADD-AUC computed from per-frame mean-over-parts ADD.
        # If -ref is enabled, we default to the strict GT sanity-check trajectory (pre).
        add_cfg = PartAddConfig(
            num_points_per_part=args.add_num_points,
            auc_threshold_m=add_auc_thresh_m,
            auc_step_m=args.add_auc_step,
        )
        metric_key = "part_add_mean"
        if args.show_reference:
            metric_key = "part_add_mean_pre"
        # Robustify against rare non-finite frames (often at the terminal step).
        # For ADD-AUC, treating non-finite as a large error is equivalent to a failure frame.
        arr = np.asarray(eval_data[metric_key].reshape(-1), dtype=np.float64)
        invalid = ~np.isfinite(arr)
        if invalid.any():
            arr = arr.copy()
            arr[invalid] = float(add_cfg.auc_threshold_m + 1.0)
            eval_data["_add_invalid_count"] = np.array([int(invalid.sum())], dtype=np.int32)
        # Additional paper-adjacent summary: per-frame success rate at the AUC threshold.
        # (Useful when AUC is low: tells whether it's a late failure vs always-bad tracking.)
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            success_at_thresh = float((arr[finite_mask] < add_cfg.auc_threshold_m).mean())
        else:
            success_at_thresh = 0.0
        eval_data["success_at_thresh"] = np.array([success_at_thresh], dtype=np.float32)

        auc_out = add_auc_from_episode_part_add_mean(arr, cfg=add_cfg)
        eval_data["add_auc"] = np.array([auc_out["add_auc"]], dtype=np.float32)
        eval_data["add_auc_details"] = auc_out

        # Backwards-compat fields (so older notebooks/scripts keep working)
        eval_data["add_auc3"] = eval_data["add_auc"]
        eval_data["add_auc3_details"] = eval_data["add_auc_details"]
        eval_data["success_at_3cm"] = eval_data["success_at_thresh"]

        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        save_episode_metrics_sidecar(eval_data, ckpt_eval_fname, eps=eps, metric_key=metric_key)
        mode_str = f" ({metric_key})" if args.show_reference else ""
        extra = ""
        extra += f" | succ@thresh={success_at_thresh:.3f}"
        print(f"Saved eval data to {ckpt_eval_fname} | ADD-AUC{mode_str}={auc_out['add_auc']:.3f}{extra}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item() 
        if args.record_video: 
            # save video with moviepy
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip(frames, fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")
    
    print("Done evaluating")


if __name__ == '__main__':
    main()
