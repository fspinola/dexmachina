import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import sys
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
from datetime import datetime 

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
 

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

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
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
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
            obs, rew, dones, infos = env.step(actions) 
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            # print(f"Step {env_step}: Obj pos: {obj_pos.cpu().numpy()}")
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())

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
    parser.add_argument('--task_env', type=str, default=None,
        help="Run dir / params/env.pkl / checkpoint whose saved env.pkl defines the TASK "
             "(demo reference, object, episode length). Policy weights STILL come from "
             "--checkpoint. Use for cross-clip transfer eval, e.g. run the -40 task with the "
             "-30 policy. Requires the two runs share obs/action config (same object, hand, "
             "action_mode, obs flags) so the -30 weights load into the -40 env.")
    parser.add_argument('--out_tag', type=str, default=None,
        help="Suffix for the eval output folder (default: a clip token parsed from --task_env, "
             "e.g. 'on_mixer40-340'). Keeps the cross-eval separate from the normal eval.")
    parser.add_argument('--cam_pos', type=float, nargs=3, default=None,
        help="Override record-video camera position (x y z). Default is ARCTIC-framed; OakInk "
             "scenes sit near z~1.0 so pass e.g. --cam_pos 0.5 -0.9 1.4.")
    parser.add_argument('--cam_lookat', type=float, nargs=3, default=None,
        help="Override record-video camera lookat (x y z). OakInk e.g. --cam_lookat -0.1 -0.1 1.0.")

    args = parser.parse_args()
    # Only override the saved action_mode when the user EXPLICITLY passes -am/--action_mode.
    # (The shared parser defaults action_mode to 'residual', which must not clobber the value
    #  baked into the run's env.pkl.) Use e.g. `-am kinematic` for an open-loop reference replay.
    override_action_mode = any(a in ('-am', '--action_mode') for a in sys.argv)

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")

    # --task_env: take the env kwargs (demo reference, object, episode length = the TASK) from a
    # DIFFERENT run's env.pkl; policy weights still come from --checkpoint. Accepts a run dir, a
    # params/env.pkl path, or any checkpoint inside the run (nn/*.pth).
    task_token = None
    if args.task_env is not None:
        te = args.task_env.rstrip("/")
        if te.endswith("env.pkl"):
            task_run_dir = os.path.dirname(os.path.dirname(te))
        elif te.endswith(".pth"):
            task_run_dir = os.path.dirname(os.path.dirname(te))
        else:
            task_run_dir = te
        saved_cfg_fname = os.path.join(task_run_dir, "params", "env.pkl")
        m = re.search(r'_([a-zA-Z]+[0-9]+-[0-9]+)', os.path.basename(task_run_dir))
        task_token = m.group(1) if m else "xtask"
        print(f"[eval] TASK env from {saved_cfg_fname}; policy weights from {args.checkpoint}")

    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_eval"))
    if override_action_mode:
        ckpt_data_folder += f"_{args.action_mode}"  # e.g. *_eval_kinematic; keep policy eval intact
    out_tag = args.out_tag if args.out_tag is not None else (f"on_{task_token}" if task_token else None)
    if out_tag:
        ckpt_data_folder += f"_{out_tag}"  # keep cross-task evals separate from the run's own eval
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

    if override_action_mode:
        for side, cfg in env_kwargs['robot_cfgs'].items():
            if isinstance(cfg, dict) and 'action_mode' in cfg:
                cfg['action_mode'] = args.action_mode
        print(f"[eval] action_mode -> '{args.action_mode}' for robots {list(env_kwargs['robot_cfgs'])}"
              f"  (kinematic = open-loop reference replay, policy ignored); results -> {ckpt_data_folder}")

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
            pos=tuple(args.cam_pos) if args.cam_pos is not None else (0.5, -1.5, 1.2),
            lookat=tuple(args.cam_lookat) if args.cam_lookat is not None else (0.0, -1.58, 2.0),
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

    for eps in range(args.eval_episodes):
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference
            )
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        print(f"Saved eval data to {ckpt_eval_fname}")
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

