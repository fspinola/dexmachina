"""Self-contained OakInk policy eval + video render.

eval_rl_games.py bakes in ARCTIC assumptions (flat demo_data['obj_pos'],
objects[0]-only, ARCTIC-framed camera), which break on the OakInk two-object
per-side env. This script loads the env straight from the checkpoint's env.pkl,
sets an OakInk-framed camera, restores the policy, rolls out one deterministic
episode, records it, and reports per-object tracking distance from rew_dict.

Run from the dexmachina repo root:
  python eval_oakink_video.py -ck <path/to/nn/last_..._ep_2000_..pth> \
      --cam_pos 0.6 -1.1 1.5 --cam_lookat -0.1 -0.1 1.0 -o video_ep2000.mp4
"""
import argparse
import math
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-ck", "--checkpoint", required=True)
    p.add_argument("-B", "--num_envs", type=int, default=1)
    p.add_argument("-ne", "--eval_episodes", type=int, default=1)
    p.add_argument("-o", "--out", type=str, default=None, help="output mp4 path")
    p.add_argument("--cam_pos", type=float, nargs=3, default=[0.6, -1.1, 1.5])
    p.add_argument("--cam_lookat", type=float, nargs=3, default=[-0.1, -0.1, 1.0])
    args = p.parse_args()

    ckpt_dir = "/".join(args.checkpoint.split("/")[:-2])
    env_pkl = os.path.join(ckpt_dir, "params", "env.pkl")
    assert os.path.exists(env_pkl), f"missing {env_pkl}"
    with open(env_pkl, "rb") as f:
        env_kwargs = pickle.load(f)

    assert env_kwargs["env_cfg"]["use_rl_games"]
    env_kwargs["env_cfg"]["is_eval"] = True
    env_kwargs["env_cfg"]["early_reset_threshold"] = 0.0
    env_kwargs["env_cfg"]["num_envs"] = args.num_envs
    env_kwargs["env_cfg"]["rand_init_ratio"] = 0.0
    env_kwargs["rand_cfg"]["randomize"] = False
    env_kwargs.pop("curriculum_cfg", None)
    for cfg in env_kwargs["object_cfgs"].values():
        cfg["actuated"] = False  # kinematic objects, no PD at eval

    env_kwargs["env_cfg"]["scene_kwargs"]["use_visualizer"] = True
    env_kwargs["env_cfg"]["record_video"] = True
    env_kwargs["env_cfg"]["camera_kwargs"]["front"] = dict(
        res=(640, 640), fov=42,
        pos=tuple(args.cam_pos), lookat=tuple(args.cam_lookat),
    )

    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")
    env = BaseEnv(**env_kwargs)
    uenv = env

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env}
    )
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(os.path.abspath(args.checkpoint))
    agent.reset()

    obj_names = list(uenv.object_names)
    print(f"[eval] objects={obj_names}  ep_len={uenv.max_episode_length}  B={uenv.num_envs}")

    for eps in range(args.eval_episodes):
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        uenv.start_recording()
        uenv.max_video_frames = int(uenv.max_episode_length)

        stats = defaultdict(list)
        for _ in range(uenv.max_episode_length):
            with torch.inference_mode():
                actions = agent.get_action(obs, is_deterministic=True)
                obs, rew, dones, infos = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                rd = uenv.rew_dict
                for k in ("pos_dist", "rot_dist", "arti_dist", "task_rew", "imi_rew",
                          "con_rew", "contact_rew_left", "contact_rew_right"):
                    if k in rd:
                        stats[k].append(float(rd[k].float().mean().cpu()))

        print(f"[eval ep{eps}] per-step means over episode:")
        for k, v in stats.items():
            print(f"    {k:20s} mean={np.mean(v):.4f}  min={np.min(v):.4f}  max={np.max(v):.4f}")

        frames = uenv.get_recorded_frames()
        out = args.out or os.path.join(ckpt_dir, f"oakink_eval_ep{eps}.mp4")
        from moviepy import ImageSequenceClip
        ImageSequenceClip(frames, fps=int(1 / uenv.dt / 2)).write_videofile(out)
        print(f"[eval] wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
