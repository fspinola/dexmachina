"""Evaluate/deploy a *shared* multi-object policy by specifying the target object.

This is an additive entrypoint for the "single policy for many tasks" setup.

It intentionally mirrors `dexmachina/rl/eval_rl_games.py` but adds:
- `--clips` list used to define the task-id space
- `--object` to select the inference task (object-conditioned)

The policy must have been trained with task-id appended to observations, using
`train_rl_games_multi_sequence.py` (or an equivalent setup).

Notes
-----
- This script loads `env.pkl` saved in the checkpoint run folder to recreate the
  environment configuration.
- It then overrides only what is needed for eval.
- It uses the standard RL-Games Runner/Player just like existing scripts.
"""

from __future__ import annotations

import os
import math
import yaml
import pickle
import argparse

import torch
import numpy as np
import genesis as gs

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from dexmachina.rl.sequence_sampler import expand_clip_ranges
from dexmachina.envs.multi_sequence_inference_env import (
    InferenceTask,
    MultiSequenceInferenceCfg,
    MultiSequenceInferenceEnv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-ck", type=str, required=True, help="Path to rl-games .pth checkpoint")
    parser.add_argument("--clips", nargs="+", required=True, help="Clips used to define task-id space")
    parser.add_argument("--object", "-o", type=str, required=True, help="Target object name (e.g., box)")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--vis", "-v", action="store_true")
    parser.add_argument("--show_markers", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--raytrace", action="store_true")
    parser.add_argument("--task_id_mode", choices=["onehot", "int", "none"], default="onehot")
    parser.add_argument("--task_id_key", choices=["object", "clip"], default="object")

    args = parser.parse_args()

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    if not os.path.exists(saved_cfg_fname):
        raise FileNotFoundError(f"Saved env config not found: {saved_cfg_fname}")

    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)

    # Eval overrides (similar to eval_rl_games.py)
    env_kwargs["env_cfg"]["is_eval"] = True
    env_kwargs["env_cfg"]["early_reset_threshold"] = 0.0
    env_kwargs["rand_cfg"]["randomize"] = False
    env_kwargs["env_cfg"]["num_envs"] = int(args.num_envs)
    env_kwargs["env_cfg"]["rand_init_ratio"] = 0.0

    if args.raytrace and args.record_video:
        env_kwargs["env_cfg"]["scene_kwargs"]["raytrace"] = True

    if args.vis:
        env_kwargs["env_cfg"]["scene_kwargs"]["use_visualizer"] = True
        env_kwargs["env_cfg"]["scene_kwargs"]["show_viewer"] = True

    if args.record_video:
        env_kwargs["env_cfg"]["record_video"] = True
        env_kwargs["env_cfg"]["scene_kwargs"]["use_visualizer"] = True

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
            num_vis_contacts=16,
            sources=["demo"],
            obj_parts=["top", "bottom"],
            hand_sides=["left", "right"],
        )
        env_kwargs["contact_marker_cfgs"] = marker_cfgs
        env_kwargs["env_cfg"]["scene_kwargs"]["visualize_contact"] = True

    # Remove curriculum during eval
    env_kwargs.pop("curriculum_cfg", None)

    device = torch.device("cuda:0")
    gs.init(backend=gs.gpu, logging_level="warning")

    parsed_clips = expand_clip_ranges(args.clips)
    cfg = MultiSequenceInferenceCfg(
        clips=parsed_clips,
        add_task_id_to_obs=(args.task_id_mode != "none"),
        task_id_mode=("onehot" if args.task_id_mode == "onehot" else "int"),
        task_id_key=args.task_id_key,
    )

    env = MultiSequenceInferenceEnv(base_env_kwargs=env_kwargs, cfg=cfg, device=device)
    env.set_task(InferenceTask(obj_name=args.object))

    # RL-Games config
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    # play policy
    runner.run({"train": False, "play": True, "checkpoint": os.path.abspath(args.checkpoint)})


if __name__ == "__main__":
    main()
