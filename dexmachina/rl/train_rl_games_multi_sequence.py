"""Train one RL policy across multiple ARCTIC sequences.

This is an additive training entrypoint. It mirrors
`dexmachina/rl/train_rl_games.py` but:
- accepts multiple `--clips` specs
- constructs `MultiSequenceEnv` instead of a single `BaseEnv`
- makes experiment naming sequence-agnostic

Caveat
------
This implementation rebuilds the underlying `BaseEnv` when an episode ends.
That is correct but slower than a true per-env task switch. It's a safe first
step that doesn't require invasive changes to `BaseEnv`/`RewardModule`.

If you need high throughput: see the plan in the assistant message.
"""

from __future__ import annotations

import os
import math
import yaml
import torch
import wandb
import pickle
import argparse
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from dexmachina.rl.sequence_sampler import expand_clip_ranges
from dexmachina.envs.multi_sequence_env import MultiSequenceEnv, MultiSequenceEnvCfg


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    if not filename.endswith("yaml"):
        filename += ".yaml"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    def to_list(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()

    if isinstance(data, dict):
        to_list(data)

    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)


def main():
    parser = get_common_argparser()

    # multi-sequence args
    parser.add_argument(
        "--clips",
        nargs="+",
        required=True,
        help=(
            "One or more clips. Each clip is 'obj-start-end-subject-uXX'. "
            "You can also pass ranges like 'box-40-200-s01-u01..u05'."
        ),
    )
    parser.add_argument("--exp_name", "-exp", type=str, default="multi")
    parser.add_argument("--horizon", "-ho", type=int, default=16)
    parser.add_argument("--checkpoint", "-ck", type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003)
    parser.add_argument("--wandb_project", "-wp", type=str, default="dexmachina")
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    parser.add_argument("--task_id_mode", choices=["onehot", "int", "none"], default="onehot")

    args = parser.parse_args()

    # We still need a "clip" for config construction (it triggers asset loads).
    # We'll use the *first* clip for initial env construction.
    parsed_clips = expand_clip_ranges(args.clips)
    args.clip = parsed_clips[0].clip_str

    hand_prefix = str(args.hand).split("_")[0]
    exp_name = hand_prefix + "-" + args.exp_name
    exp_name += f"_multi{len(parsed_clips)}_B{args.num_envs}"
    exp_name += "_" + args.action_mode
    exp_name += f"_thres{args.early_reset_threshold}"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if args.rand_init_ratio > 0:
        exp_name += f"_rand{args.rand_init_ratio}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"

    # Build base env kwargs template
    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    env_kwargs["env_cfg"]["use_rl_games"] = True

    device = torch.device("cuda:0")

    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")

    multi_cfg = MultiSequenceEnvCfg(
        clips=parsed_clips,
        seed=args.seed,
        add_task_id_to_obs=(args.task_id_mode != "none"),
        task_id_mode=("onehot" if args.task_id_mode == "onehot" else "int"),
    )

    env = MultiSequenceEnv(base_env_kwargs=env_kwargs, multi_cfg=multi_cfg, device=device)

    # RL-Games config
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)

    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["name"] = args.hand

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", args.hand))
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), f"Checkpoint file not found: {args.checkpoint}"

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5))
    agent_cfg["params"]["config"]["num_steps_per_env"] = args.horizon
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate

    # If we add task_id to obs, we must keep the policy net input consistent.
    # RL-Games infers obs dim from env.observation_space.

    # Save params for reproducibility
    env_save_kwargs = dict(env_kwargs)
    env_save_kwargs.pop("demo_data", None)
    env_save_kwargs.pop("retarget_data", None)

    wandb_cfg = agent_cfg.copy()
    wandb_cfg["env_kwargs"] = env_save_kwargs
    wandb_cfg["clips"] = [c.clip_str for c in parsed_clips]
    wandb_cfg["hand"] = args.hand

    run = wandb.init(
        project=args.wandb_project,
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )

    run_name = run.name
    run_id = run.id
    env_save_kwargs["wandb"] = dict(run_name=run_name, run_id=run_id)

    dump_yaml(os.path.join(log_root_path, exp_name, "params", "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "agent.yaml"), agent_cfg)
    pickle.dump(env_kwargs, open(os.path.join(log_root_path, exp_name, "params", "env.pkl"), "wb"))

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint)
    runner.run(runner_args)


if __name__ == "__main__":
    main()
