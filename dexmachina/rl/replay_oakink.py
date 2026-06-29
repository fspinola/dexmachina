"""Kinematic set-state replay for an OakInk .pt (first port milestone: validate coordinates).

Builds the DexMachina env for an exported OakInk clip (export_dexmachina_oakink.py) and, each
frame, drives both hands to their ``residual_qpos`` and sets both rigid objects to their demo
poses (BaseEnv.set_retarget_states). With ``--vis`` it opens a Genesis viewer to eyeball
hand/object alignment + table seating (the same thing your viser OakInk viz shows). Headless
(no ``--vis``) it runs a numeric check: every object's simulated root pose must match its demo
pose (set-state), and no frame may be NaN.

Usage:
    python -m dexmachina.rl.replay_oakink --oakink \
        --oakink_pt /nas/.../assets/retargeted/inspire_hand/oakink/<clip>_vector_oakink.pt \
        --hand inspire_hand -B 1            # headless numeric check
    python -m dexmachina.rl.replay_oakink --oakink --oakink_pt ... --hand inspire_hand -B 1 --vis
"""

import argparse

import numpy as np
import torch
import genesis as gs

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser


def main():
    parser = get_common_argparser()
    parser.add_argument("--max_steps", type=int, default=120, help="headless: frames to check (clamped to clip len)")
    args = parser.parse_args()
    assert args.oakink and args.oakink_pt, "Pass --oakink --oakink_pt <clip>.pt"

    gs.init(backend=gs.gpu, logging_level="warning")
    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    env = BaseEnv(**env_kwargs)
    env.reset()

    n = int(env.reward_module.get_demo_length())
    steps = n if args.vis else min(n, args.max_steps)
    print(f"OakInk replay: {len(env.objects)} objects {list(env.objects)}, {n} frames, checking {steps}.")

    max_obj_err = 0.0
    nan_frames = 0
    for step in range(steps):
        env.set_retarget_states(step)
        if env.nan_envs.any():
            nan_frames += 1
        for name, obj in env.objects.items():
            demo_pos = env.reward_module.demo_tensors[f"obj_pos::{name}"][step].to(obj.root_pos.device)
            err = float(torch.norm(obj.root_pos[0] - demo_pos))
            max_obj_err = max(max_obj_err, err)
    print(f"[headless check] max object root vs demo error = {max_obj_err*1000:.2f} mm | nan frames = {nan_frames}")
    if not args.vis:
        ok = max_obj_err < 0.02 and nan_frames == 0  # set-state should be ~exact
        print("RESULT:", "PASS" if ok else "FAIL (inspect seating / demo wiring)")
    else:
        import time
        print("Viewer up; looping. Ctrl-C to exit.")
        while True:
            for step in range(n):
                env.set_retarget_states(step)
                time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
