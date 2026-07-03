"""Render the pure KINEMATIC REFERENCE of an OakInk .pt (no policy).

Each frame teleports both hands to their retargeted joint config and both rigid objects to
their demo poses (BaseEnv.set_retarget_states, which set_dofs_position-teleports and renders
headless), then writes an mp4. This shows exactly what the kinref asks the RL policy to
reproduce -- use it to check whether the retargeting places the hands correctly on the objects
(the user's hypothesis: bad initial hand placement) independent of any RL.

Run from the dexmachina repo root:
  python render_oakink_reference.py --oakink \
    --oakink_pt dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt \
    --hand allegro_hand -B 1 --out oakink_reference.mp4
"""
import argparse
import os

import genesis as gs

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser


def main():
    parser = get_common_argparser()
    parser.add_argument("--out", type=str, default="oakink_reference.mp4")
    parser.add_argument("--cam_pos", type=float, nargs=3, default=[0.6, -1.1, 1.5])
    parser.add_argument("--cam_lookat", type=float, nargs=3, default=[-0.1, -0.1, 1.0])
    args = parser.parse_args()
    assert args.oakink and args.oakink_pt, "Pass --oakink --oakink_pt <clip>.pt"

    gs.init(backend=gs.gpu, logging_level="warning")
    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    ec = env_kwargs["env_cfg"]
    ec["num_envs"] = 1
    ec["is_eval"] = True
    ec["record_video"] = True
    ec["scene_kwargs"]["use_visualizer"] = True
    ec["render_camera"] = "front"
    ec["camera_kwargs"]["front"] = dict(
        res=(640, 640), fov=42,
        pos=tuple(args.cam_pos), lookat=tuple(args.cam_lookat),
    )

    env = BaseEnv(**env_kwargs)
    env.reset()
    n = int(env.reward_module.get_demo_length())
    print(f"[ref] objects={list(env.objects)}  frames={n}")

    env.max_video_frames = n
    env.start_recording()
    for step in range(n):
        env.set_retarget_states(step)  # teleports hands+objects and renders headless

    frames = env.get_recorded_frames(wait_for_max=False)
    assert frames, "no frames recorded"
    from moviepy import ImageSequenceClip
    ImageSequenceClip(frames, fps=30).write_videofile(args.out)
    print(f"[ref] wrote {os.path.abspath(args.out)}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
