"""Verify the offline BC observation builder against a LIVE Genesis env (GPU).

For one OakInk clip, hard-sets hands + objects to the reference at each frame
(no physics stepping), records env.get_observations(), and compares per block
with dexmachina/rl/bc_dataset.py's offline reconstruction. Also asserts that
the env's actuated-dof order matches the canonical order hardcoded in
bc_dataset.py. Run once per new data source / after any obs-layout change;
NOT needed for BC training itself.

Velocity blocks are reported separately: sim velocities are zero under
set-state while the offline builder uses reference finite differences (the
documented teacher-forcing approximation).

    python dexmachina/rl/verify_bc_obs.py --oakink \
        --oakink_pt dexmachina/assets/retargeted/allegro_hand/oakink/e76b2_at3_vector_oakink.pt \
        --hand allegro_hand -B 1 -am hybrid --hybrid_scales 0.1 1.0

Pass --save_fixture <path.npz> to refresh dexmachina/tests/fixtures/.
"""

import numpy as np
import torch
import genesis as gs

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg, get_common_argparser
from dexmachina.rl.bc_dataset import (
    NDOF,
    allegro_dof_names,
    build_clip_observations,
    clip_action_labels,
    load_clip,
)

NONVEL_TOL = 1e-4


def block_layout(clip):
    """(name, start, end, is_velocity) per obs block, mirroring get_observations order."""
    n_kpts = next(iter(clip.hands.values())).kpt_pos.shape[1]
    blocks, b = [], 0

    def add(name, dim, vel=False):
        nonlocal b
        blocks.append((name, b, b + dim, vel))
        b += dim

    for side in ("left", "right"):
        add(f"{side}.dof_target_pos", NDOF)
        add(f"{side}.dof_pos", NDOF)
        add(f"{side}.dof_vel", NDOF, vel=True)
        add(f"{side}.kpt_pos", 3 * n_kpts)
        add(f"{side}.wrist_pose", 7)
    for obj in clip.objects:
        add(f"{obj.name}.parts_pos", 3)
        add(f"{obj.name}.parts_quat", 4)
        add(f"{obj.name}.dof_pos", obj.arti.shape[1])
        add(f"{obj.name}.state_diff", 8)
        add(f"{obj.name}.root_ang_vel", 3, vel=True)
        add(f"{obj.name}.root_lin_vel", 3, vel=True)
    add("phase", 1)
    return blocks, b


def main():
    parser = get_common_argparser()
    parser.add_argument("--steps", type=int, default=120, help="frames to compare (clamped to clip length)")
    parser.add_argument("--save_fixture", type=str, default=None,
                        help="also save the recorded env obs as a compressed .npz fixture")
    args = parser.parse_args()
    assert args.oakink and args.oakink_pt, "pass --oakink --oakink_pt <clip>.pt"
    hybrid_scales = tuple(args.hybrid_scales)

    clip = load_clip(args.oakink_pt, hand=args.hand)
    labels, diag = clip_action_labels(clip, hybrid_scales, horizon=1)
    offline = build_clip_observations(clip, labels, hybrid_scales, horizon=1).numpy()
    print(diag.summary())

    gs.init(backend=gs.gpu, logging_level="warning")
    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    env_kwargs["env_cfg"]["use_rl_games"] = True
    env = BaseEnv(**env_kwargs)
    env.reset()

    for side, robot in env.robots.items():
        expected = allegro_dof_names(side)
        if list(robot.actuated_dof_names) != expected:
            raise AssertionError(
                f"{side}: env dof order {list(robot.actuated_dof_names)} != bc_dataset "
                f"canonical order {expected} — bc_dataset.py must be updated"
            )
    print("dof order OK; env obs_dim =", env.obs_dim, "| offline obs_dim =", offline.shape[1])
    assert env.obs_dim == offline.shape[1]

    n = min(offline.shape[0], args.steps)
    rows = []
    for step in range(n):
        env.episode_length_buf[:] = step
        for robot in env.robots.values():
            robot.episode_length_buf[:] = step
            robot.set_joint_position(robot.residual_qpos[step][None], env_idxs=[0])
        for obj in env.objects.values():
            obj.episode_length_buf[:] = step
            obj.set_to_demo_step(step)
        env._compute_intermediate_values()
        rows.append(env.get_observations()["policy"][0].cpu().numpy().copy())
    sim = np.stack(rows)
    if args.save_fixture:
        np.savez_compressed(args.save_fixture, obs=sim.astype(np.float32))
        print("fixture saved:", args.save_fixture)

    blocks, total = block_layout(clip)
    assert total == sim.shape[1]
    worst = 0.0
    for name, s, e, vel in blocks:
        d = np.abs(offline[:n, s:e] - sim[:, s:e])
        tag = " [velocity: sim=0 under set-state, offline=finite-difference]" if vel else ""
        print(f"{name:32s} max|d|={d.max():.3e} mean={d.mean():.3e}{tag}")
        if not vel:
            worst = max(worst, float(d.max()))
    ok = worst < NONVEL_TOL
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (non-velocity max |d| = {worst:.3e}, tol {NONVEL_TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
