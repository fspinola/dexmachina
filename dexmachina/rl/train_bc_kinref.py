"""Offline BC warm-start: clone kinematic references into the rl_games PPO actor.

Teacher-forced supervised regression only — no simulator, no env stepping, no
critic training. The actor/model is byte-compatible with train_rl_games.py
(same yaml network, same obs/action spaces), so the produced checkpoint loads
through the existing ``--warmstart_ckpt`` path:

    python dexmachina/rl/train_bc_kinref.py \
        --data dexmachina/assets/retargeted/allegro_hand/oakink/*_vector_oakink.pt \
        --out logs/bc_kinref/allegro_oakink

    python dexmachina/rl/train_rl_games.py ... -am hybrid --hybrid_scales 0.1 1.0 \
        --warmstart_ckpt logs/bc_kinref/allegro_oakink/nn/bc_best.pth \
        --warmstart_sigma -1.6

The checkpoint initializes the ACTOR ONLY in any meaningful sense: the value
head and value_mean_std are saved at fresh-init values (kinematic references
are not simulator transitions — nothing here can train a critic). The fixed
log-std parameter is saved at --log_std_init and can be overridden at RL time
with --warmstart_sigma.
"""

import argparse
import csv
import json
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import torch
import yaml

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.rl.bc_dataset import (
    BCKinrefDataset,
    build_bc_arrays,
    clip_action_labels,
    compose_hybrid_targets,
    load_clip,
    split_clips,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="OakInk kinref .pt files (>=1)")
    parser.add_argument("--out", required=True, help="run dir (checkpoints under <out>/nn/)")
    parser.add_argument("--hand", default="allegro_hand")
    parser.add_argument("--val_clips", type=int, default=1,
                        help="clips held out for validation (trajectory-level split); forced to 0 with a single clip")
    parser.add_argument("--label_horizon", type=int, default=1,
                        help="teacher targets ref[t+h]; 1 matches the t+1 reward frame, 0 = kinematic-replay teacher")
    parser.add_argument("--hybrid_scales", type=float, nargs=2, default=[0.1, 1.0],
                        help="MUST match the RL run's --hybrid_scales")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_std_init", type=float, default=0.0,
                        help="value written to the fixed log-std param (yaml init 0.0); not trained by BC")
    parser.add_argument("--clip_warn_frac", type=float, default=0.01,
                        help="warn if more than this fraction of labels get clipped to [-1, 1]")
    return parser.parse_args()


def build_rl_games_model(obs_dim: int, action_dim: int, device: str) -> torch.nn.Module:
    """The exact model A2CAgent builds (a2c_continuous.py) from the shared yaml."""
    with open(get_rl_config_path("rl_games_ppo_cfg")) as f:
        params = yaml.full_load(f)["params"]
    from rl_games.algos_torch.model_builder import ModelBuilder

    network = ModelBuilder().load(params)
    model = network.build({
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": params["config"]["normalize_value"],
        "normalize_input": params["config"]["normalize_input"],
    })
    return model.to(device)


def save_checkpoint(path: str, model: torch.nn.Module, metadata: dict) -> None:
    """rl_games-compatible: --warmstart_ckpt reads ['model'] via agent.set_weights."""
    torch.save({"model": model.state_dict(), "epoch": 0, "bc_metadata": metadata}, path)


@torch.no_grad()
def reconstruction_report(model, clips, arrays, hybrid_scales, horizon, device):
    """Next-joint-target error of PREDICTED actions vs ref[t+h], per group."""
    model.eval()
    stats = {"wrist_trans_m": [], "wrist_rot_rad": [], "finger_rad": [], "pred_clip_frac": []}
    for clip in clips:
        rows = arrays.clip_slices[clip.path]
        obs = arrays.observations[rows].to(device)
        mu, _, _, _ = model.a2c_network({"obs": obs})
        pred = mu.cpu()
        stats["pred_clip_frac"].append(float((pred.abs() > 1.0).float().mean()))
        n = pred.shape[0]
        for side, off in (("left", 0), ("right", 22)):
            h = clip.hands[side]
            targets = compose_hybrid_targets(pred[:, off:off + 22], h.ref_qpos[:n], h.dof_limits, hybrid_scales)
            err = (targets - h.ref_qpos[horizon:horizon + n]).abs()
            stats["wrist_trans_m"].append(float(err[:, 0:3].mean()))
            stats["wrist_rot_rad"].append(float(err[:, 3:6].mean()))
            stats["finger_rad"].append(float(err[:, 6:22].mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


@torch.no_grad()
def mse_over(model, dataset, device, batch_size=8192):
    model.eval()
    total, n = 0.0, 0
    per_dim = torch.zeros(dataset.actions.shape[1])
    for start in range(0, len(dataset), batch_size):
        obs = dataset.observations[start:start + batch_size].to(device)
        act = dataset.actions[start:start + batch_size].to(device)
        mu, _, _, _ = model.a2c_network({"obs": obs})
        sq = (mu - act) ** 2
        total += float(sq.sum())
        per_dim += sq.sum(dim=0).cpu()
        n += obs.shape[0]
    return total / (n * per_dim.numel()), (per_dim / n).numpy()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.data) < 2 and args.val_clips > 0:
        print("WARNING: single clip -> no trajectory-level validation split (val_clips=0). "
              "Overfitting that clip is intended for a single-sequence warm-start.")
        args.val_clips = 0
    train_paths, val_paths = split_clips(args.data, args.val_clips, args.seed)
    hybrid_scales = tuple(args.hybrid_scales)

    train_arrays = build_bc_arrays(train_paths, hybrid_scales, args.label_horizon, hand=args.hand)
    val_arrays = build_bc_arrays(val_paths, hybrid_scales, args.label_horizon, hand=args.hand) if val_paths else None
    val_clips = [load_clip(p, hand=args.hand) for p in val_paths]

    obs_dim = train_arrays.observations.shape[1]
    action_dim = train_arrays.actions.shape[1]
    print(f"train: {len(train_paths)} clips, {len(train_arrays.observations)} samples | "
          f"val: {len(val_paths)} clips, {0 if val_arrays is None else len(val_arrays.observations)} samples | "
          f"obs_dim={obs_dim} action_dim={action_dim}")
    label_stats = {}
    for diag in train_arrays.diagnostics + ([] if val_arrays is None else val_arrays.diagnostics):
        print("  " + diag.summary())
        label_stats[diag.path] = {
            "clipped_fraction": diag.clipped_fraction,
            "label_mean": diag.label_mean.tolist(),
            "label_std": diag.label_std.tolist(),
            "label_min": diag.label_min.tolist(),
            "label_max": diag.label_max.tolist(),
        }
        if diag.clipped_fraction > args.clip_warn_frac:
            print(f"  *** WARNING: {100 * diag.clipped_fraction:.2f}% of labels clipped in {diag.path} — "
                  "check --hybrid_scales / --label_horizon / reference smoothness ***")

    model = build_rl_games_model(obs_dim, action_dim, args.device)
    with torch.no_grad():
        model.a2c_network.sigma.fill_(args.log_std_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_set = BCKinrefDataset(train_arrays)
    val_set = BCKinrefDataset(val_arrays) if val_arrays is not None else None
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )

    os.makedirs(os.path.join(args.out, "nn"), exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        commit = "unknown"
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "action_convention": "hybrid (robot.py::translate_actions): wrist a=(ref[t+h]-ref[t])/hybrid_scales, "
                             "fingers a=2*(ref[t+h]-lo)/(hi-lo)-1; see dexmachina/rl/bc_dataset.py",
        "label_horizon": args.label_horizon,
        "hybrid_scales": list(hybrid_scales),
        "hand": args.hand,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_clips": train_paths,
        "val_clips": val_paths,
        "seed": args.seed,
        "log_std_init": args.log_std_init,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "critic_initialized": False,
        "label_stats": label_stats,
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=1)

    best_val = float("inf")
    metrics_path = os.path.join(args.out, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse", "recon_wrist_trans_m",
                         "recon_wrist_rot_rad", "recon_finger_rad", "pred_clip_frac"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, n_batches = 0.0, 0
            for batch in loader:
                obs = batch["obs"].to(args.device)
                act = batch["action"].to(args.device)
                mu, _, _, _ = model.a2c_network({"obs": obs})
                loss = torch.nn.functional.mse_loss(mu, act)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                n_batches += 1
            train_mse = running / n_batches

            row = [epoch, train_mse, "", "", "", "", ""]
            if val_set is not None:
                val_mse, _ = mse_over(model, val_set, args.device)
                recon = reconstruction_report(model, val_clips, val_arrays, hybrid_scales,
                                              args.label_horizon, args.device)
                row = [epoch, train_mse, val_mse, recon["wrist_trans_m"], recon["wrist_rot_rad"],
                       recon["finger_rad"], recon["pred_clip_frac"]]
                score = val_mse
            else:
                score = train_mse
            writer.writerow(row)
            f.flush()

            save_checkpoint(os.path.join(args.out, "nn", "bc_latest.pth"), model, metadata)
            if score < best_val:
                best_val = score
                save_checkpoint(os.path.join(args.out, "nn", "bc_best.pth"), model, metadata)
            if epoch % 10 == 0 or epoch == 1:
                extra = ""
                if val_set is not None:
                    extra = (f" val_mse={val_mse:.3e} recon(wrist {recon['wrist_trans_m'] * 1000:.1f} mm/"
                             f"{np.degrees(recon['wrist_rot_rad']):.2f} deg, "
                             f"finger {np.degrees(recon['finger_rad']):.2f} deg)")
                print(f"epoch {epoch:4d} train_mse={train_mse:.3e}{extra}")

    # Final per-dim MSE on the training set (catches a dead output dim).
    final_mse, per_dim = mse_over(model, train_set, args.device)
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump({
            "final_train_mse": final_mse,
            "best_score": best_val,
            "per_dim_train_mse": per_dim.tolist(),
        }, f, indent=1)
    print(f"done: best score {best_val:.3e}; checkpoints in {os.path.join(args.out, 'nn')}")
    print("NOTE: only the actor is meaningfully initialized. The critic (value head + "
          "value_mean_std) is at fresh-init values and must be trained from real rollouts; "
          "consider --warmstart_sigma (e.g. -1.6) and a critic-friendly warmup at RL time.")


if __name__ == "__main__":
    main()
