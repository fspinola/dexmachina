"""Compute Spider-style object tracking success metrics from DexMachina eval .npy files.

This mirrors the core metric logic in Spider's `spider/postprocess/get_success_rate.py`:

- Compute mean object position tracking error (meters)
- Compute mean object orientation tracking error (radians)
- Define success if both are below thresholds

DexMachina eval files are produced by `dexmachina/rl/eval_rl_games.py` and
`dexmachina/rl/eval_rl_games_with_metrics.py` and typically contain:
  - obj_state: (T, num_envs, D) or (T, D) with [pos(3), quat(4), arti...]
  - demo_state: (T, D) with [pos(3), quat(4), arti...]

Outputs:
  - complete_data.csv: one row per eval file
  - summary.csv: aggregated success rates grouped by (hand_family, run_name, eval_name)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EvalRow:
    eval_file: str
    hand_family: str
    run_name: str
    eval_name: str
    episode: int
    checkpoint_ep: int | None
    reward: float | None
    num_frames: int
    pos_err_mean: float
    quat_err_mean: float
    arti_err_mean: float | None
    success: bool
    pos_err_threshold: float
    quat_err_threshold: float
    arti_err_threshold: float | None
    quat_format: str
    center_pos: bool
    env_idx: int
    timestamp: str


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    if not mask.any():
        return float("nan")
    return float(x[mask].mean())


def _parse_episode_from_filename(name: str) -> int:
    m = re.search(r"eval_ep(\d+)\.npy$", name)
    if not m:
        raise ValueError(f"Could not parse episode from filename: {name}")
    return int(m.group(1))


def _parse_checkpoint_ep(eval_name: str) -> int | None:
    m = re.search(r"_ep_(\d+)", eval_name)
    return int(m.group(1)) if m else None


def _parse_reward(eval_name: str) -> float | None:
    m = re.search(r"_rew_(-?\d+(?:\.\d+)?)", eval_name)
    return float(m.group(1)) if m else None


def _normalize_quat_wxyz(q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, eps, np.inf)
    return q / n


def _quat_angle_error_wxyz(q_traj: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Return per-step angle error in radians in [0, pi].

    Uses: angle = 2 * arccos(|<q1, q2>|)
    Assumes quaternions are normalized and in wxyz convention.
    """

    q_traj = _normalize_quat_wxyz(q_traj)
    q_ref = _normalize_quat_wxyz(q_ref)
    dot = np.sum(q_traj * q_ref, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _ensure_wxyz(quat: np.ndarray, quat_format: str) -> np.ndarray:
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quat last dim=4, got shape={quat.shape}")
    if quat_format == "wxyz":
        return quat
    if quat_format == "xyzw":
        # xyzw -> wxyz
        return quat[..., (3, 0, 1, 2)]
    raise ValueError(f"Unknown quat_format={quat_format!r}")


def _select_obj_state(obj_state: np.ndarray, env_idx: int) -> np.ndarray:
    obj_state = np.asarray(obj_state)
    if obj_state.ndim == 3:
        if not (0 <= env_idx < obj_state.shape[1]):
            raise IndexError(
                f"env_idx={env_idx} out of range for obj_state shape={obj_state.shape}"
            )
        return obj_state[:, env_idx, :]
    if obj_state.ndim == 2:
        return obj_state
    raise ValueError(f"Unsupported obj_state shape={obj_state.shape}")


def compute_spider_tracking_errors(
    eval_dict: dict[str, Any],
    *,
    env_idx: int,
    quat_format: str,
    center_pos: bool,
    include_arti: bool,
) -> tuple[float, float, float | None, int]:
    """Compute mean position / quaternion (and optional articulation) tracking errors.

    Returns: (pos_err_mean, quat_err_mean, arti_err_mean_or_None, num_frames)
    """

    if "obj_state" not in eval_dict or "demo_state" not in eval_dict:
        raise KeyError("eval_dict must contain 'obj_state' and 'demo_state'")

    obj_state = _select_obj_state(eval_dict["obj_state"], env_idx=env_idx)
    demo_state = np.asarray(eval_dict["demo_state"])

    if demo_state.ndim != 2:
        raise ValueError(f"Unsupported demo_state shape={demo_state.shape}")

    T = int(min(obj_state.shape[0], demo_state.shape[0]))
    if T <= 0:
        raise ValueError("No frames available after aligning obj_state and demo_state")
    obj_state = np.asarray(obj_state[:T], dtype=np.float64)
    demo_state = np.asarray(demo_state[:T], dtype=np.float64)

    if obj_state.shape[1] < 7 or demo_state.shape[1] < 7:
        raise ValueError(
            f"Expected state dim >= 7 ([pos3, quat4, ...]), got obj={obj_state.shape}, demo={demo_state.shape}"
        )

    pos_traj = obj_state[:, 0:3]
    pos_ref = demo_state[:, 0:3]
    if center_pos:
        pos_traj = pos_traj - pos_traj.mean(axis=0, keepdims=True)
        pos_ref = pos_ref - pos_ref.mean(axis=0, keepdims=True)
    pos_err = np.linalg.norm(pos_traj - pos_ref, axis=-1)

    quat_traj = _ensure_wxyz(obj_state[:, 3:7], quat_format)
    quat_ref = _ensure_wxyz(demo_state[:, 3:7], quat_format)
    quat_err = _quat_angle_error_wxyz(quat_traj, quat_ref)

    arti_err_mean: float | None = None
    if include_arti:
        if obj_state.shape[1] > 7 and demo_state.shape[1] > 7:
            arti_traj = obj_state[:, 7:]
            arti_ref = demo_state[:, 7:]
            D = min(arti_traj.shape[1], arti_ref.shape[1])
            if D > 0:
                arti_err = np.linalg.norm(arti_traj[:, :D] - arti_ref[:, :D], axis=-1)
                arti_err_mean = _safe_mean(arti_err)
        else:
            arti_err_mean = None

    return _safe_mean(pos_err), _safe_mean(quat_err), arti_err_mean, T


def _iter_eval_files(eval_root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in eval_root.glob(pattern) if p.is_file())


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _group_key(row: EvalRow) -> tuple[str, str, str]:
    return (row.hand_family, row.run_name, row.eval_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Spider-style success metric from DexMachina eval_ep*.npy files."
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="logs/rl_games",
        help="Root directory to search (relative or absolute).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*_eval/eval_ep*.npy",
        help="Glob pattern under eval_root.",
    )
    parser.add_argument(
        "--eval_files",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of eval .npy files (overrides --eval_root/--pattern).",
    )
    parser.add_argument("--env_idx", type=int, default=0, help="Env index to evaluate.")
    parser.add_argument(
        "--quat_format",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion convention stored in eval files.",
    )
    parser.add_argument(
        "--center_pos",
        action="store_true",
        help="If set, subtract mean position from traj and ref before error.",
    )
    parser.add_argument(
        "--pos_err_threshold",
        type=float,
        default=0.03,
        help="Success threshold for mean position error (meters).",
    )
    parser.add_argument(
        "--quat_err_threshold",
        type=float,
        default=0.5,
        help="Success threshold for mean orientation error (radians).",
    )
    parser.add_argument(
        "--include_arti",
        action="store_true",
        help="If set, also compute an articulation tracking error (mean L2).",
    )
    parser.add_argument(
        "--arti_err_threshold",
        type=float,
        default=None,
        help="Optional articulation threshold; if set, included in success condition.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <eval_root>/processed_spider_metric",
    )

    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    if args.eval_files:
        eval_files = [Path(p).expanduser().resolve() for p in args.eval_files]
    else:
        eval_files = _iter_eval_files(eval_root, args.pattern)

    if not eval_files:
        print(f"No eval files found. eval_root={eval_root} pattern={args.pattern!r}")
        return 2

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (eval_root / "processed_spider_metric")
    )
    complete_csv = out_dir / "complete_data.csv"
    summary_csv = out_dir / "summary.csv"

    rows: list[EvalRow] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for p in eval_files:
        try:
            d = np.load(str(p), allow_pickle=True).item()
            pos_err_mean, quat_err_mean, arti_err_mean, num_frames = (
                compute_spider_tracking_errors(
                    d,
                    env_idx=int(args.env_idx),
                    quat_format=str(args.quat_format),
                    center_pos=bool(args.center_pos),
                    include_arti=bool(args.include_arti),
                )
            )

            # Parse metadata from path relative to eval_root if possible.
            try:
                rel = p.resolve().relative_to(eval_root)
                parts = rel.parts
                hand_family = parts[0] if len(parts) > 0 else ""
                run_name = parts[1] if len(parts) > 1 else ""
                eval_name = parts[2] if len(parts) > 2 else p.parent.name
            except Exception:
                hand_family = ""
                run_name = ""
                eval_name = p.parent.name

            episode = _parse_episode_from_filename(p.name)
            checkpoint_ep = _parse_checkpoint_ep(eval_name)
            reward = _parse_reward(eval_name)

            success = (
                math.isfinite(pos_err_mean)
                and math.isfinite(quat_err_mean)
                and (pos_err_mean <= float(args.pos_err_threshold))
                and (quat_err_mean <= float(args.quat_err_threshold))
            )
            if args.arti_err_threshold is not None:
                # If user asked to threshold articulation, require finite arti err.
                success = (
                    success
                    and (arti_err_mean is not None)
                    and math.isfinite(float(arti_err_mean))
                    and (float(arti_err_mean) <= float(args.arti_err_threshold))
                )

            rows.append(
                EvalRow(
                    eval_file=str(p),
                    hand_family=hand_family,
                    run_name=run_name,
                    eval_name=eval_name,
                    episode=int(episode),
                    checkpoint_ep=checkpoint_ep,
                    reward=reward,
                    num_frames=int(num_frames),
                    pos_err_mean=float(pos_err_mean),
                    quat_err_mean=float(quat_err_mean),
                    arti_err_mean=float(arti_err_mean)
                    if (arti_err_mean is not None and math.isfinite(float(arti_err_mean)))
                    else (None if arti_err_mean is None else float(arti_err_mean)),
                    success=bool(success),
                    pos_err_threshold=float(args.pos_err_threshold),
                    quat_err_threshold=float(args.quat_err_threshold),
                    arti_err_threshold=float(args.arti_err_threshold)
                    if args.arti_err_threshold is not None
                    else None,
                    quat_format=str(args.quat_format),
                    center_pos=bool(args.center_pos),
                    env_idx=int(args.env_idx),
                    timestamp=now,
                )
            )
        except Exception as e:
            print(f"Warning: failed to process {p}: {e}")
            continue

    if not rows:
        print("No valid eval files processed.")
        return 2

    # Write complete_data.csv
    complete_dicts = [asdict(r) for r in rows]
    complete_fields = list(complete_dicts[0].keys())
    _write_csv(complete_csv, complete_dicts, complete_fields)
    print(f"Wrote complete data: {complete_csv}")

    # Aggregate summary
    grouped: dict[tuple[str, str, str], list[EvalRow]] = {}
    for r in rows:
        grouped.setdefault(_group_key(r), []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for (hand_family, run_name, eval_name), grp in sorted(grouped.items()):
        n = len(grp)
        succ = sum(1 for r in grp if r.success)
        summary_rows.append(
            {
                "hand_family": hand_family,
                "run_name": run_name,
                "eval_name": eval_name,
                "num_eval_files": n,
                "success_rate": succ / max(1, n),
                "avg_pos_err_mean": float(np.nanmean([r.pos_err_mean for r in grp])),
                "avg_quat_err_mean": float(np.nanmean([r.quat_err_mean for r in grp])),
                "pos_err_threshold": float(args.pos_err_threshold),
                "quat_err_threshold": float(args.quat_err_threshold),
                "timestamp": now,
            }
        )

    summary_fields = list(summary_rows[0].keys())
    _write_csv(summary_csv, summary_rows, summary_fields)
    print(f"Wrote summary: {summary_csv}")

    overall_success = sum(1 for r in rows if r.success) / max(1, len(rows))
    print(
        f"Overall success rate: {overall_success:.4f} ({overall_success * 100:.2f}%) over {len(rows)} eval files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())