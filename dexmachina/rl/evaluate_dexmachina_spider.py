"""SPIDER-protocol object-tracking evaluation for DexMachina.

This is a faithful port of SPIDER's ``spider/postprocess/evaluate_dexmachina.py``
(the ``evaluate_rl_eval`` path) so that DexMachina RL rollouts can be scored with
*exactly* the same metric definitions SPIDER reports in its paper, enabling an
apples-to-apples comparison.

------------------------------------------------------------------------------
WHAT SPIDER'S PROTOCOL IS (and how it differs from DexMachina / ManipTrans)
------------------------------------------------------------------------------
DexMachina / ManipTrans report a *success-style* object metric -- part-wise ADD
followed by an AUC over a distance threshold (``add_auc`` / ``success_at_thresh``
in our eval files). That metric saturates (success or not) and is driven by an
object-diameter-relative threshold.

SPIDER instead reports the *raw mean tracking error* over the WHOLE rollout, with
three channels:

  - pos_dist  : L2 distance ``||obj_pos - demo_pos||``                 (meters)
  - rot_dist  : geodesic rotation distance
                ``2 * arcsin( clip(|| (q_obj * conj(q_demo))[1:4] ||, 0, 1) )``  (radians)
  - arti_dist : L1 distance ``|obj_arti - demo_arti|``                 (radians)

Each channel is averaged over *all* frames of the rollout (the episode is run to
full length -- DexMachina eval already sets ``early_reset_threshold = 0.0`` so
there is no early termination), then reported as mean +/- std.

Across a replicate axis (seeds in SPIDER; objects/runs here) the per-rollout means
are aggregated into a second mean +/- std, and printed as Markdown / LaTeX tables.

Important fidelity notes for DexMachina:
  * DexMachina's native ``rotation_distance`` (envs/reward_utils.py) already uses
    the *same* arcsin geodesic formula, and ``position_distance`` is L2 -- so the
    pos/rot channels saved in the eval ``rew_dict`` match SPIDER. We still recompute
    from raw ``obj_state`` / ``demo_state`` to be self-contained and convention-safe.
  * DexMachina's saved ``arti_dist`` is ``(obj_arti - demo_arti)**2 / 2`` (a squared
    half-distance used for the reward), which is NOT SPIDER's L1 articulation error.
    This script recomputes arti as L1, matching SPIDER.

------------------------------------------------------------------------------
FRAME ALIGNMENT (the off-by-one that matters for a *fair* comparison)
------------------------------------------------------------------------------
DexMachina's eval loop (eval_rl_games[_with_metrics].py) stores, at loop iter t:
    obj_state[t]  = obj.root_pos read AFTER env.step()  -> the object at frame t+1
    demo_state[t] = demo[t]       captured BEFORE the step (episode_length_buf=t)
i.e. the saved ``demo_state`` lags the saved ``obj_state`` by exactly one frame.
Meanwhile DexMachina's *reward* increments episode_length_buf BEFORE computing the
reward (base_env.step), so it correctly compares ``obj@(t+1)`` against ``demo[t+1]``.

Consequence: naively pairing ``obj_state[t]`` with ``demo_state[t]`` (what SPIDER's
``evaluate_rl_eval`` literally does) measures tracking against a 1-frame-stale
reference, slightly inflating the error. SPIDER's own-method evaluator
(``evaluate_trajectory``) is aware of this and applies a ``+1`` correction.

This script therefore defaults to ``--align shift``: it pairs ``obj_state[t]`` with
``demo_state[t+1]`` (dropping the final, post-reset frame), which reproduces the
DexMachina reward's per-frame pos/rot tracking error EXACTLY -- the true tracking
error the policy achieved, consistent with SPIDER's +1-corrected own-method numbers.
Use ``--align raw`` for byte-identical behavior to SPIDER's ``evaluate_rl_eval``
(no shift), or ``--align clamp`` for the +1/min-clamp variant that keeps all frames.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    # Per-run table over every eval file under logs/rl_games/inspire_hand
    python -m dexmachina.rl.evaluate_dexmachina_spider --hand inspire_hand

    # Grouped comparison: rows = object clips, cols = method tags (+ Average row)
    python -m dexmachina.rl.evaluate_dexmachina_spider --hand inspire_hand --summary

    # LaTeX tables, only some methods / tasks, pick best-reward checkpoint per run
    python -m dexmachina.rl.evaluate_dexmachina_spider --summary --latex \
        --methods graphcon para --ckpt_select best_reward

    # Score explicit files (mirrors compute_spider_metric.py's --eval_files)
    python -m dexmachina.rl.evaluate_dexmachina_spider --eval_files A/eval_ep0.npy B/eval_ep0.npy

    # Sanity-check: confirm recomputed pos/rot match the values saved in rew_dict
    python -m dexmachina.rl.evaluate_dexmachina_spider --eval_files A/eval_ep0.npy --sanity
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers -- (w, x, y, z) convention, matching Genesis / DexMachina.
# These are byte-for-byte the same math as SPIDER's evaluate_dexmachina.py.
# ---------------------------------------------------------------------------

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _rotation_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic rotation distance in radians (SPIDER convention)."""
    qd = _quat_mul(q1, _quat_conjugate(q2))
    return 2.0 * np.arcsin(
        np.clip(np.linalg.norm(qd[..., 1:4], axis=-1), a_min=0.0, a_max=1.0)
    )


# Unified metric keys / labels (match SPIDER's naming).
METRIC_KEYS = ["pos_dist", "rot_dist", "arti_dist"]
METRIC_LABELS = {
    "pos_dist": "Position Distance (L2, m)",
    "rot_dist": "Rotation Distance (geodesic, rad)",
    "arti_dist": "Articulation Distance (L1, rad)",
}

# Known ARCTIC object names (used to locate the object/clip token in run names).
OBJECT_NAMES = ["box", "ketchup", "laptop", "mixer", "notebook", "waffleiron",
                "capsulemachine", "espressomachine", "microwave", "phone", "scissors"]


# ---------------------------------------------------------------------------
# Core metric: recompute SPIDER tracking errors from a single eval .npy
# ---------------------------------------------------------------------------

def _align_obj_demo(obj: np.ndarray, demo: np.ndarray, align: str):
    """Return (obj_sel, demo_sel) paired per the chosen frame alignment.

    See the module docstring. ``obj_state[t]`` is the object AT frame t+1, while
    ``demo_state[t]`` is the reference at frame t.
      - shift : pair obj_state[t] with demo_state[t+1]; drop the final (post-reset)
                frame. Reproduces the env reward's tracking error exactly. (default)
      - clamp : pair obj_state[t] with demo_state[min(t+1, T-1)]; keep all frames.
      - raw   : pair obj_state[t] with demo_state[t]; what SPIDER's evaluate_rl_eval does.
    """
    T = int(min(obj.shape[0], demo.shape[0]))
    obj, demo = obj[:T], demo[:T]
    if align == "raw":
        return obj, demo
    if align == "clamp":
        idx = np.minimum(np.arange(T) + 1, T - 1)
        return obj, demo[idx]
    if align == "shift":
        return obj[:-1], demo[1:]
    raise ValueError(f"unknown align={align!r}")


def evaluate_eval_npy(npy_path: str, env_idx: int = 0, align: str = "shift") -> dict:
    """Recompute SPIDER pos/rot/arti tracking errors from a DexMachina eval file.

    The eval ``.npy`` (produced by eval_rl_games[_with_metrics].py) is a dict with:
        obj_state  : (T, num_envs, >=8)  -- [pos(3), quat(4, wxyz), arti(>=1)]
        demo_state : (T, >=8)            -- same layout, the reference

    ``align`` controls the obj/demo frame pairing (see ``_align_obj_demo`` and the
    module docstring). Returns a dict with, for each channel, ``<key>_mean`` and
    ``<key>_std`` over the paired frames, plus ``num_frames``. Env 0 is the rollout.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    if "obj_state" not in data or "demo_state" not in data:
        raise KeyError(f"{npy_path} missing 'obj_state'/'demo_state'")

    obj = np.asarray(data["obj_state"], dtype=np.float64)
    demo = np.asarray(data["demo_state"], dtype=np.float64)

    # obj_state may be (T, num_envs, D) or (T, D); pick the rollout env.
    if obj.ndim == 3:
        obj = obj[:, env_idx, :]
    if demo.ndim == 3:           # defensive; demo is normally (T, D)
        demo = demo[:, 0, :]
    if obj.shape[-1] < 8 or demo.shape[-1] < 8:
        raise ValueError(
            f"Expected state dim >= 8 ([pos3, quat4, arti>=1]), got "
            f"obj={obj.shape}, demo={demo.shape}"
        )

    obj_s, demo_s = _align_obj_demo(obj, demo, align)

    pos_dist = np.linalg.norm(obj_s[:, :3] - demo_s[:, :3], axis=-1)
    rot_dist = _rotation_distance(obj_s[:, 3:7], demo_s[:, 3:7])
    # Articulation: L1 over all DOF columns (index 7:). Single-DOF ARCTIC objects
    # reduce this to SPIDER's exact scalar |obj[:,7] - demo[:,7]|.
    arti_dist = np.abs(obj_s[:, 7:] - demo_s[:, 7:]).mean(axis=-1)

    out = {"num_frames": int(pos_dist.shape[0]), "align": align}
    for key, vals in [("pos_dist", pos_dist), ("rot_dist", rot_dist), ("arti_dist", arti_dist)]:
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))

    # Sanity payload: the per-frame values DexMachina saved in rew_dict, for env_idx,
    # aligned to the same frame count (saved arti is (d^2)/2 -> convert to L1 |d|).
    out["_saved"] = {}
    n = pos_dist.shape[0]
    for k in ("pos_dist", "rot_dist", "arti_dist"):
        if k not in data:
            continue
        sv = np.asarray(data[k], dtype=np.float64)
        if sv.ndim == 2:
            sv = sv[:, min(env_idx, sv.shape[1] - 1)]
        sv = sv.reshape(-1)[:n]
        if k == "arti_dist":
            sv = np.sqrt(np.clip(2.0 * sv, 0.0, None))  # (d^2)/2 -> |d| (L1)
        out["_saved"][k] = float(np.nanmean(sv))
    return out


# ---------------------------------------------------------------------------
# Discovery / path parsing
# ---------------------------------------------------------------------------

def _find_object_clip_token(run_dir: str) -> tuple[str | None, str | None]:
    """From a run directory name, return (object_name, clip_token).

    Run names look like:
        inspire-graphcon_box30-230-s01-u01_B12000_hybrid_thres0.6_...
        inspire-graphcon_j178563_ketchup40-340-s01-u02_B12000_hybrid_...
    The clip token is the underscore field that starts with a known object name
    followed by a digit, e.g. 'box30-230-s01-u01'.
    """
    for tok in run_dir.split("_"):
        m = re.match(r"([a-zA-Z]+)\d", tok)
        if m and m.group(1).lower() in OBJECT_NAMES:
            return m.group(1).lower(), tok
    # Fallback: any <alpha><digits>-<digits> token (e.g. unknown object).
    for tok in run_dir.split("_"):
        m = re.match(r"([a-zA-Z]+)\d+-\d+", tok)
        if m:
            return m.group(1).lower(), tok
    return None, None


def parse_run_dir(run_dir: str) -> dict:
    """Parse a run directory name into {robot, method, object, clip, batch}."""
    fields = run_dir.split("_")
    head = fields[0]                          # e.g. 'inspire-graphcon'
    if "-" in head:
        robot, method = head.split("-", 1)
    else:
        robot, method = head, head
    obj_name, clip = _find_object_clip_token(run_dir)
    batch = next((f for f in fields if re.fullmatch(r"B\d+", f)), None)
    return {
        "robot": robot,
        "method": method,
        "object": obj_name,
        "clip": clip,             # full token, used as the cross-method 'task' key
        "batch": batch,
        "run_dir": run_dir,
    }


def _parse_ckpt_dir(ckpt_dir: str) -> dict:
    """Parse an eval directory name '<ckpt>_eval' -> {ep, reward}.

    Examples:
        'last_inspire_hand_ep_5000_rew_162.21649_eval' -> ep=5000, reward=162.21649
        'inspire_hand_eval'                            -> ep=None,  reward=None (final)
    """
    name = ckpt_dir[:-len("_eval")] if ckpt_dir.endswith("_eval") else ckpt_dir
    ep = re.search(r"_ep_(\d+)", name)
    rew = re.search(r"_rew_(-?\d+(?:\.\d+)?)", name)
    return {
        "ckpt_name": name,
        "ep": int(ep.group(1)) if ep else None,
        "reward": float(rew.group(1)) if rew else None,
    }


def discover_eval_files(eval_root: str, episode: int | str = 0) -> list[dict]:
    """Walk ``eval_root`` for ``<run>/<ckpt>_eval/eval_ep<episode>.npy`` files.

    Returns a list of records with parsed metadata (robot/method/object/clip,
    checkpoint ep & reward, and the npy path). ``episode='*'`` collects every
    saved episode.
    """
    pat = "eval_ep*.npy" if episode == "*" else f"eval_ep{episode}.npy"
    paths = sorted(glob.glob(os.path.join(eval_root, "**", "*_eval", pat), recursive=True))
    records = []
    for p in paths:
        ckpt_dir = os.path.basename(os.path.dirname(p))
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
        rec = parse_run_dir(run_dir)
        rec.update(_parse_ckpt_dir(ckpt_dir))
        ep_m = re.search(r"eval_ep(\d+)\.npy$", p)
        rec["episode"] = int(ep_m.group(1)) if ep_m else 0
        rec["path"] = p
        records.append(rec)
    return records


def select_checkpoint(records: list[dict], how: str) -> list[dict]:
    """Keep one eval record per (run_dir) according to ``how``.

    how in {'all', 'final', 'latest_ep', 'best_reward'}.
      - all         : keep every record (no selection)
      - final       : the '<hand>_eval' dir (ep is None), else fall back to latest_ep
      - latest_ep   : highest checkpoint episode
      - best_reward : highest parsed reward (falls back to latest_ep, then final)
    """
    if how == "all":
        return records
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_run[r["run_dir"]].append(r)

    chosen = []
    for run_dir, recs in by_run.items():
        if how == "final":
            finals = [r for r in recs if r["ep"] is None]
            pick = finals[0] if finals else max(recs, key=lambda r: (r["ep"] or -1))
        elif how == "latest_ep":
            pick = max(recs, key=lambda r: (r["ep"] if r["ep"] is not None else -1))
        elif how == "best_reward":
            with_rew = [r for r in recs if r["reward"] is not None]
            if with_rew:
                pick = max(with_rew, key=lambda r: r["reward"])
            else:
                pick = max(recs, key=lambda r: (r["ep"] if r["ep"] is not None else -1))
        else:
            raise ValueError(f"unknown ckpt_select={how!r}")
        chosen.append(pick)
    return chosen


# ---------------------------------------------------------------------------
# Formatting helpers (match SPIDER style)
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    if mean != mean:  # NaN
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


def _fmt_latex(mean: float, std: float) -> str:
    if mean != mean:
        return "N/A"
    return f"${mean:.4f} \\pm {std:.4f}$"


def _aggregate(values: list[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if v == v], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


# ---------------------------------------------------------------------------
# Per-run detail table
# ---------------------------------------------------------------------------

def print_per_run_table(rows: list[dict], align: str = "shift") -> None:
    print(f"\n{'='*120}")
    print(f"SPIDER-protocol per-run object tracking (mean ± std over frames, align={align})")
    print(f"{'='*120}")
    header = (f"  {'method':<14} {'clip':<22} {'ckpt':<10} {'T':>4}  "
              f"{'pos_dist (m)':>20}  {'rot_dist (rad)':>20}  {'arti_dist (rad)':>20}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        m = r["metrics"]
        ckpt = f"ep{r['ep']}" if r["ep"] is not None else "final"
        print(
            f"  {str(r['method']):<14} {str(r['clip']):<22} {ckpt:<10} "
            f"{m['num_frames']:>4}  "
            f"{_fmt(m['pos_dist_mean'], m['pos_dist_std']):>20}  "
            f"{_fmt(m['rot_dist_mean'], m['rot_dist_std']):>20}  "
            f"{_fmt(m['arti_dist_mean'], m['arti_dist_std']):>20}"
        )


# ---------------------------------------------------------------------------
# Grouped summary tables: rows = task (clip), cols = method
# ---------------------------------------------------------------------------

def build_grouped(rows: list[dict]) -> tuple[dict, list[str], list[str]]:
    """results[method][task] = list of per-rollout metric dicts."""
    results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    methods: list[str] = []
    tasks: list[str] = []
    for r in rows:
        method = str(r["method"])
        task = str(r["clip"])
        results[method][task].append(r["metrics"])
        if method not in methods:
            methods.append(method)
        if task not in tasks:
            tasks.append(task)
    return results, sorted(methods), sorted(tasks)


def _cell_mean_std(records: list[dict], mk: str) -> tuple[float, float]:
    """Aggregate one (method, task) cell for metric ``mk``.

    If multiple rollouts (seeds/episodes/runs) -> mean +/- std ACROSS rollouts.
    If a single rollout -> its frame mean +/- frame std (matches SPIDER single-run).
    """
    if not records:
        return float("nan"), float("nan")
    if len(records) == 1:
        return records[0][f"{mk}_mean"], records[0][f"{mk}_std"]
    return _aggregate([rec[f"{mk}_mean"] for rec in records])


def print_markdown_tables(results, methods, tasks) -> None:
    for mk in METRIC_KEYS:
        print(f"\n## {METRIC_LABELS[mk]} (lower is better)\n")
        cols = ["Task"] + methods + (["Winner"] if len(methods) >= 2 else [])
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join("---" for _ in cols) + " |")

        per_method_task_means: dict[str, list[float]] = {m: [] for m in methods}
        for task in tasks:
            cells = [task]
            means = {}
            for method in methods:
                mean, std = _cell_mean_std(results.get(method, {}).get(task, []), mk)
                cells.append(_fmt(mean, std))
                means[method] = mean
                if mean == mean:
                    per_method_task_means[method].append(mean)
            if len(methods) >= 2:
                valid = {m: v for m, v in means.items() if v == v}
                cells.append(min(valid, key=valid.get) if len(valid) >= 2 else "-")
            print("| " + " | ".join(cells) + " |")

        # Average across tasks (mean +/- std over the per-task cell means).
        avg_cells = ["**Average**"]
        avg_means = {}
        for method in methods:
            mean, std = _aggregate(per_method_task_means[method])
            avg_cells.append(f"**{_fmt(mean, std)}**")
            avg_means[method] = mean
        if len(methods) >= 2:
            valid = {m: v for m, v in avg_means.items() if v == v}
            avg_cells.append(f"**{min(valid, key=valid.get)}**" if len(valid) >= 2 else "-")
        print("| " + " | ".join(avg_cells) + " |")


def print_latex_tables(results, methods, tasks) -> None:
    col_spec = "l" + "c" * len(methods)
    for mk in METRIC_KEYS:
        print(f"\n% {METRIC_LABELS[mk]} (lower is better)")
        print("\\begin{table}[h]\n\\centering")
        print(f"\\caption{{{METRIC_LABELS[mk]} (lower is better)}}")
        print(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule")
        print("Task & " + " & ".join(m.replace("_", "\\_") for m in methods) + " \\\\")
        print("\\midrule")

        per_method_task_means: dict[str, list[float]] = {m: [] for m in methods}
        for task in tasks:
            means = {m: _cell_mean_std(results.get(m, {}).get(task, []), mk) for m in methods}
            best = min((v[0] for v in means.values() if v[0] == v[0]), default=float("nan"))
            cells = [task.replace("_", "\\_")]
            for method in methods:
                mean, std = means[method]
                if mean == mean:
                    per_method_task_means[method].append(mean)
                cell = _fmt_latex(mean, std)
                if mean == mean and mean == best:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            print(" & ".join(cells) + " \\\\")

        print("\\midrule")
        avg_cells = ["\\textbf{Average}"]
        for method in methods:
            avg_cells.append(_fmt_latex(*_aggregate(per_method_task_means[method])))
        print(" & ".join(avg_cells) + " \\\\")
        print("\\bottomrule\n\\end{tabular}\n\\end{table}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SPIDER-protocol object tracking eval for DexMachina eval_ep*.npy files."
    )
    ap.add_argument("--eval_root", type=str, default="logs/rl_games",
                    help="Root to search (default: logs/rl_games).")
    ap.add_argument("--hand", type=str, default=None,
                    help="Restrict to one hand family, e.g. inspire_hand (joined onto --eval_root).")
    ap.add_argument("--eval_files", type=str, nargs="*", default=None,
                    help="Explicit eval .npy files (bypasses discovery).")
    ap.add_argument("--episode", type=str, default="0",
                    help="Episode index to read, or '*' for all saved episodes (default: 0).")
    ap.add_argument("--ckpt_select", choices=("all", "final", "latest_ep", "best_reward"),
                    default="best_reward", help="Per-run checkpoint selection (default: best_reward).")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Filter to these method tags (e.g. graphcon para).")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Filter to these clip tokens / objects (substring match).")
    ap.add_argument("--env_idx", type=int, default=0, help="Env index of the rollout (default 0).")
    ap.add_argument("--align", choices=("shift", "clamp", "raw"), default="shift",
                    help="obj/demo frame pairing. shift=+1 correction (reproduces the env "
                         "reward, default); raw=SPIDER evaluate_rl_eval literal; clamp=+1 keep-all.")
    ap.add_argument("--summary", action="store_true",
                    help="Print grouped tables (rows=task, cols=method) instead of per-run.")
    ap.add_argument("--latex", action="store_true", help="Also print LaTeX tables in --summary mode.")
    ap.add_argument("--sanity", action="store_true",
                    help="Print recomputed vs saved pos/rot means to verify the port.")
    args = ap.parse_args()

    # ---- Resolve eval file records ------------------------------------------
    if args.eval_files:
        records = []
        for p in args.eval_files:
            p = os.path.abspath(p)
            ckpt_dir = os.path.basename(os.path.dirname(p))
            run_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
            rec = parse_run_dir(run_dir)
            rec.update(_parse_ckpt_dir(ckpt_dir))
            ep_m = re.search(r"eval_ep(\d+)\.npy$", p)
            rec["episode"] = int(ep_m.group(1)) if ep_m else 0
            rec["path"] = p
            records.append(rec)
    else:
        eval_root = args.eval_root
        if args.hand:
            eval_root = os.path.join(eval_root, args.hand)
        eval_root = os.path.abspath(eval_root)
        episode = "*" if args.episode == "*" else int(args.episode)
        records = discover_eval_files(eval_root, episode=episode)
        records = select_checkpoint(records, args.ckpt_select)

    if not records:
        print("No eval files found.")
        return 2

    # ---- Filters ------------------------------------------------------------
    if args.methods:
        keep = set(args.methods)
        records = [r for r in records if str(r["method"]) in keep]
    if args.tasks:
        subs = args.tasks
        records = [r for r in records
                   if any(s in (str(r["clip"]) + " " + str(r["object"])) for s in subs)]
    if not records:
        print("No eval files left after filtering.")
        return 2

    # ---- Compute metrics ----------------------------------------------------
    rows = []
    for rec in sorted(records, key=lambda r: (str(r["method"]), str(r["clip"]), r.get("ep") or -1)):
        try:
            rec["metrics"] = evaluate_eval_npy(rec["path"], env_idx=args.env_idx, align=args.align)
        except Exception as e:
            print(f"WARNING: failed to process {rec['path']}: {e}")
            continue
        rows.append(rec)

    if not rows:
        print("No valid eval files processed.")
        return 2

    # ---- Sanity cross-check -------------------------------------------------
    if args.sanity:
        print(f"\n{'='*100}")
        print(f"SANITY: recomputed (SPIDER, align={args.align}) vs saved rew_dict per-frame means")
        print("  saved arti shown as L1 = sqrt(2*saved). With align=shift all three should")
        print("  match the env reward (~1e-3); with align=raw they will differ (frame lag).")
        print(f"{'='*100}")
        hdr = f"  {'clip':<24} {'channel':<10} {'recomputed':>14} {'saved':>14} {'abs_diff':>12}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            m = r["metrics"]
            for k in ("pos_dist", "rot_dist", "arti_dist"):
                saved = m["_saved"].get(k, float("nan"))
                recomputed = m[f"{k}_mean"]
                print(f"  {str(r['clip']):<24} {k:<10} {recomputed:>14.6f} "
                      f"{saved:>14.6f} {abs(recomputed - saved):>12.6f}")
        return 0

    # ---- Output -------------------------------------------------------------
    if args.summary:
        results, methods, tasks = build_grouped(rows)
        print(f"\n{'='*70}")
        print("SPIDER-protocol summary  (rows = object clip, cols = method)")
        print(f"Methods: {methods}")
        print(f"Tasks:   {len(tasks)}  |  eval files: {len(rows)}  |  "
              f"ckpt_select={args.ckpt_select}  |  align={args.align}")
        print(f"{'='*70}")
        print_markdown_tables(results, methods, tasks)
        if args.latex:
            print(f"\n{'='*70}\nLATEX\n{'='*70}")
            print_latex_tables(results, methods, tasks)
    else:
        print_per_run_table(rows, align=args.align)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
