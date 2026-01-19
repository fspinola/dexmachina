"""Offline aggregator for DexMachina part-wise ADD-AUC.

This script loads `eval_ep*.npy` files produced by `dexmachina/rl/eval_rl_games.py`
(after the part-ADD patch) and prints mean/std ADD-AUC.

It is intentionally lightweight and does not require Genesis.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List

import numpy as np

from dexmachina.rl.part_add_metrics import PartAddConfig, summarize_add_auc


def load_episode(fname: str) -> Dict[str, Any]:
    data = np.load(fname, allow_pickle=True).item()
    if "part_add_mean" not in data:
        raise KeyError(
            f"{fname} does not contain 'part_add_mean'. "
            "Re-run eval with the updated eval_rl_games.py that computes part ADD."
        )
    # ensure 1D
    data["part_add_mean"] = np.asarray(data["part_add_mean"]).reshape(-1)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", dest="glob_pat", type=str, required=True, help="Glob for eval_ep*.npy")
    parser.add_argument("--auc_thresh", type=float, default=0.03)
    parser.add_argument("--auc_step", type=float, default=0.001)
    args = parser.parse_args()

    fnames = sorted(glob.glob(args.glob_pat))
    if not fnames:
        raise SystemExit(f"No files matched: {args.glob_pat}")

    episodes = [load_episode(f) for f in fnames]
    cfg = PartAddConfig(auc_threshold_m=args.auc_thresh, auc_step_m=args.auc_step)
    out = summarize_add_auc(episodes, cfg=cfg)

    print(f"Files: {len(fnames)}")
    print(f"ADD-AUC mean: {out['add_auc_mean']:.4f}")
    print(f"ADD-AUC std:  {out['add_auc_std']:.4f}")


if __name__ == "__main__":
    main()
