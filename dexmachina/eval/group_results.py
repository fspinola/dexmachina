import argparse
import json
import os
from collections import defaultdict
from os.path import join
from typing import Any, Dict, List, Optional

import numpy as np
from pathlib import Path

from dexmachina.eval.utils import (
    extract_clip,
    find_config_path,
    flatten_dict,
    infer_object_name_from_clip,
    list_eval_files,
    load_config,
    match_dict,
)


def check_and_conditions(flat_config: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
    for condition in conditions:
        key = condition["key"]
        value = condition["value"]
        if key not in flat_config:
            if key == "env_kwargs.curriculum_cfg.type":
                return False
            if "env_kwargs.object_cfgs" in key and ".actuated" in key:
                continue
            continue
        if flat_config.get(key, None) != value:
            return False
    return True


def check_or_conditions(flat_config: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
    for condition in conditions:
        key = condition["key"]
        value = condition["value"]
        if flat_config.get(key, None) == value:
            return True
    return False


def load_add_stats(eval_dir: str, stats_name: str) -> Optional[Dict[str, Any]]:
    stats_path = join(eval_dir, stats_name)
    if not os.path.exists(stats_path):
        return None
    with open(stats_path, "r") as f:
        return json.load(f)


def find_add_path(eval_path: str) -> Optional[str]:
    eval_dir = os.path.dirname(eval_path)
    direct_path = join(eval_dir, "add.npy")
    if os.path.exists(direct_path):
        return direct_path
    cfg_path = find_config_path(eval_path)
    if cfg_path:
        run_root = os.path.dirname(cfg_path)
        root_path = join(run_root, "add.npy")
        if os.path.exists(root_path):
            return root_path
        matches = list(Path(run_root).rglob("add.npy"))
        if matches:
            return str(matches[0])
    return None


def compute_auc(mean_add: np.ndarray, thresholds) -> float:
    accuracies = []
    for thres in thresholds:
        acc = np.mean(mean_add < thres)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    x_values = np.linspace(0, 1, len(thresholds))
    return float(np.trapz(accuracies, x=x_values))


def compute_add_stats(add_data: Dict[str, np.ndarray], thresholds) -> Dict[str, Dict]:
    mean_add = {part: float(np.mean(add_data[part])) for part in add_data.keys()}
    std_add = {part: float(np.std(add_data[part])) for part in add_data.keys()}
    auc = {part: compute_auc(add_data[part], thresholds) for part in add_data.keys()}
    return dict(
        mean_add={
            **mean_add,
            "overall": float(np.mean(list(mean_add.values()))),
        },
        std_add={
            **std_add,
            "overall": float(np.mean(list(std_add.values()))),
        },
        auc={
            **auc,
            "overall": float(np.mean(list(auc.values()))),
        },
        thresholds=list(thresholds),
    )


def parse_config_for_flat(config: Dict[str, Any]) -> Dict[str, Any]:
    if "env_kwargs" in config and isinstance(config["env_kwargs"], dict):
        env_kwargs = config["env_kwargs"].get("value", config["env_kwargs"])
    else:
        env_kwargs = {}
    if "params" in config and isinstance(config["params"], dict):
        params = config["params"].get("value", config["params"])
    else:
        params = {}
    clip = extract_clip(config)
    config_dict = dict(env_kwargs=env_kwargs, params=params, clip=clip)
    return flatten_dict(config_dict)


def group_results(
    eval_files: List[str],
    group_cfg: Dict[str, Any],
    stats_name: str,
    use_auc: bool,
) -> Dict[str, Any]:
    methods = group_cfg["methods"]
    hand_types = group_cfg["hand_types"]
    object_clips = group_cfg["object_clips"]
    seeds = group_cfg["seeds"]

    results = {
        "methods": list(methods.keys()),
        "method_names": [method["name"] for method in methods.values()],
        "hand_types": list(hand_types.keys()),
        "hand_names": [hand_type["name"] for hand_type in hand_types.values()],
        "objects": object_clips,
        "data": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {}))),
    }

    for eval_path in eval_files:
        cfg_path = find_config_path(eval_path)
        if not cfg_path:
            continue
        config = load_config(cfg_path)
        flat_cfg = parse_config_for_flat(config)
        if not match_dict(flat_cfg, group_cfg.get("filter_conditions", {})):
            continue
        run_seed = flat_cfg.get("params.seed", None)
        clip = flat_cfg.get("clip", None)
        if clip is None:
            continue
        obj_name = infer_object_name_from_clip(clip)
        if obj_name is None or clip not in object_clips:
            continue

        method_match = None
        for method_name, method_config in methods.items():
            and_match = check_and_conditions(flat_cfg, method_config["conditions"])
            or_match = True
            if method_config.get("or_conditions", []):
                or_match = check_or_conditions(flat_cfg, method_config["or_conditions"])
            set_match = True
            if method_config.get("condition_sets", None) is not None:
                set_match = False
                for sets in method_config["condition_sets"]:
                    set_and_match = check_and_conditions(flat_cfg, sets.get("and", []))
                    set_or_match = True
                    if sets.get("or", []):
                        set_or_match = check_or_conditions(flat_cfg, sets["or"])
                    set_kw_match = True
                    set_keywords = sets.get("keywords", [])
                    if set_keywords:
                        set_kw_match = all(kw in cfg_path for kw in set_keywords)
                    if set_and_match and set_or_match and set_kw_match:
                        set_match = True
                        break
            keywords = method_config.get("keywords", [])
            keyword_match = all(kw in cfg_path for kw in keywords) if keywords else True
            if and_match and or_match and set_match and keyword_match:
                method_match = method_name
                break

        hand_type_match = None
        for hand_name, hand_config in hand_types.items():
            if check_and_conditions(flat_cfg, hand_config["conditions"]):
                hand_type_match = hand_name
                break

        if method_match is None or hand_type_match is None:
            continue

        eval_dir = os.path.dirname(eval_path)
        stats = load_add_stats(eval_dir, stats_name)
        if stats is None:
            add_path = find_add_path(eval_path)
            if add_path:
                add_data = np.load(add_path, allow_pickle=True).item()
                thresholds = list(np.arange(0.01, 0.1, 0.01))
                stats = compute_add_stats(add_data, thresholds)
            else:
                print(f"Missing {stats_name} and add.npy for {eval_path}")
                continue

        if use_auc:
            metric = stats["auc"]["overall"]
            std = 0.0
        else:
            metric = stats["mean_add"]["overall"]
            std = stats["std_add"]["overall"]

        results["data"][method_match][hand_type_match][clip][run_seed] = {
            "success_rate": metric,
            "std": std,
            "run_path": os.path.dirname(cfg_path),
        }

    processed_results = {
        "methods": results["methods"],
        "method_names": results["method_names"],
        "hand_types": results["hand_types"],
        "hand_names": results["hand_names"],
        "objects": results["objects"],
        "avg_success_rates": {},
        "run_paths": {},
    }

    for method in results["methods"]:
        processed_results["avg_success_rates"][method] = {}
        processed_results["run_paths"][method] = {}
        for hand_type in results["hand_types"]:
            processed_results["avg_success_rates"][method][hand_type] = {}
            processed_results["run_paths"][method][hand_type] = {}
            for object_type in results["objects"]:
                seed_runs = results["data"][method][hand_type][object_type]
                print(f"{method}/{hand_type}/{object_type}: {len(seed_runs)} seeds")
                if seed_runs:
                    success_rates = [run["success_rate"] for run in seed_runs.values()]
                    processed_results["avg_success_rates"][method][hand_type][object_type] = {
                        "value": float(sum(success_rates) / len(success_rates)),
                        "err": float(np.std(success_rates) / np.sqrt(len(success_rates))),
                        "n_seeds": len(success_rates),
                        "seeds_used": list(seed_runs.keys()),
                    }
                    processed_results["run_paths"][method][hand_type][object_type] = [
                        seed_runs[s]["run_path"] for s in seed_runs
                    ]
                else:
                    processed_results["avg_success_rates"][method][hand_type][object_type] = {
                        "value": None,
                        "err": None,
                        "n_seeds": 0,
                        "seeds_used": [],
                    }
                    processed_results["run_paths"][method][hand_type][object_type] = []
    return processed_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_config", default="dexmachina_main")
    parser.add_argument("--config_path", default="dexmachina/eval/group_cfgs")
    parser.add_argument("--input", help="Eval file or root directory")
    parser.add_argument("--pattern", default="**/eval_ep*.npy")
    parser.add_argument(
        "--run_paths_json",
        default=None,
        help="Optional run_paths.json to restrict which runs to load",
    )
    parser.add_argument(
        "--run_paths_base",
        default=None,
        help="Base directory for relative paths in run_paths.json",
    )
    parser.add_argument("--stats_name", default="grouped_ADD_stats.json")
    parser.add_argument("--use_auc", action="store_true")
    parser.add_argument("--output_dir", default="dexmachina/eval/stats")
    parser.add_argument("--output_name", default="results_ADD")
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    cfg_path = args.group_config
    if not cfg_path.endswith(".yaml"):
        cfg_path += ".yaml"
    if not os.path.isabs(cfg_path):
        cfg_path = join(args.config_path, cfg_path)
    group_cfg = load_config(cfg_path)

    if args.run_paths_json:
        with open(args.run_paths_json, "r") as f:
            run_paths_data = json.load(f)
        run_paths = []
        for hand_data in run_paths_data.values():
            for task_paths in hand_data.values():
                run_paths.extend(task_paths)
        base_dir = args.run_paths_base
        if base_dir is None:
            base_dir = os.path.dirname(args.run_paths_json)
        eval_files = []
        for run_path in sorted(set(run_paths)):
            if os.path.isabs(run_path):
                run_root = run_path
            else:
                rel = run_path.lstrip("./")
                if rel.startswith("stats/wb_data/"):
                    rel = os.path.join("runs", os.path.basename(rel))
                run_root = os.path.join(base_dir, rel)
            # breakpoint()
            eval_files.extend(list_eval_files(run_root, args.pattern))
        if not eval_files:
            raise ValueError(f"No eval files found in run paths from {args.run_paths_json}")
    else:
        eval_files = list_eval_files(args.input, args.pattern)
    if not eval_files:
        raise ValueError(f"No eval files found for {args.input}")
    if args.dryrun:
        filtered = 0
        for eval_path in eval_files:
            cfg_path = find_config_path(eval_path)
            if not cfg_path:
                continue
            config = load_config(cfg_path)
            flat_cfg = parse_config_for_flat(config)
            if not match_dict(flat_cfg, group_cfg.get("filter_conditions", {})):
                continue
            clip = flat_cfg.get("clip", None)
            if clip not in group_cfg.get("object_clips", []):
                continue
            method_match = None
            for method_name, method_config in group_cfg["methods"].items():
                and_match = check_and_conditions(flat_cfg, method_config["conditions"])
                or_match = True
                if method_config.get("or_conditions", []):
                    or_match = check_or_conditions(flat_cfg, method_config["or_conditions"])
                set_match = True
                if method_config.get("condition_sets", None) is not None:
                    set_match = False
                    for sets in method_config["condition_sets"]:
                        set_and_match = check_and_conditions(flat_cfg, sets.get("and", []))
                        set_or_match = True
                        if sets.get("or", []):
                            set_or_match = check_or_conditions(flat_cfg, sets["or"])
                        set_kw_match = True
                        set_keywords = sets.get("keywords", [])
                        if set_keywords:
                            set_kw_match = all(kw in cfg_path for kw in set_keywords)
                        if set_and_match and set_or_match and set_kw_match:
                            set_match = True
                            break
                keywords = method_config.get("keywords", [])
                keyword_match = all(kw in cfg_path for kw in keywords) if keywords else True
                if and_match and or_match and set_match and keyword_match:
                    method_match = method_name
                    break
            hand_match = None
            for hand_name, hand_config in group_cfg["hand_types"].items():
                if check_and_conditions(flat_cfg, hand_config["conditions"]):
                    hand_match = hand_name
                    break
            if method_match and hand_match:
                filtered += 1
        print(f"Filtered eval files: {filtered}")
        return

    processed = group_results(
        eval_files=eval_files,
        group_cfg=group_cfg,
        stats_name=args.stats_name,
        use_auc=args.use_auc,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_AUC" if args.use_auc else ""
    out_path = join(args.output_dir, f"{args.output_name}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
