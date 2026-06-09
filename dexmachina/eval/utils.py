import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def match_dict(flat_dict: Dict[str, Any], filter_conditions: Dict[str, Any]) -> bool:
    for key_path, condition in filter_conditions.items():
        if key_path.startswith("_"):
            continue
        try:
            value = flat_dict.get(key_path, None)
            if value is None:
                raise KeyError(key_path)
            if isinstance(condition, list):
                if value not in condition:
                    return False
            elif isinstance(condition, str) and ("*" in condition or "?" in condition):
                if not fnmatch.fnmatch(str(value), condition):
                    return False
            elif isinstance(condition, dict) and "_range" in condition:
                range_min, range_max = condition["_range"]
                if not (range_min <= value <= range_max):
                    return False
            else:
                if value != condition:
                    return False
        except (KeyError, AttributeError):
            return False
    return True


def find_config_path(start_path: str) -> Optional[str]:
    path = Path(start_path).resolve()
    if path.is_file():
        path = path.parent
    for parent in [path] + list(path.parents):
        cfg = parent / "config.yaml"
        if cfg.exists():
            return str(cfg)
    return None


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_clip(config: Dict[str, Any]) -> Optional[str]:
    clip = config.get("clip", None)
    if isinstance(clip, dict):
        clip = clip.get("value", None)
    if isinstance(clip, str):
        return clip
    params = config.get("params", {})
    if isinstance(params, dict):
        params_val = params.get("value", params)
        exp = (
            params_val.get("config", {})
            .get("full_experiment_name", None)
        )
        if isinstance(exp, str):
            for snippet in exp.split("_"):
                if re.search(r"[a-zA-Z]+\\d", snippet):
                    return snippet
    return None


def infer_object_name_from_clip(clip: str) -> Optional[str]:
    if not clip:
        return None
    match = re.match(r"([a-zA-Z]+)", clip)
    return match.group(1) if match else None


def list_eval_files(input_path: str, pattern: str) -> List[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(p) for p in path.glob(pattern)]
    return []


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
