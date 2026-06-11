"""Inference-friendly multi-sequence environment.

This is a small, additive companion to `dexmachina/envs/multi_sequence_env.py`.

Goal
----
At inference/deployment time, you usually *know* which object/task you want (e.g.
"grasp box"). You don't want the env to resample tasks.

This wrapper:
- lets you select a task by object name (recommended) or by explicit clip
- appends the same task-id encoding to observations as used during training
- keeps the task fixed for the whole run

It rebuilds the underlying `BaseEnv` once when the task is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
from dexmachina.rl.sequence_sampler import ClipSpec


@dataclass
class InferenceTask:
    """A deploy-time task selector.

    You can either:
    - specify `obj_name` (object-conditioned policy)
    - or specify full `clip` (sequence-conditioned policy)

    If `obj_name` is used, we pick the first clip matching that object.
    """

    obj_name: Optional[str] = None
    clip: Optional[ClipSpec] = None


@dataclass
class MultiSequenceInferenceCfg:
    clips: Sequence[ClipSpec]
    add_task_id_to_obs: bool = True
    task_id_mode: str = "onehot"  # "onehot" or "int"
    task_id_key: str = "object"  # "object" or "clip"


class MultiSequenceInferenceEnv:
    """A fixed-task env exposing the same API as `BaseEnv`.

    This is intended for deployment/evaluation where the user chooses the object.
    """

    def __init__(self, *, base_env_kwargs: dict, cfg: MultiSequenceInferenceCfg, device: torch.device):
        self._device = device
        self._base_env_kwargs_template = dict(base_env_kwargs)
        self._cfg = cfg
        self._clips = list(cfg.clips)
        if not self._clips:
            raise ValueError("cfg.clips must be non-empty")

        # Build task dictionaries
        if cfg.task_id_key == "object":
            self._task_names = sorted({c.obj_name for c in self._clips})
            self._task_index = {name: i for i, name in enumerate(self._task_names)}
        elif cfg.task_id_key == "clip":
            self._task_names = [c.clip_str for c in self._clips]
            self._task_index = {name: i for i, name in enumerate(self._task_names)}
        else:
            raise ValueError(f"Invalid task_id_key: {cfg.task_id_key}")

        self._num_tasks = len(self._task_names)
        self._task_id = 0
        self._task_clip_len = 0
        self._env: Optional[BaseEnv] = None

        # default: load first clip so spaces are defined
        self.set_task(InferenceTask(clip=self._clips[0]))

    # --- delegate common attributes used by RL-Games wrapper ---

    @property
    def device(self):
        return self._env.device

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def obs_dim(self):
        base = self._env.obs_dim
        if not self._cfg.add_task_id_to_obs:
            return base
        if self._cfg.task_id_mode == "onehot":
            return base + self._num_tasks
        if self._cfg.task_id_mode == "int":
            return base + 1
        raise ValueError(f"Invalid task_id_mode: {self._cfg.task_id_mode}")

    @property
    def num_actions(self):
        return self._env.num_actions

    @property
    def is_finite_horizon(self):
        return self._env.is_finite_horizon

    @property
    def unwrapped(self):
        return self

    # --- user API ---

    def list_tasks(self) -> Sequence[str]:
        """Return available task names (objects or clips)."""
        return tuple(self._task_names)

    def set_task(self, task: InferenceTask):
        """Set the active task and rebuild the underlying env."""
        clip = None
        if task.clip is not None:
            clip = task.clip
        elif task.obj_name is not None:
            # choose first clip matching this object
            for c in self._clips:
                if c.obj_name == task.obj_name:
                    clip = c
                    break
            if clip is None:
                raise ValueError(f"No clip found for object: {task.obj_name}")
        else:
            raise ValueError("InferenceTask must set either obj_name or clip")

        self._build_env_for_clip(clip)

        # compute task_id
        if self._cfg.task_id_key == "object":
            self._task_id = int(self._task_index[clip.obj_name])
        else:
            self._task_id = int(self._task_index[clip.clip_str])

        self._task_clip_len = int(clip.frame_end - clip.frame_start)

    # --- core env API ---

    def reset(self):
        obs, info = self._env.reset()
        return self._augment_obs(obs), info

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self._env.step(actions)

        # If you set episode_length to max clip length, still terminate early for shorter clips.
        env_step = self._env.episode_length_buf
        early_done = env_step >= int(self._task_clip_len)
        if early_done.any():
            terminated = terminated | early_done

        return self._augment_obs(obs), rew, terminated, truncated, extras

    def close(self):
        return self._env.close()

    # --- helpers ---

    def _build_env_for_clip(self, clip: ClipSpec):
        demo_data = get_demo_data(
            obj_name=clip.obj_name,
            frame_start=clip.frame_start,
            frame_end=clip.frame_end,
            hand_name=self._base_env_kwargs_template["robot_cfgs"]["left"]["name"],
            subject_name=clip.subject,
            use_clip=clip.use_clip,
            load_retarget_contact=self._base_env_kwargs_template["reward_cfg"].get("use_retarget_contact", False),
        )
        _, retarget_data = load_genesis_retarget_data(
            obj_name=clip.obj_name,
            hand_name=self._base_env_kwargs_template["robot_cfgs"]["left"]["name"],
            frame_start=clip.frame_start,
            frame_end=clip.frame_end,
            save_name=self._base_env_kwargs_template.get("retarget_name", "genesis"),
            use_clip=clip.use_clip,
            subject_name=clip.subject,
        )

        env_kwargs = dict(self._base_env_kwargs_template)
        env_kwargs["demo_data"] = demo_data
        env_kwargs["retarget_data"] = retarget_data

        # Keep a stable episode length across tasks (max over provided clips)
        max_len = max(int(c.frame_end - c.frame_start) for c in self._clips)
        env_cfg = dict(env_kwargs["env_cfg"])
        env_cfg["episode_length"] = max_len
        env_kwargs["env_cfg"] = env_cfg

        self._env = BaseEnv(**env_kwargs)

    def _augment_obs(self, obs):
        if not self._cfg.add_task_id_to_obs:
            return obs

        if isinstance(obs, dict):
            base = obs["obs"]
        else:
            base = obs

        if self._cfg.task_id_mode == "int":
            tid = torch.full((base.shape[0], 1), float(self._task_id), device=base.device, dtype=base.dtype)
        elif self._cfg.task_id_mode == "onehot":
            tid = torch.zeros((base.shape[0], self._num_tasks), device=base.device, dtype=base.dtype)
            tid[:, self._task_id] = 1.0
        else:
            raise ValueError(f"Invalid task_id_mode: {self._cfg.task_id_mode}")

        aug = torch.cat([base, tid], dim=-1)
        if isinstance(obs, dict):
            out = dict(obs)
            out["obs"] = aug
            return out
        return aug
