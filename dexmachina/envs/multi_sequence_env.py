"""Multi-sequence wrapper environment for training a *single* policy.

Why this exists
---------------
The stock DexMachina RL pipeline bakes a single ARCTIC clip into `BaseEnv` by
loading `demo_data` + `retarget_data` in `get_all_env_cfg(args, ...)`.
This makes the environment (and thus the policy) sequence-specific.

To train one policy that generalizes across sequences, we want the environment
instances to see *different* clips over training.

Design
------
This wrapper holds a `BaseEnv` internally and swaps its demonstration clip
whenever some envs reset.

Important constraints / assumptions
----------------------------------
- We treat each (obj, subject, use_clip, frame_start/end) as a "task id".
- All clips used together should have compatible observation and action
  dimensions. (They do in DexMachina since the embodiment is fixed.)
- Episode length varies with clip length. RL-Games expects a fixed horizon
  length for rollout collection, but the env can terminate earlier.

Implementation choices
----------------------
- We keep a *single* maximum episode length (max over all clips), and for
  shorter clips we terminate early when `episode_length_buf >= clip_len`.
- We add a small one-hot task id (or integer id) to observations, so the policy
  can disambiguate tasks if needed. (Can be disabled.)

This file is meant to be additive: it doesn't modify existing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
from dexmachina.rl.sequence_sampler import ClipSampler, ClipSpec


@dataclass
class MultiSequenceEnvCfg:
    clips: Sequence[ClipSpec]
    seed: int = 0
    add_task_id_to_obs: bool = True
    task_id_mode: str = "onehot"  # "onehot" or "int"


class MultiSequenceEnv:
    """A thin wrapper exposing the same API as `BaseEnv`.

    It delegates almost everything to an internal `BaseEnv` instance.
    On resets, it can resample a clip and rebuild the internal `BaseEnv`.

    Note: Rebuilding is heavier than swapping buffers, but it's the safest
    approach without invasive changes to `RewardModule` and per-robot caches.
    """

    def __init__(
        self,
        *,
        base_env_kwargs: dict,
        multi_cfg: MultiSequenceEnvCfg,
        device: torch.device,
    ):
        self._device = device
        self._base_env_kwargs_template = dict(base_env_kwargs)
        self._multi_cfg = multi_cfg
        self._sampler = ClipSampler(list(multi_cfg.clips), seed=multi_cfg.seed)

        # task identity state
        self._num_tasks = len(self._sampler.clips)
        self._task_ids = torch.zeros(
            int(base_env_kwargs["env_cfg"]["num_envs"]), device=self._device, dtype=torch.int64
        )
        self._task_clip_lens = torch.zeros_like(self._task_ids, dtype=torch.int32)

        # build first env
        first = self._sampler.sample()
        self._build_env_for_clip(first, env_selector=None)

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
        if not self._multi_cfg.add_task_id_to_obs:
            return base
        if self._multi_cfg.task_id_mode == "onehot":
            return base + self._num_tasks
        if self._multi_cfg.task_id_mode == "int":
            return base + 1
        raise ValueError(f"Invalid task_id_mode: {self._multi_cfg.task_id_mode}")

    @property
    def num_actions(self):
        return self._env.num_actions

    @property
    def is_finite_horizon(self):
        return self._env.is_finite_horizon

    @property
    def unwrapped(self):
        # match existing wrapper expectations
        return self

    # --- core API expected by `RlGamesVecEnvWrapper` and existing eval scripts ---

    def reset(self):
        obs, info = self._env.reset()
        return self._augment_obs(obs), info

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self._env.step(actions)

        # enforce per-env clip-length termination (for shorter clips)
        # This assumes BaseEnv increments `episode_length_buf`.
        # We use env's internal buffer via the delegated attribute.
        env_step = self._env.episode_length_buf
        early_done = env_step >= self._task_clip_lens
        if early_done.any():
            terminated = terminated | early_done

        # on termination, resample tasks for those envs and rebuild env
        # For now, rebuild the entire env when *any* env resets.
        # This keeps code simple and avoids partial buffer swaps.
        # If you want higher throughput, see notes in the plan.
        if terminated.any() or truncated.any():
            # pick a new clip globally (coarse-grained)
            new_clip = self._sampler.sample()
            self._build_env_for_clip(new_clip, env_selector=None)

            # after rebuild, the env is already in post-reset state, so we overwrite outputs
            obs, info = self._env.reset()
            extras = dict(extras)
            extras["episode"] = extras.get("episode", {})
            extras["episode"]["task_switched"] = torch.ones_like(rew)
            return self._augment_obs(obs), rew, terminated, truncated, extras

        return self._augment_obs(obs), rew, terminated, truncated, extras

    def close(self):
        return self._env.close()

    # ---- helpers ----

    def _build_env_for_clip(self, clip: ClipSpec, env_selector: Optional[torch.Tensor]):
        """(Re)build the internal BaseEnv with demo/retarget data for `clip`.

        env_selector is reserved for future partial swapping; currently unused.
        """

        # load data for this clip
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

        # update env kwargs
        env_kwargs = dict(self._base_env_kwargs_template)
        env_kwargs["demo_data"] = demo_data
        env_kwargs["retarget_data"] = retarget_data

        # episode length: set to max clip length among all tasks for stability
        clip_len = int(clip.frame_end - clip.frame_start)
        max_len = max(int(c.frame_end - c.frame_start) for c in self._sampler.clips)
        env_cfg = dict(env_kwargs["env_cfg"])
        env_cfg["episode_length"] = max_len
        env_kwargs["env_cfg"] = env_cfg

        # build
        self._env = BaseEnv(**env_kwargs)

        # update task metadata buffers
        # (global task id: index in clip list)
        clip_list = list(self._sampler.clips)
        task_id = clip_list.index(clip)
        self._task_ids[:] = int(task_id)
        self._task_clip_lens[:] = int(clip_len)

    def _augment_obs(self, obs):
        if not self._multi_cfg.add_task_id_to_obs:
            return obs

        # env returns dict when use_rl_games=True
        if isinstance(obs, dict):
            base = obs["obs"]
        else:
            base = obs

        if self._multi_cfg.task_id_mode == "int":
            tid = self._task_ids.to(dtype=base.dtype)[:, None]
        elif self._multi_cfg.task_id_mode == "onehot":
            tid = torch.zeros((base.shape[0], self._num_tasks), device=base.device, dtype=base.dtype)
            tid.scatter_(1, self._task_ids[:, None], 1.0)
        else:
            raise ValueError(f"Invalid task_id_mode: {self._multi_cfg.task_id_mode}")

        aug = torch.cat([base, tid], dim=-1)
        if isinstance(obs, dict):
            out = dict(obs)
            out["obs"] = aug
            return out
        return aug
