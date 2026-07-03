# Kinematic-Reference (kinref) FPS & Frame-Sync Notes

**Investigated:** 2026-06-15
**Question:** ARCTIC mocap is 30 Hz. Are my retargeted kinrefs 30 Hz or 15 Hz? What
reference FPS does dexmachina assume, and is there a frame-sync mismatch if I
provide fewer reference frames?

**TL;DR:** Verified — the `*_vector_graph.pt` kinrefs are at ARCTIC's native **30 Hz,
frame-matched 1:1** with the ARCTIC object trajectory for every trained clip. dexmachina
does **no** reference resampling: it plays exactly one reference frame per `dt = 1/60`
control step. There is **no object↔hand desync**. The only thing to be aware of is that a
30 Hz reference replayed at the 60 Hz sim rate runs at ~2× wall-clock speed — this is
inherent to dexmachina's design (not a bug introduced by the retargeting), and it does not
break synchronization.

---

## 1. How kinrefs are integrated (data flow)

Two **separate** reference sources are merged frame-by-frame:

| Reference | Source file | Loader |
|---|---|---|
| Object pose (`obj_pos`, `obj_quat`, `obj_arti`) | `assets/arctic/processed/<subj>/<obj>_use_<clip>.npy` (raw ARCTIC) | `get_demo_data()` — `dexmachina/envs/demo_data.py:12-43` |
| Hand (`wrist_pose`, `kpt_pos`, `residual_qpos`) | `assets/retargeted/<hand>/<subj>/<obj>_use_<clip>_vector_graph.pt` (your retargeting) | `load_genesis_retarget_data()` — `dexmachina/envs/demo_data.py:60-132` |

Pipeline:

1. `make_env`/constructor calls both loaders with the **same** `frame_start`/`frame_end`
   (`dexmachina/envs/constructors.py:69-87`). The retarget loader's own `demo_data` is
   discarded (`_, retarget_data = ...`); the object always comes from ARCTIC.
2. `episode_length = frame_end - frame_start` (`constructors.py:89`).
3. `RewardModule.load_demo()` stacks object + hand tensors into `self.demo_tensors` and
   **asserts every tensor shares the same first-dim length** as `obj_pos`
   (`dexmachina/envs/rewards.py:101-136`, assert at **rewards.py:132-134**).
4. At runtime, `episode_length_buf` increments by 1 each env step
   (`dexmachina/envs/base_env.py:587`) and indexes the reference via
   `match_demo_state()` (`dexmachina/envs/rewards.py:141-146`). Residual/ff action
   targets are indexed the same way in `dexmachina/envs/robot.py`.

**→ One reference frame is consumed per control step. No stride, no interpolation.**

## 2. dexmachina's reference timing / FPS

- Physics & control timestep: `dt = 1/60 s` — `get_scene_cfg()` in
  `dexmachina/envs/base_env.py:25-26` (`substeps=2`), confirmed in run configs
  (`logs/.../params/env.yaml` → `dt: 0.016666666666666666`).
- Reference advances 1 frame/step ⇒ **effective playback = 60 frames per second of sim time.**
- The retargeter's `fps=30` (`dexmachina/retargeting/learned_retargeter_pkg/retargeter.py:169`)
  is used **only** to scale finite-difference velocities during preprocessing — it does
  **not** set playback rate.
- An upsampling hook exists (`--interp` → `interpolate_demo_states`) but is **explicitly
  disabled for retarget data**: `dexmachina/envs/constructors.py:89-91` raises
  `NotImplementedError` if `interp > 1`. No training/eval script passes `--interp`.

## 3. Verification — are the kinrefs 30 Hz (not 15 Hz)?

torch can't load in the analysis sandbox, so frame counts were read directly from the
`.pt` zip archives two independent ways:
- **Shape extraction** via a custom unpickler (`_rebuild_tensor_v2` size args + numpy
  `__setstate__` shapes).
- **Raw storage byte counts** (interpretation-independent): e.g. box `joint_qpos` storage =
  3556 bytes = 889 × 4 (float32) ⇒ 889 frames.

Both agree. Batch result over all trained clips (`training_scripts/runs_graph.tsv`),
loaded with `--retarget_name graph`:

| Clip | Frame range | ARCTIC | allegro `graph` | inspire `graph` | Match |
|---|---|---|---|---|---|
| ketchup-30-130-s01-u01 | 30–130 | 697 | 697 | 697 | ✅ |
| box-30-230-s01-u01 | 30–230 | 889 | 889 | 889 | ✅ |
| mixer-30-200-s01-u01 | 30–200 | 638 | 638 | 638 | ✅ |
| ketchup-40-340-s01-u02 | 40–340 | 769 | 769 | 769 | ✅ |
| mixer-40-340-s01-u01 | 40–340 | 638 | 638 | 638 | ✅ |
| notebook-40-340-s02-u02 | 40–340 | 706 | 706 | 706 | ✅ |
| waffleiron-40-340-s01-u01 | 40–340 | 605 | 605 | 605 | ✅ |

Every kinref equals its ARCTIC object trajectory length (and both hands agree), and every
trained `frame_end` fits within the file. **The kinrefs are at ARCTIC's native 30 Hz,
1:1 — not downsampled to 15 Hz.** (If they had been 15 Hz, the hand tensor would be ~half
the object length after slicing and the rewards.py:132 assert would have crashed the run.)

> Note: the file is the full clip (e.g. 889 frames); a run uses a slice of it
> (`box-30-230` → 200 frames). Both object and hand are sliced identically, keeping them
> frame-locked.

## 4. Is there a problem?

- **Object ↔ hand sync: none.** Same length, same slice, asserted equal, advanced together.
- **30 Hz ref @ 60 Hz sim = ~2× wall-clock playback.** A 200-frame clip plays in ~3.3 s of
  sim time instead of the real ~6.7 s. This is **inherent to dexmachina** (interp disabled),
  applies equally to object and hand, and was the regime the method was tuned in — not a
  defect of the retargeting.
- **Built-in guard:** mismatched-length kinrefs **crash** at load (rewards.py:132); they do
  not silently desync.

## 5. Guidance: providing fewer / lower-rate kinrefs

- You **cannot just "provide fewer" frames.** There is no resampling path; fewer hand frames
  than the ARCTIC object slice → assertion crash.
- To use a 15 Hz (or any sub-30 Hz) source, **upsample it to exactly one pose per ARCTIC
  frame** (same indexing as `arctic/processed/<obj>_use_<clip>.npy`) before saving the `.pt`.
- Alternatively you could downsample the ARCTIC object data and shrink `episode_length` to
  match — but then both play even faster relative to wall-clock, and you'd be off
  dexmachina's tuned regime.
- **Watch out for the `_para` / `_learned_qsup` baseline files: they are 600 frames** (≠ the
  889-frame ARCTIC box clip). They are **not** used by the `graph`/`graphcon` runs. If you
  ever train/eval with them against ARCTIC object data, confirm the 600 is a head-trim (still
  aligned) and not a resample (silent temporal offset).
