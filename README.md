# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation 

[Mandi Zhao](https://mandizhao.github.io), [Yifan Hou](https://yifan-hou.github.io), [Dieter Fox](https://homes.cs.washington.edu/~fox), [Yashraj Narang](https://research.nvidia.com/person/yashraj-narang), [Shuran Song*](https://shurans.github.io), [Ajay Mandlekar*](https://ai.stanford.edu/~amandlek)

*Equal Advising

[arXiv](http://arxiv.org/abs/2505.24853) | [Project Website](https://project-dexmachina.github.io) | [Code Documentation](https://mandizhao.github.io/dexmachina-docs) 

![Teaser](dexmachina-teaser-website.png)

## Code Release Status 
- 06/11/2025: 
Released all dexterous hand assets and ARCTIC assets used in our recent [arXiv preprint](http://arxiv.org/abs/2505.24853). Released detailed instructions for processing new hand assets: see code in `dexmachina/hand_proc` and [hand processing doc page](https://mandizhao.github.io/dexmachina-docs/1_process_hands.html). Pushed a new `dexmachina.yaml` file for conda env install. RL training example in `examples/train_rl.sh`
- 06/03/2025: Initial Release


TODOs 
- [ ] Advanced rendering code
- [ ] RL eval code
- [x] Instructions for processing new hands and demonstrations 

## Installation
 
1. We recommend using conda environment with Python=3.10
```
conda create -n dexmachina python=3.10
conda activate dexmachina
```
2. Clone and install the below custom forks of Genesis and rl-games:

```
pip install torch==2.5.1
git clone https://github.com/MandiZhao/Genesis.git
cd Genesis
pip install -e .
pip install libigl==2.5.1 # NOTE: this is a temporary fix specifically for my fork of Genesis

git clone https://github.com/MandiZhao/rl_games.git
cd rl_games
pip install -e .
```
Additional packages needed for RL training:
```
pip install gymnasium ray seaborn wandb trimesh
# an old version of moviepy
pip install moviepy==1.0.3
```

**If you'd like to install the full conda environment that includes all the packages, use the below yaml file:**
```
# this is obtained from: conda export -f dexmachina.yaml
conda env create -f dexmachina.yaml
```
4. Local install the `dexmachina` package:
```
cd dexmachina
pip install -e .
```

See the full [documentation](https://mandizhao.github.io/dexmachina-docs) for additional installation instructions for dexterous hand and demonstration data processing, kinematic retargeting, raytracer rendering, etc. 

[for newer versions of torch] Patch in /home/fspinola/venvs/dexmachina-venv2/lib/python3.10/site-packages/rl_games/algos_torch/torch_ext.py:
```
def safe_load(filename):
    return safe_filesystem_op(lambda f: torch.load(f, weights_only=False), filename)
```

## Graph Retargeting as Kinematic References (custom)

This fork can drive RL training from **graph-SLSQP retargeting** outputs (from the
`learned_retargeter`/GeoRT pipeline) instead of the paper's original references. The
converted references live alongside the originals and are selected at train time with
`--retarget_name graph` — the `_para` baseline is left untouched.

### What a reference contains

DexMachina loads object/contact `demo_data` separately from `arctic/processed/`, so a
reference `.pt` only needs the **hand** `retarget_data['left'/'right']`:

- `joint_qpos` — dict `{actuated_joint_name -> (N,) tensor}` over the 6-DoF floating wrist
  (`*_forearm_tx/ty/tz` prismatic + `roll/pitch/yaw` revolute) plus the finger joints.
  Mimic joints are auto-derived by the env and are **not** included.
- `kpt_pos (N, K, 3)` + `kpt_names` — keypoint world positions for the imitation reward.
  Must match the `_para` baseline set exactly (25 links for Allegro, 18 for Inspire,
  including `base_link`): the env overwrites cfg keypoints with these and indexes
  `wrist_link_idx` into them.
- `wrist_pose (N, 7)` — `[x, y, z, qw, qx, qy, qz]` (quaternion **wxyz**) of the wrist link
  (`base_dummy_link` for Allegro, `base_link` for Inspire).

### Coordinate conventions

The graph retargeter emits, per frame, finger `qpos` plus a `wrist_xi (6,)` axis-angle pose.
`wrist_xi` encodes the retargeting URDF's **`base_link`** pose in the per-frame *canonical*
(wrist-centered) frame — verified by a fingertip-matching test (the "base_link" hypothesis
wins at ~6.5 mm vs ~67/97 mm for the palm/wrist alternatives). The world pose is then:

```
T_world_baselink  = hand_wrist_T_world[t] @ se3(wrist_xi[t])      # ARCTIC raw npz; same frame as obj_T_world
T_world_dexbase   = T_world_baselink @ T_mybase_dexbase           # per-side base-frame alignment (see below)
T_world_wristlink = T_world_dexbase @ Trans(0,0,z)               # z = -0.095 (Allegro base_dummy), 0 (Inspire)
(tx,ty,tz,roll,pitch,yaw) = decompose(T_world_wristlink)          # Trans(tx,ty,tz) @ Rz(roll) @ Rx(pitch) @ Ry(-yaw)
```

The 6-DoF forearm chain is a ZXY-intrinsic Euler (yaw axis negated); decomposition is checked
with an FK round-trip assert (≤1e-7). `kpt_pos`/`wrist_pose` are then obtained by FK-ing the
6-DoF URDF (`yourdfpy`) at the full qpos.

**Per-side base alignment.** The retargeting URDF and DexMachina's 6-DoF URDF can use different
`base_link` conventions (notably Inspire, ~[-91, 0, 178]°). The converter computes a constant
`T_mybase_dexbase` via Procrustes on matched-qpos fingertips and applies it when the fit is rigid
(residual ≤ 10 mm); otherwise it falls back to identity. Allegro RH/LH align at identity; Inspire
at ~1.9 mm.

**Allegro-left joint convention.** DexMachina's `allegro_hand_left_6dof.urdf` is the *same physical
left hand* as the retargeting URDF, but labels index↔ring oppositely and uses the opposite
abduction-axis sign (verified by FK: the four fingers match to 0 mm under that remap; only a ~10 mm
thumb-mount offset remains). The converter applies this `{dex_joint: (my_joint, sign)}` remap for
Allegro-left (`finger_convention_remap`), so left-hand fingertip fidelity matches the right
(~7–10 mm). No collision-config change is needed. Separately, the ~10 mm thumb offset was a genuine
incomplete-mirror bug in `allegro_hand_left_6dof.urdf` (`joint_13` had its axis mirrored but not its
origin `y`); fixed by one number (`0.005` → `-0.005`), so the thumb now mirrors correctly too.

### Producing references

The converter is `learned_retargeter/kinematic/export_dexmachina_kinref.py` (in the
`learned_retargeter` repo). One clip, both hands:

```
python -m learned_retargeter.kinematic.export_dexmachina_kinref \
  --hand allegro_hand \
  --retarget-left  <LH retarget_outputs/<seq>.npz> \
  --retarget-right <RH retarget_outputs/<seq>.npz> \
  --processed-npy  dexmachina/assets/arctic/processed/s01/box_use_01.npy \
  --out            dexmachina/assets/retargeted/allegro_hand/s01/box_use_01_vector_graph.pt \
  --dexmachina-root <path to this repo>
```

`scripts/export_dexmachina_kinrefs.sh` batches the 6 paper source sequences × {allegro, inspire}
(`s01:{box,ketchup:01,ketchup:02,mixer,waffleiron}`, `s02:notebook:02`). Frames align 1:1 with
`arctic/processed` (full-length, indexed from 0); the `--clip` range crops both at train time.
References shipped under `assets/retargeted/{allegro_hand,inspire_hand}/{s01,s02}/` as
`*_vector_graph.pt`.

### Training on the graph references

`training_scripts/train_array_graph.slurm` runs the paper hybrid recipe over the 7 paper
experiments × 2 hands (14 tasks in `training_scripts/runs_graph.tsv`): `ketchup-100`, `box-200`,
`mixer-170`, `ketchup-300`, `mixer-300`, `notebook-300`, `waffleiron-300`.

```
mkdir -p slurm_logs
sbatch --array=1 training_scripts/train_array_graph.slurm   # smoke-test task 1
sbatch training_scripts/train_array_graph.slurm             # full sweep, contact OFF (exp graphnc)
sbatch training_scripts/train_array_graph_contact.slurm     # full sweep, contact ON -con 3 (exp graphcon)
```

Single clip directly:

```
python dexmachina/rl/train_rl_games.py -B 12000 --hand allegro_hand \
  --clip box-30-230 --retarget_name graph --actuate_object -am hybrid \
  -imi 0.3 -bc 0.3 -imw 0.5 -con 0 ...   # contact OFF: no --use_retarget_contact, -con 0
```

**Contact reward (now supported for graph refs).** The contact targets in `contact_retarget/*`
are the ARCTIC human contact points grouped onto the *nearest robot collision-link AABB center*
per frame — an assignment that depends on the retargeted hand poses, so the shipped para-based
files are stale for graph poses (~25% of link assignments shift). They have been regenerated from
the graph references with the same `map_contacts.py` pipeline (only the pose source differs):

```
python dexmachina/retargeting/map_contacts.py --hand allegro_hand \
  --load_fname dexmachina/assets/arctic/processed/s01/box_use_01.npy \
  --retarget_pt dexmachina/assets/retargeted/allegro_hand/s01/box_use_01_vector_graph.pt \
  --save_suffix _graph
```

`--retarget_pt` poses the hands from the reference `.pt` (URDF mimic joints filled from the mimic
tags); `--save_suffix` writes e.g. `box_use_01_graph.npy` next to the original. The loader prefers
`{obj}_use_{clip}_{retarget_name}.npy` and falls back to the unsuffixed para file, so
`--retarget_name para` behavior is unchanged. Shipped: `contact_retarget/{allegro,inspire}_hand/`
`_graph.npy` for the 6 paper source sequences, full-length, same collision link names/order as the
para baseline (the matched-contact reward asserts this against the env's collision links). Enable
the full paper recipe with `--use_retarget_contact -con 3 --retarget_name graph`.

### ff_residual action mode

`-am ff_residual` is `residual` anchored on the **next** reference frame: the action base is
`residual_qpos[k+1]` (matching the t+1 reward target, so zero action tracks the reference), the
policy additionally observes the next-frame reference error `ref[k+1] - dof_pos`, and episodes
reset to `ref[start]`. Scaling is residual's limit-based mapping (`a=±1` reaches the joint limits —
in-range by construction, no extra knobs); `--res_cap` optionally bounds the wrist residual to
`±hybrid_scales` (a one-sided sign bug in `res_cap` was fixed in this fork). In contrast,
`residual` centers on `ref[k]` and `hybrid` uses absolute finger actions, so `ff_residual` is the
ManipTrans-style feed-forward-residual parameterization. Note the extra observation block changes
the obs dim: `ff_residual` checkpoints are not compatible with other modes.

### Notes / limitations

- All action modes (`kinematic`, `residual`, `hybrid`, `absolute`, `ff_residual`) load and consume
  the graph references; validated by headless kinematic playback (correct bimanual grasp cycles).
  The contact reward is orthogonal to the action mode (gated only by `-con > 0`), so it combines
  with any of them, including `ff_residual`.
- Allegro LH fingertip fidelity matches RH (~6–10 mm) via the `finger_convention_remap` plus the
  one-line thumb-origin mirror fix in `allegro_hand_left_6dof.urdf`.
- Inspire references sit ~13–25 mm from the human. This is mostly **inherent retargeting quality**
  (Inspire is an underactuated 6-DoF hand, so it matches the human less closely than the 16-DoF
  Allegro) — the mimic (joint-coupling) ratio difference between the two URDFs adds only ~2 mm (the
  base-alignment residual). The reference is self-consistent and exactly reachable by the simulated
  hand, so RL is unaffected. The mimic ratios model the real Inspire hardware, so they are left
  as-is; native re-retargeting on DexMachina's URDF would shave off only that ~2 mm.

## Citation
This codebase is released with the following preprint:

Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar*, Shuran Song*. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation. arXiV, 2025.

*Equal Advising 

If you find this codebase useful, please consider citing:
```
@misc{mandi2025dexmachinafunctionalretargetingbimanual,
      title={DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation}, 
      author={Zhao Mandi and Yifan Hou and Dieter Fox and Yashraj Narang and Ajay Mandlekar and Shuran Song},
      year={2025},
      eprint={2505.24853},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.24853}, 
}
```
