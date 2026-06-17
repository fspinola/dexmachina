import os 
import sys 
import torch  
import numpy as np 
from os.path import join
from dexmachina.asset_utils import get_asset_path

ARCTIC_PROCESSED_DIR = get_asset_path("arctic/processed")
RETARGET_DIR = get_asset_path("retargeted")
RETARGET_CONTACT_DIR= get_asset_path("contact_retarget")

def get_demo_data(
    obj_name="box",
    frame_start=10,
    frame_end=30,
    hand_name='inspire_hand',
    subject_name="s01",
    use_clip="01",
    load_retarget_contact=False,
    retarget_name="genesis",
):
    """ This is only processed Arctic data, not the raw data. Not including dexterous hand retargeting data. """
    demo_fname = f"{ARCTIC_PROCESSED_DIR}/{subject_name}/{obj_name}_use_{use_clip}.npy"
    demo_data = np.load(demo_fname, allow_pickle=True).item() 
    world_coord = demo_data["world_coord"] 
    # this contains dict_keys(['joints.left', 'joints.right', 'contacts.left', 'valid_contacts.left', 'contacts.right', 'valid_contacts.right', 'contact_threshold', 'contact_links_left', 'contact_links_right'])
    demo_data = demo_data["params"] 
    
    demo_data = {
        "obj_pos": demo_data["obj_trans"][frame_start:frame_end], 
        "obj_quat": demo_data["obj_quat"][frame_start:frame_end],
        "obj_arti": demo_data["obj_arti"][frame_start:frame_end], 
        "contact_links_left": world_coord["contact_links_left"][frame_start:frame_end],
        "contact_links_right": world_coord["contact_links_right"][frame_start:frame_end],
    }
    if load_retarget_contact:
        retar_contact = load_contact_retarget_data(
            obj_name=obj_name, hand_name=hand_name, frame_start=frame_start, frame_end=frame_end,
            use_clip=use_clip, subject_name=subject_name, save_name=retarget_name,
        )
        print(f"Replacing demo_data with retarget contact data")
        demo_data.update(retar_contact) 
    return demo_data
 
def get_joint_init_limits(joint_pos_dict):
    limits = dict()
    default_qpos = dict()
    default_margin = 0.15
    for k, v in joint_pos_dict.items():
        margin = default_margin
        if 'tx' in k or 'ty' in k or 'tz' in k:
            margin = 0.2 # 20 cm
        if 'roll' in k or 'pitch' in k or 'yaw' in k:
            # print(f"Using 30 degrees margin for {k}")
            margin = 0.5 # 30 degrees
        limits[k] = (min(v) - margin, max(v) + margin)
        default_qpos[k] = v[0]
    return limits, default_qpos


# +-6.2 rad is the limit on the synthetic 6-DOF "forearm" roll/pitch/yaw joints in every
# allegro/inspire *_6dof.urdf (just under a full +-2*pi turn). Translations are +-5 m.
FOREARM_ROT_LIMIT = 6.2

def wrap_forearm_qpos_into_limits(qpos_dict, rot_limit=FOREARM_ROT_LIMIT):
    """Seat each forearm roll/pitch/yaw reference inside [-rot_limit, rot_limit].

    Retargeted wrist DOFs are continuous (unwrapped) Euler angles that can drift past
    +-2*pi, i.e. outside the joint's physical range; the sim then silently clamps them and
    the grasp breaks (object never articulates -> bad ADD/AUC). We shift each (already
    frame-sliced) trajectory by a whole number of 2*pi turns chosen to put it in range.
    A 2*pi shift on a single revolute joint is an exact identity rotation, so the wrist
    orientation, the trajectory continuity, and the FK-derived kpt_pos / wrist_pose / contact
    targets are all preserved -- only the PD-reference branch changes, becoming reachable.

    Because this runs on the post-slice trajectory it fits exactly the frames being trained.
    A residual overflow (continuous span > 2*rot_limit, e.g. a >2-turn wrist spin) is clamped
    and warned about. Non-forearm-rotation joints are returned untouched, and an already
    in-range joint is left exactly as-is (no-op for the in-range `para` references).
    """
    two_pi = 2.0 * np.pi
    lo, hi = -rot_limit, rot_limit
    out = dict(qpos_dict)
    for jname, q in qpos_dict.items():
        if not any(tag in jname for tag in ("forearm_roll", "forearm_pitch", "forearm_yaw")):
            continue
        is_t = torch.is_tensor(q)
        arr = q.detach().cpu().numpy() if is_t else np.asarray(q)
        cont = np.unwrap(arr.astype(np.float64))
        lo_c, hi_c = float(cont.min()), float(cont.max())
        if (hi_c - lo_c) <= (hi - lo):
            # smallest 2*pi shift that fits the whole trajectory in range (prefer no shift)
            t_lo = np.ceil((lo - lo_c) / two_pi)
            t_hi = np.floor((hi - hi_c) / two_pi)
            turns = float(np.clip(0.0, t_lo, t_hi)) if t_lo <= t_hi \
                else round((-(lo_c + hi_c) / 2.0) / two_pi)
        else:
            turns = round((-(lo_c + hi_c) / 2.0) / two_pi)  # span too big: centre, then clip
        shifted = cont + two_pi * turns
        n_oor = int(np.count_nonzero((shifted < lo) | (shifted > hi)))
        if turns == 0 and n_oor == 0:
            continue  # already in range -> leave the original object untouched
        fixed = np.clip(shifted, lo, hi) if n_oor else shifted
        if n_oor:
            print(f"WARNING: {jname}: {n_oor} frame(s) still outside +-{rot_limit} after a "
                  f"2*pi wrap (continuous span exceeds the joint range); clamped. Inspect "
                  f"this clip's wrist retargeting.")
        out[jname] = torch.as_tensor(fixed, dtype=q.dtype) if is_t else fixed.astype(arr.dtype)
    return out
 
def load_genesis_retarget_data(
    obj_name="box",
    hand_name='inspire_hand',
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
    given_data_fname=None,
):
    """ data saved from new retargeting code """
    ret_type = "vector"
    if 'shadow' in hand_name:
        print(f"Using position retargeting for {hand_name}")
        ret_type = "position"
    if given_data_fname is not None:
        data_fname = given_data_fname
    else:
        data_fname = f"{RETARGET_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}_{ret_type}_{save_name}.npy"
    loaded_tensor = False
    if not os.path.exists(data_fname):
        # try .pt extension
        data_fname = data_fname.replace(".npy", ".pt")
        assert os.path.exists(data_fname), f"File {data_fname} not found"

    if data_fname.endswith(".npy"):
        data = np.load(data_fname, allow_pickle=True).item()
    else:
        data = torch.load(data_fname, weights_only=False)
        loaded_tensor = True 

    demo_data = data["demo_data"] 
    demo_data = {k: v[frame_start:frame_end] for k, v in demo_data.items()}
    if len(demo_data['obj_arti'].shape) > 1:
        demo_data['obj_arti'] = demo_data['obj_arti'][:, 0] # shape (num_frames,)

    retarget_loaded = data["retarget_data"] 
    retarget_data = dict()
    for side in ['left', 'right']:
        loaded = retarget_loaded[side]
        residual_qpos = loaded["joint_qpos"]
        sliced_qpos = {k: v[frame_start:frame_end] for k, v in residual_qpos.items()}
        # Make the forearm wrist reference physically reachable BEFORE limits/init are
        # derived from it, so the residual base, the data-derived joint limits and init_qpos
        # are all in-range and mutually consistent (see wrap_forearm_qpos_into_limits).
        sliced_qpos = wrap_forearm_qpos_into_limits(sliced_qpos)

        qpos_targets = None
        if 'joint_targets' in loaded:
            print("Using joint_targets")
            qpos_targets = loaded["joint_targets"]
            qpos_targets = {k: v[frame_start:frame_end] for k, v in qpos_targets.items()}
            qpos_targets = wrap_forearm_qpos_into_limits(qpos_targets)
        limits, init_pos = get_joint_init_limits(sliced_qpos) # this is a dict
        kpt_pos = loaded["kpt_pos"]
        if len(kpt_pos.shape) > 3:
            print("Omitting the first dimension of kpt_pos")
            kpt_pos = kpt_pos[0]
        kpt_info = dict(
            kpt_pos=kpt_pos[frame_start:frame_end],
            kpt_names=loaded["kpt_names"],
        )
        wrist_pose = loaded[f"wrist_pose"]
        if len(wrist_pose.shape) > 2:
            print("Omitting the first dimension of wrist_pose")
            wrist_pose = wrist_pose[0]
        wrist_pose = wrist_pose[frame_start:frame_end]
        num_frames = wrist_pose.shape[0]
        retarget_data[side] = dict(
            init_qpos=init_pos, 
            limits=limits, 
            residual_qpos=sliced_qpos,
            qpos_targets=qpos_targets,
            num_frames=num_frames,
            kpts_data=kpt_info,
            wrist_pose=wrist_pose, # need this for contact frame reward
            ) 
    return demo_data, retarget_data

def load_contact_retarget_data(
    obj_name="box",
    hand_name='inspire_hand',
    frame_start=0,
    frame_end=100,
    save_name="genesis",
    use_clip="01",
    subject_name="s01",
):
    # Prefer a retarget-specific file (e.g. box_use_01_graph.npy, from map_contacts.py
    # --save_suffix _{save_name}); fall back to the unsuffixed original (para baseline).
    fname = f"{RETARGET_CONTACT_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}_{save_name}.npy"
    if not os.path.exists(fname):
        fname = f"{RETARGET_CONTACT_DIR}/{hand_name}/{subject_name}/{obj_name}_use_{use_clip}.npy"
    assert os.path.exists(fname), f"File {fname} not found"
    print(f"Loading retarget contact data from {fname}")
    loaded = np.load(fname, allow_pickle=True).item()
    retar_contact = dict()
    for side in ['left', 'right']:
        data = loaded[side] 
        for source_key, target_key in zip(
            ["dexlink_contacts", "dexlink_valid_contacts"],
            [f"contact_links_{side}", f"contact_links_valid_{side}"]
        ):
            retar_contact[target_key] = data[source_key][frame_start:frame_end]
        retar_contact[side] = {key: data[key] for key in ["collision_link_names", "collision_link_local_idxs"]}
        
    return retar_contact
