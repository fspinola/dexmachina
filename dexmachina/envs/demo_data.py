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


# Limit on the synthetic 6-DOF "forearm" roll/pitch/yaw joints in every allegro/inspire
# *_6dof.urdf. WIDENED 6.2 -> 12.0 rad (~1.9 turns): these are synthetic joints (no real
# hardware), so the range is free to widen. ~12 lets the smooth (unwrapped + whole-turn-shifted)
# wrist trajectory fit even for gimbal-winding clips (ketchup-40-340, waffleiron) WITHOUT
# clamping or a branch-flip re-encode. MUST match the URDF <limit> (also set to +-12.0).
FOREARM_ROT_LIMIT = 12.0

def _forearm_rotation(roll, pitch, yaw):
    """Wrist orientation from forearm joints, per the URDF chain (roll axis=Z, pitch=X,
    yaw=-Y, applied in that order): R = Rz(roll) @ Rx(pitch) @ Ry(-yaw)."""
    from scipy.spatial.transform import Rotation as _Rot
    rpy = np.stack([np.asarray(roll, float), np.asarray(pitch, float), -np.asarray(yaw, float)], axis=-1)
    return _Rot.from_euler('ZXY', rpy)


def _reencode_forearm_rotations(roll, pitch, yaw, rot_limit=FOREARM_ROT_LIMIT):
    """Re-derive (roll,pitch,yaw) so every frame lies within +-rot_limit, is continuous, and
    is minimal-travel, WITHOUT changing the wrist orientation. This rescues clips whose Euler
    wrist angles balloon past +-2*pi near GIMBAL LOCK (pitch ~ +-90deg, where roll/yaw become
    degenerate and inflate together): a whole-turn shift cannot compress that span, but a
    different in-range Euler branch represents the identical orientation. Each frame's
    orientation has two ZXY solutions (roll+-pi, pi-pitch, yaw+-pi); combined with whole-turn
    shifts that yields a small candidate set per frame (FK-checked to reproduce the exact
    orientation and to lie in range), and a DP picks the continuous least-travel path through
    them. Returns (roll',pitch',yaw') or None if some orientation is genuinely unreachable in
    range (then the caller falls back to clamping)."""
    roll = np.asarray(roll, float); pitch = np.asarray(pitch, float); yaw = np.asarray(yaw, float)
    T = len(roll)
    Rt = _forearm_rotation(roll, pitch, yaw)          # exact target orientations (correct)
    two_pi = 2.0 * np.pi
    tol = np.radians(0.05)

    def cands(t):
        a, b, c = Rt[t].as_euler('ZXY')
        reps = []
        for (r0, p0, y0) in [(a, b, -c), (a + np.pi, np.pi - b, -(c + np.pi))]:
            p0 = (p0 + np.pi) % two_pi - np.pi
            if abs(p0) > rot_limit:
                continue
            for kr in range(-2, 3):
                r = r0 + two_pi * kr
                if abs(r) > rot_limit:
                    continue
                for ky in range(-2, 3):
                    y = y0 + two_pi * ky
                    if abs(y) > rot_limit:
                        continue
                    cand = np.array([r, p0, y])
                    if (_forearm_rotation(cand[0:1], cand[1:2], cand[2:3])[0].inv() * Rt[t]).magnitude() < tol:
                        reps.append(cand)
        return reps

    cl = [cands(t) for t in range(T)]
    if any(len(c) == 0 for c in cl):
        return None                                    # some frame unreachable in range
    # Bottleneck DP: minimise the WORST per-frame jump first (smoothness), then total travel.
    # Min-total-travel alone takes a large branch-flip shortcut at the gimbal (a ~150 deg/frame
    # jump) even when a perfectly smooth in-range branch exists -- the PD reference must stay
    # continuous, so the max jump is the primary objective and total travel only breaks ties.
    # (Where the orientation genuinely winds the gimbal, e.g. waffleiron, the best achievable
    # max jump is still large and this returns that minimal-jump path.)
    dp = [(0.0, float(np.abs(c).sum())) for c in cl[0]]   # (max_jump, total_travel); centre frame 0
    bp = [[-1] * len(cl[0])]
    prev = cl[0]
    for t in range(1, T):
        cur = cl[t]; nd = []; nb = []
        for cj in cur:
            best = (1e18, 1e18); bi = -1
            for i, ci in enumerate(prev):
                step = np.abs(cj - ci)
                cand = (max(dp[i][0], float(step.max())), dp[i][1] + float(step.sum()))
                if cand < best:
                    best = cand; bi = i
            nd.append(best); nb.append(bi)
        dp, prev = nd, cur; bp.append(nb)
    j = int(min(range(len(dp)), key=lambda k: dp[k])); path = [None] * T
    for t in range(T - 1, -1, -1):
        path[t] = cl[t][j]; j = bp[t][j]
    out = np.asarray(path)
    err = max((_forearm_rotation(out[t:t + 1, 0], out[t:t + 1, 1], out[t:t + 1, 2])[0].inv() * Rt[t]).magnitude()
              for t in range(T))
    if err > np.radians(0.1) or float(np.abs(out).max()) > rot_limit + 1e-6:
        return None                                    # safety: reject if FK drifts or out of range
    return out[:, 0], out[:, 1], out[:, 2]


def wrap_forearm_qpos_into_limits(qpos_dict, rot_limit=FOREARM_ROT_LIMIT):
    """Seat each forearm roll/pitch/yaw reference inside [-rot_limit, rot_limit].

    Retargeted wrist DOFs are continuous (unwrapped) Euler angles that can drift past +-2*pi,
    i.e. outside the joint's physical range; the sim then silently clamps them and the grasp
    breaks. First we try the cheap fix: shift each joint by a whole number of 2*pi turns (an
    exact identity rotation, so orientation / continuity / FK-derived kpt_pos / wrist_pose /
    contacts are all preserved). That handles the common 'wrong-branch' case and is a no-op for
    already-in-range `para` refs.

    When a whole-turn shift CANNOT seat a side's wrist in range (GIMBAL LOCK: roll+yaw balloon
    >2 turns near pitch~+-90deg, e.g. ketchup-40-340 / waffleiron-40-340), we re-encode that
    side's three rotation joints jointly via _reencode_forearm_rotations -- the same wrist
    orientation expressed on an in-range Euler branch (FK-verified to 0deg). Only that side's
    angles change; orientation/kpt/contacts are untouched, so no contact regen is needed. If the
    re-encode is genuinely infeasible we fall back to clamping + a warning (old behaviour).
    """
    two_pi = 2.0 * np.pi
    lo, hi = -rot_limit, rot_limit
    out = dict(qpos_dict)
    rot_axes = ("roll", "pitch", "yaw")

    def whole_turn(q):
        is_t = torch.is_tensor(q)
        arr = (q.detach().cpu().numpy() if is_t else np.asarray(q)).astype(np.float64)
        cont = np.unwrap(arr)
        lo_c, hi_c = float(cont.min()), float(cont.max())
        if (hi_c - lo_c) <= (hi - lo):
            t_lo = np.ceil((lo - lo_c) / two_pi); t_hi = np.floor((hi - hi_c) / two_pi)
            turns = float(np.clip(0.0, t_lo, t_hi)) if t_lo <= t_hi else round((-(lo_c + hi_c) / 2.0) / two_pi)
        else:
            turns = round((-(lo_c + hi_c) / 2.0) / two_pi)
        shifted = cont + two_pi * turns
        n_oor = int(np.count_nonzero((shifted < lo) | (shifted > hi)))
        return is_t, arr, shifted, n_oor

    def store(jname, vals):
        q = qpos_dict[jname]
        out[jname] = torch.as_tensor(vals, dtype=q.dtype) if torch.is_tensor(q) else vals.astype(np.asarray(q).dtype)

    handled = set()
    # process complete L_/R_ forearm rotation triples together (so gimbal re-encode can couple them)
    for side in ("L_forearm", "R_forearm"):
        names = {}
        for jname in qpos_dict:
            if jname.startswith(side):
                for ax in rot_axes:
                    if f"forearm_{ax}" in jname:
                        names[ax] = jname
        if len(names) != 3:
            continue
        wt = {ax: whole_turn(qpos_dict[names[ax]]) for ax in rot_axes}
        if all(w[3] == 0 for w in wt.values()):
            # whole-turn shift fits -> apply it (no-op where unchanged), preserving old behaviour
            for ax in rot_axes:
                is_t, arr, shifted, _ = wt[ax]
                if not np.allclose(shifted, arr):
                    store(names[ax], shifted)
                handled.add(names[ax])
            continue
        # a whole-turn shift would clamp -> try the gimbal-safe joint re-encode
        res = _reencode_forearm_rotations(wt["roll"][1], wt["pitch"][1], wt["yaw"][1], rot_limit)
        if res is not None:
            print(f"INFO: {side} wrist re-encoded (gimbal-safe) into +-{rot_limit} without "
                  f"clamping; orientation preserved (0 deg).")
            for ax, vals in zip(rot_axes, res):
                store(names[ax], vals)
                handled.add(names[ax])
        else:
            for ax in rot_axes:               # infeasible -> clamp + warn (old behaviour)
                is_t, arr, shifted, n_oor = wt[ax]
                fixed = np.clip(shifted, lo, hi)
                if n_oor:
                    print(f"WARNING: {names[ax]}: {n_oor} frame(s) outside +-{rot_limit}; "
                          f"re-encode infeasible; clamped. Inspect this clip's wrist retargeting.")
                store(names[ax], fixed)
                handled.add(names[ax])

    # any forearm-rotation joint not part of a complete triple -> per-joint whole-turn wrap
    for jname, q in qpos_dict.items():
        if jname in handled or not any(f"forearm_{ax}" in jname for ax in rot_axes):
            continue
        is_t, arr, shifted, n_oor = whole_turn(q)
        if np.allclose(shifted, arr) and n_oor == 0:
            continue
        fixed = np.clip(shifted, lo, hi) if n_oor else shifted
        if n_oor:
            print(f"WARNING: {jname}: {n_oor} frame(s) still outside +-{rot_limit} after a 2*pi wrap; clamped.")
        store(jname, fixed)
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
    if "objects" in demo_data:
        # OakInk per-object demo: slice each object's arrays; carry the hand->object hints
        # (strings) and any per-side contact arrays verbatim.
        sliced = {"objects": {
            name: {f: od[f][frame_start:frame_end] for f in od}
            for name, od in demo_data["objects"].items()
        }}
        for hint in ("object_left", "object_right"):
            if hint in demo_data:
                sliced[hint] = demo_data[hint]
        for ck in ("contact_links_left", "contact_links_right"):
            if ck in demo_data:
                sliced[ck] = demo_data[ck][frame_start:frame_end]
        demo_data = sliced
    else:
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
