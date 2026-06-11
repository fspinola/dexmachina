"""Inference-only wrapper around the stage-1 retargeter for drop-in use.

Loads a trained checkpoint (model weights + layout + hand profile) and exposes
two entry points:

- :meth:`Retargeter.predict_sequence` — bulk inference over a full sequence.
- :meth:`Retargeter.predict_step` — single-frame inference with internal
  sliding-window buffer + autoregressive state, for online use inside an env
  rollout.

Output schema matches the ``predictions.npz`` written by
``learned_retargeter.kinematic.evaluate.evaluate_stage1_checkpoint`` minus the
target-side keys (no GT at inference). Wrist outputs are in the model's
**native** frame: ``base_link`` for ``predicted_wrist_frame="base"`` ckpts (e.g.
``qsup_wsup``), ``retarget_origin`` for ``"origin"`` ckpts. The eval pipeline
composes base→origin for metric computation; this wrapper skips that step so a
base-trained ckpt can drive a base-frame consumer directly.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._geometry import Stage1RobotGeometry
from ._model import BaselineInputLayout, Stage1MLPBaseline
from .preprocess import (
    HAND_FEAT_DIM,
    HAND_TOKENS,
    estimate_outward_normals,
    pack_frame_features,
    sample_object_points,
)


def _resolve_state_path(ckpt_path: str | Path) -> Path:
    """Resolve checkpoint payload from a directory or explicit file path."""

    src = Path(ckpt_path).expanduser().resolve()
    if src.is_file():
        return src
    for candidate in (src / "model_state.pth", src / "model_state_last.pth"):
        if candidate.exists():
            return candidate
    legacy = sorted(src.glob("model_state*.pth"))
    if legacy:
        return legacy[0]
    raise FileNotFoundError(
        f"No checkpoint payload under {src}. Pass a .pth path or a directory "
        "containing model_state.pth / model_state_last.pth."
    )


def _infer_model_config_from_state(
    *,
    state_dict: dict[str, Any],
    layout_meta: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    joint_dim: int,
) -> dict[str, Any]:
    """Reconstruct the model_config dict needed to rebuild the nn.Module.

    Mirrors the heuristics in ``evaluate._load_model`` so old checkpoints that
    pre-date some flags still load cleanly: we trust the saved state-dict shapes
    over the saved config when they disagree.
    """

    has_wrist_decoder = any(str(k).startswith("wrist_decoder.") for k in state_dict)
    has_decoder_input_norm = any(str(k).startswith("decoder_input_norm.") for k in state_dict)
    fusion_dim = int(layout_meta["hand_tokens"]) * int(layout_meta.get("hand_feat_dim", 12)) + 256

    decoder_in_dim = int(state_dict["decoder.net.0.weight"].shape[1])
    autoregressive_enabled = bool(decoder_in_dim > fusion_dim)
    autoregressive_detach_prev_q = bool(
        model_cfg.get(
            "autoregressive_detach_prev_q",
            train_cfg.get("autoregressive_detach_prev_q", True),
        )
    )

    wrist_hint_dim = int(model_cfg.get("wrist_hint_dim", 9))
    autoregressive_wrist_enabled = bool(
        model_cfg.get(
            "autoregressive_wrist_enabled",
            train_cfg.get("autoregressive_wrist_enabled", False),
        )
    )
    autoregressive_detach_prev_wrist = bool(
        model_cfg.get(
            "autoregressive_detach_prev_wrist",
            train_cfg.get("autoregressive_detach_prev_wrist", True),
        )
    )
    if "wrist_decoder.net.0.weight" in state_dict:
        in_features = int(state_dict["wrist_decoder.net.0.weight"].shape[1])
        excess = int(max(0, in_features - fusion_dim))
        if autoregressive_wrist_enabled:
            wrist_hint_dim = int(max(0, excess - 9))
        else:
            wrist_hint_dim = excess

    return {
        "predict_wrist_pose": bool(model_cfg.get("predict_wrist_pose", has_wrist_decoder)),
        "wrist_hint_dim": int(wrist_hint_dim),
        "autoregressive_enabled": bool(autoregressive_enabled),
        "autoregressive_detach_prev_q": bool(autoregressive_detach_prev_q),
        "autoregressive_wrist_enabled": bool(autoregressive_wrist_enabled),
        "autoregressive_detach_prev_wrist": bool(autoregressive_detach_prev_wrist),
        "decoder_input_layernorm": bool(
            model_cfg.get(
                "decoder_input_layernorm",
                train_cfg.get("decoder_input_layernorm", has_decoder_input_norm),
            )
            or has_decoder_input_norm
        ),
        "decoder_hidden_dim": int(train_cfg.get("decoder_hidden_dim", 512)),
        "decoder_layers": int(train_cfg.get("decoder_layers", 4)),
        "joint_dim": int(joint_dim),
    }


class Retargeter:
    """Drop-in inference wrapper around ``Stage1MLPBaseline``."""

    def __init__(
        self,
        *,
        model: Stage1MLPBaseline,
        geometry: Stage1RobotGeometry,
        window_size: int,
        object_points: int,
        joint_dim: int,
        fps: float,
        device: torch.device,
        seed: int,
    ):
        self._model = model.to(device).eval()
        self._geometry = geometry
        self._window_size = int(window_size)
        self._object_points = int(object_points)
        self._joint_dim = int(joint_dim)
        self._fps = float(fps)
        self._device = device
        self._rng_seed = int(seed)

        self._ar_q = bool(getattr(model, "autoregressive_enabled", False))
        self._ar_wrist = bool(getattr(model, "autoregressive_wrist_enabled", False))

        self._rng: np.random.Generator
        self._buffer: deque[dict[str, np.ndarray]]
        self._prev_kpts: np.ndarray | None
        self._prev_q: np.ndarray | None
        self._prev_wrist: np.ndarray | None
        self.reset()

    # -- public API -----------------------------------------------------------

    @staticmethod
    def load(
        ckpt_path: str | Path,
        *,
        hand_config_path: str | Path | None = None,
        device: str = "cuda",
        fps: float = 30.0,
        seed: int = 0,
    ) -> "Retargeter":
        """Load a checkpoint and return a ready-to-use ``Retargeter``.

        Args:
            ckpt_path: Either a directory containing ``model_state.pth`` (or
                ``model_state_last.pth``) or an explicit ``.pth`` file path.
            hand_config_path: Optional override for the hand-profile YAML path
                baked into the checkpoint. Pass when the original path no
                longer exists on this machine (e.g. a different repo checkout).
            device: ``"cuda"``, ``"cpu"``, or any explicit torch device string.
            fps: Sequence frame rate, used to scale finite-difference
                velocities at preprocessing time. The training shards use
                ``30.0``; pass the matching value if your demos differ.
            seed: RNG seed used when subsampling object points to layout K.
        """

        payload = torch.load(_resolve_state_path(ckpt_path), map_location="cpu")
        layout_meta = dict(payload.get("layout", {}) or {})
        if not layout_meta:
            raise ValueError("Checkpoint is missing the 'layout' block.")

        resolved_hand_config = str(hand_config_path) if hand_config_path else str(
            payload.get("hand_config_path", "") or ""
        )
        if not resolved_hand_config:
            raise ValueError(
                "Cannot resolve hand_config_path. Either pass hand_config_path explicitly "
                "or use a checkpoint that recorded it."
            )

        geometry = Stage1RobotGeometry.from_hand_config(
            hand_name=None,
            hand_config_path=resolved_hand_config,
            strict_urdf=True,
        )
        joint_dim = int(payload.get("joint_dim", geometry.joint_count))

        state_dict = dict(payload.get("model_state_dict", {}) or {})
        cfg = _infer_model_config_from_state(
            state_dict=state_dict,
            layout_meta=layout_meta,
            model_cfg=dict(payload.get("model_config", {}) or {}),
            train_cfg=dict(payload.get("train_config", {}) or {}),
            joint_dim=joint_dim,
        )

        model = Stage1MLPBaseline(
            layout=BaselineInputLayout(
                window_size=int(layout_meta["window_size"]),
                hand_tokens=int(layout_meta["hand_tokens"]),
                object_points=int(layout_meta["object_points"]),
                fingertip_targets=int(layout_meta["fingertip_targets"]),
                hand_feat_dim=int(layout_meta.get("hand_feat_dim", 12)),
                object_feat_dim=int(layout_meta.get("object_feat_dim", 6)),
            ),
            joint_dim=joint_dim,
            robot_geometry=geometry,
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            decoder_layers=cfg["decoder_layers"],
            predict_wrist_pose=cfg["predict_wrist_pose"],
            wrist_hint_dim=cfg["wrist_hint_dim"],
            autoregressive_enabled=cfg["autoregressive_enabled"],
            autoregressive_detach_prev_q=cfg["autoregressive_detach_prev_q"],
            autoregressive_wrist_enabled=cfg["autoregressive_wrist_enabled"],
            autoregressive_detach_prev_wrist=cfg["autoregressive_detach_prev_wrist"],
            decoder_input_layernorm=cfg["decoder_input_layernorm"],
            compute_full_fk=False,  # wrapper doesn't expose full_fk_points / primitive_points
        )
        model.load_state_dict(state_dict)

        dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        return Retargeter(
            model=model,
            geometry=geometry,
            window_size=int(layout_meta["window_size"]),
            object_points=int(layout_meta["object_points"]),
            joint_dim=joint_dim,
            fps=fps,
            device=dev,
            seed=seed,
        )

    @property
    def joint_dim(self) -> int:
        return self._joint_dim

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def object_points(self) -> int:
        return self._object_points

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._geometry.joint_lower, dtype=np.float32),
            np.asarray(self._geometry.joint_upper, dtype=np.float32),
        )

    def reset(self) -> None:
        """Clear sliding-window buffer + autoregressive state."""

        self._rng = np.random.default_rng(self._rng_seed)
        self._buffer = deque(maxlen=self._window_size)
        self._prev_kpts = None
        self._prev_q = None
        self._prev_wrist = None

    def predict_sequence(
        self,
        mano_kpts: np.ndarray,
        obj_points: np.ndarray,
        wrist_world: np.ndarray,
        *,
        obj_normals: np.ndarray | None = None,
        hand_rot6d: np.ndarray | None = None,
        batch_size: int = 512,
    ) -> dict[str, np.ndarray]:
        """Run inference over a full sequence using training-style sliding windows.

        Per frame ``t``, builds a window covering frames ``[t-W+1, ..., t]``
        with edge-padding for ``t < W-1`` (so the output covers all ``T``
        frames). Each window is an independent forward — autoregressive state
        rolls *within* the window (model-internal) and resets at the window
        boundary, exactly mirroring training/eval. For streaming ``W=1``
        checkpoints this falls back to the sequential AR-carry loop in
        ``evaluate._predict`` (carries last-frame ``q`` / ``wrist`` between
        steps).

        Args:
            mano_kpts: ``[T, 21, 3]`` MANO keypoints in world frame.
            obj_points: ``[T, N, 3]`` per-frame object points in world frame.
                ``N`` may differ from the model's layout ``K`` — points are
                resampled to ``K`` internally.
            wrist_world: ``[T, 4, 4]`` per-frame supervision-hand wrist
                transform (world frame). This is the MANO wrist transform from
                the demo; it defines the supervision-local frame the model
                operates in.
            obj_normals: optional ``[T, N, 3]`` matching outward normals. If
                ``None``, normals are estimated per-frame via local PCA.
            hand_rot6d: optional ``[T, 21, 6]`` per-keypoint rot6d. Zeros when
                absent (matches the training-time default).
            batch_size: number of windows per forward pass. Default 512 keeps
                peak GPU memory bounded for large ``K`` and long sequences.

        Returns:
            Dict with the same keys as ``predictions.npz``:

            - ``pred_q``        ``[T, J]`` joint angles (rad, clamped to URDF limits).
            - ``pred_q_raw``    ``[T, J]`` raw decoder outputs (pre-sigmoid).
            - ``pred_tips``     ``[T, F, 3]`` fingertip positions in retarget-
              origin frame (model FK output, no wrist-pose application).
            - ``pred_wrist_pos_world``    ``[T, 3]`` predicted wrist translation
              in the model's native frame (base_link for base-trained ckpts).
            - ``pred_wrist_rot6d_world``  ``[T, 6]`` predicted wrist 6D rotation
              in the model's native frame.
            - ``obj_points``    ``[T, K, 3]`` resampled object points in
              supervision-wrist-local frame (the actual model input).
        """

        kpts = np.asarray(mano_kpts, dtype=np.float32)
        objs = np.asarray(obj_points, dtype=np.float32)
        wrists = np.asarray(wrist_world, dtype=np.float32)
        if kpts.ndim != 3 or kpts.shape[1:] != (21, 3):
            raise ValueError(f"Expected mano_kpts [T, 21, 3], got {kpts.shape}.")
        T = int(kpts.shape[0])
        if objs.ndim != 3 or objs.shape[0] != T or objs.shape[-1] != 3:
            raise ValueError(f"Expected obj_points [T, N, 3] with T={T}, got {objs.shape}.")
        if wrists.shape != (T, 4, 4):
            raise ValueError(f"Expected wrist_world [{T}, 4, 4], got {wrists.shape}.")
        if obj_normals is not None:
            nrm_seq = np.asarray(obj_normals, dtype=np.float32)
            if nrm_seq.shape != objs.shape:
                raise ValueError(
                    f"obj_normals shape {nrm_seq.shape} must match obj_points {objs.shape}."
                )
        else:
            nrm_seq = None
        if hand_rot6d is not None:
            rot6d_seq = np.asarray(hand_rot6d, dtype=np.float32)
            if rot6d_seq.shape != (T, 21, 6):
                raise ValueError(f"Expected hand_rot6d [{T}, 21, 6], got {rot6d_seq.shape}.")
        else:
            rot6d_seq = None

        # Per-frame features (single-frame, single-hand).
        hand_feats = np.zeros((T, HAND_TOKENS, HAND_FEAT_DIM), dtype=np.float32)
        hand_mask = np.zeros((T, HAND_TOKENS), dtype=bool)
        obj_feats = np.zeros((T, self._object_points, 6), dtype=np.float32)
        obj_mask = np.zeros((T, self._object_points), dtype=bool)

        for ti in range(T):
            pts_world = objs[ti]
            nrm_world = (
                nrm_seq[ti] if nrm_seq is not None else estimate_outward_normals(pts_world, k=16)
            )
            pts_k, nrm_k = sample_object_points(
                pts_world, nrm_world, self._object_points, rng=self._rng
            )
            frame = pack_frame_features(
                mano_kpts_world=kpts[ti],
                wrist_world=wrists[ti],
                obj_points_world=pts_k,
                obj_normals_world=nrm_k,
                prev_kpts_world=(kpts[ti - 1] if ti > 0 else None),
                fps=self._fps,
                hand_rot6d=(rot6d_seq[ti] if rot6d_seq is not None else None),
            )
            hand_feats[ti] = frame["hand_feats"]
            hand_mask[ti] = frame["hand_mask"]
            obj_feats[ti] = frame["obj_feats"]
            obj_mask[ti] = frame["obj_mask"]

        W = self._window_size

        # Streaming AR T=1 path: sequential, carrying prev q/wrist between
        # frames. Matches use_streaming_ar_t1 branch in evaluate._predict.
        if W == 1 and (self._ar_q or self._ar_wrist):
            J = self._joint_dim
            pred_q = np.zeros((T, J), dtype=np.float32)
            pred_q_raw = np.zeros((T, J), dtype=np.float32)
            pred_tips = None
            pred_wrist_pos = np.zeros((T, 3), dtype=np.float32)
            pred_wrist_rot6d = np.zeros((T, 6), dtype=np.float32)
            prev_q = np.zeros((1, 1, J), dtype=np.float32)
            prev_wrist = np.zeros((1, 1, 9), dtype=np.float32)
            for ti in range(T):
                out = self._forward(
                    hand_feats=hand_feats[ti : ti + 1, None],
                    hand_mask=hand_mask[ti : ti + 1, None],
                    obj_feats=obj_feats[ti : ti + 1, None],
                    obj_mask=obj_mask[ti : ti + 1, None],
                    q_prev_hint=prev_q if self._ar_q else None,
                    wrist_prev_hint=prev_wrist if self._ar_wrist else None,
                )
                pred_q[ti] = out["q"][0, 0]
                pred_q_raw[ti] = out["q_raw"][0, 0]
                if pred_tips is None:
                    pred_tips = np.zeros((T,) + out["tips"].shape[2:], dtype=np.float32)
                pred_tips[ti] = out["tips"][0, 0]
                pred_wrist_pos[ti] = out["wrist_pos_world"][0, 0]
                pred_wrist_rot6d[ti] = out["wrist_rot6d_world"][0, 0]
                prev_q = out["q"][:, -1:, :].astype(np.float32)
                prev_wrist = np.concatenate(
                    [out["wrist_pos_world"][:, -1:, :], out["wrist_rot6d_world"][:, -1:, :]],
                    axis=-1,
                ).astype(np.float32)
            return {
                "pred_q": pred_q,
                "pred_q_raw": pred_q_raw,
                "pred_tips": pred_tips if pred_tips is not None else np.zeros((T, 0, 3), dtype=np.float32),
                "pred_wrist_pos_world": pred_wrist_pos,
                "pred_wrist_rot6d_world": pred_wrist_rot6d,
                "obj_points": obj_feats[..., :3].astype(np.float32),
            }

        # Window path (W > 1, or W=1 non-AR): build T sliding windows ending at
        # each frame t with past-replication edge padding (frames before t=0
        # repeat frame 0). Identical to what predict_step does online.
        w_offset = np.arange(W) - (W - 1)  # [-(W-1), ..., -1, 0]
        win_idx = np.maximum(0, np.arange(T)[:, None] + w_offset[None, :])  # [T, W]

        out_q = np.zeros((T, self._joint_dim), dtype=np.float32)
        out_q_raw = np.zeros((T, self._joint_dim), dtype=np.float32)
        out_tips: np.ndarray | None = None
        out_wpos = np.zeros((T, 3), dtype=np.float32)
        out_wrot = np.zeros((T, 6), dtype=np.float32)

        for s in range(0, T, int(batch_size)):
            e = min(s + int(batch_size), T)
            idx = win_idx[s:e]  # [B, W]
            out = self._forward(
                hand_feats=hand_feats[idx],
                hand_mask=hand_mask[idx],
                obj_feats=obj_feats[idx],
                obj_mask=obj_mask[idx],
                q_prev_hint=None,
                wrist_prev_hint=None,
            )
            out_q[s:e] = out["q"][:, -1, :]
            out_q_raw[s:e] = out["q_raw"][:, -1, :]
            if out_tips is None:
                out_tips = np.zeros((T,) + out["tips"].shape[2:], dtype=np.float32)
            out_tips[s:e] = out["tips"][:, -1]
            out_wpos[s:e] = out["wrist_pos_world"][:, -1, :]
            out_wrot[s:e] = out["wrist_rot6d_world"][:, -1, :]

        return {
            "pred_q": out_q,
            "pred_q_raw": out_q_raw,
            "pred_tips": out_tips if out_tips is not None else np.zeros((T, 0, 3), dtype=np.float32),
            "pred_wrist_pos_world": out_wpos,
            "pred_wrist_rot6d_world": out_wrot,
            "obj_points": obj_feats[..., :3].astype(np.float32),
        }

    def predict_step(
        self,
        mano_kpts: np.ndarray,
        obj_points: np.ndarray,
        wrist_world: np.ndarray,
        *,
        obj_normals: np.ndarray | None = None,
        hand_rot6d: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run single-frame inference; carries AR + window state across calls.

        Inputs are the per-frame counterparts of :meth:`predict_sequence`:
        ``mano_kpts`` ``[21, 3]``, ``obj_points`` ``[N, 3]``, ``wrist_world``
        ``[4, 4]``, optional ``obj_normals`` ``[N, 3]``, optional
        ``hand_rot6d`` ``[21, 6]``.

        For checkpoints with ``window_size > 1`` the first ``window_size - 1``
        calls pad the buffer by repeating the first frame, so the very first
        output is valid (matches the eval pipeline's edge-mode windowing).

        Returns:
            Dict with the same keys as :meth:`predict_sequence` but
            scalar-time: ``pred_q [J]``, ``pred_tips [F, 3]`` etc.
        """

        kpts = np.asarray(mano_kpts, dtype=np.float32)
        if kpts.shape != (21, 3):
            raise ValueError(f"Expected mano_kpts [21, 3], got {kpts.shape}.")
        pts_world = np.asarray(obj_points, dtype=np.float32)
        if pts_world.ndim != 2 or pts_world.shape[1] != 3:
            raise ValueError(f"Expected obj_points [N, 3], got {pts_world.shape}.")
        wrist = np.asarray(wrist_world, dtype=np.float32)
        if wrist.shape != (4, 4):
            raise ValueError(f"Expected wrist_world [4, 4], got {wrist.shape}.")
        nrm_world = (
            np.asarray(obj_normals, dtype=np.float32)
            if obj_normals is not None
            else estimate_outward_normals(pts_world, k=16)
        )
        if nrm_world.shape != pts_world.shape:
            raise ValueError(
                f"obj_normals shape {nrm_world.shape} must match obj_points {pts_world.shape}."
            )

        pts_k, nrm_k = sample_object_points(
            pts_world, nrm_world, self._object_points, rng=self._rng
        )
        frame = pack_frame_features(
            mano_kpts_world=kpts,
            wrist_world=wrist,
            obj_points_world=pts_k,
            obj_normals_world=nrm_k,
            prev_kpts_world=self._prev_kpts,
            fps=self._fps,
            hand_rot6d=hand_rot6d,
        )
        self._prev_kpts = kpts.copy()

        # Pad buffer with this frame on first calls so the very first
        # predict_step yields a valid output (matches edge-mode windowing).
        if len(self._buffer) == 0:
            for _ in range(self._window_size):
                self._buffer.append(frame)
        else:
            self._buffer.append(frame)

        hand_feats = np.stack([f["hand_feats"] for f in self._buffer], axis=0)[None]
        hand_mask = np.stack([f["hand_mask"] for f in self._buffer], axis=0)[None]
        obj_feats = np.stack([f["obj_feats"] for f in self._buffer], axis=0)[None]
        obj_mask = np.stack([f["obj_mask"] for f in self._buffer], axis=0)[None]

        # AR semantics match evaluate._predict:
        #   - W > 1: model rolls AR internally within the window; pass None so
        #     it starts from zeros at the window's first frame (training parity).
        #   - W = 1: stream-mode — pass last call's q/wrist as the AR seed so
        #     state carries frame-to-frame.
        q_prev_hint = None
        wrist_prev_hint = None
        if self._window_size == 1:
            if self._ar_q:
                seed = (
                    np.zeros((self._joint_dim,), dtype=np.float32)
                    if self._prev_q is None
                    else self._prev_q
                )
                q_prev_hint = seed.reshape(1, 1, self._joint_dim).copy()
            if self._ar_wrist:
                seed = (
                    np.zeros((9,), dtype=np.float32)
                    if self._prev_wrist is None
                    else self._prev_wrist
                )
                wrist_prev_hint = seed.reshape(1, 1, 9).copy()

        out = self._forward(
            hand_feats=hand_feats,
            hand_mask=hand_mask,
            obj_feats=obj_feats,
            obj_mask=obj_mask,
            q_prev_hint=q_prev_hint,
            wrist_prev_hint=wrist_prev_hint,
        )

        q_last = np.asarray(out["q"][0, -1], dtype=np.float32)
        q_raw_last = np.asarray(out["q_raw"][0, -1], dtype=np.float32)
        tips_last = np.asarray(out["tips"][0, -1], dtype=np.float32)
        wrist_pos_last = np.asarray(out["wrist_pos_world"][0, -1], dtype=np.float32)
        wrist_rot_last = np.asarray(out["wrist_rot6d_world"][0, -1], dtype=np.float32)

        self._prev_q = q_last.copy()
        self._prev_wrist = np.concatenate([wrist_pos_last, wrist_rot_last], axis=-1).copy()

        return {
            "pred_q": q_last,
            "pred_q_raw": q_raw_last,
            "pred_tips": tips_last,
            "pred_wrist_pos_world": wrist_pos_last,
            "pred_wrist_rot6d_world": wrist_rot_last,
            "obj_points": obj_feats[0, -1, :, :3].astype(np.float32),
        }

    # -- internal -------------------------------------------------------------

    def _forward(
        self,
        *,
        hand_feats: np.ndarray,
        hand_mask: np.ndarray,
        obj_feats: np.ndarray,
        obj_mask: np.ndarray,
        q_prev_hint: np.ndarray | None,
        wrist_prev_hint: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        """Run the underlying nn.Module on numpy inputs; return numpy outputs."""

        with torch.no_grad():
            out = self._model(
                hand_feats=torch.as_tensor(hand_feats, dtype=torch.float32, device=self._device),
                obj_feats=torch.as_tensor(obj_feats, dtype=torch.float32, device=self._device),
                hand_mask=torch.as_tensor(hand_mask, dtype=torch.bool, device=self._device),
                obj_mask=torch.as_tensor(obj_mask, dtype=torch.bool, device=self._device),
                wrist_hint_world=None,
                q_prev_hint=(
                    None
                    if q_prev_hint is None
                    else torch.as_tensor(q_prev_hint, dtype=torch.float32, device=self._device)
                ),
                wrist_prev_hint=(
                    None
                    if wrist_prev_hint is None
                    else torch.as_tensor(
                        wrist_prev_hint, dtype=torch.float32, device=self._device
                    )
                ),
            )
        return {k: v.detach().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
