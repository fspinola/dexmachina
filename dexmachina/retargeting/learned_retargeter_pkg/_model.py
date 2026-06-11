"""Stage-1 baseline model: hand flatten + PointNet object encoder + frame-wise MLP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._geometry import Stage1RobotGeometry


@dataclass(frozen=True)
class BaselineInputLayout:
    """Input tensor layout metadata for stage-1 baseline."""

    window_size: int
    hand_tokens: int
    object_points: int
    fingertip_targets: int
    hand_feat_dim: int = 12
    object_feat_dim: int = 6


class PointNetObjectEncoder(nn.Module):
    """Simple PointNet-style global object encoder per frame."""

    def __init__(self, object_feat_dim: int = 6):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(int(object_feat_dim), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, obj_feats: torch.Tensor, obj_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode object features.

        Args:
            obj_feats: ``[B*T,K,F_obj]``
            obj_mask: optional ``[B*T,K]`` boolean mask.
        Returns:
            ``[B*T,256]`` global object descriptors.
        """

        x = self.point_mlp(obj_feats)
        if obj_mask is not None:
            m = obj_mask.to(dtype=torch.bool)
            # Keep pooling finite even when a frame has no valid object points.
            neg_large = torch.full_like(x, -1.0e6)
            x = torch.where(m[..., None], x, neg_large)
            pooled = torch.max(x, dim=1).values
            invalid = ~torch.any(m, dim=1)
            if torch.any(invalid):
                pooled = torch.where(invalid[:, None], torch.zeros_like(pooled), pooled)
            return pooled
        return torch.max(x, dim=1).values


class FrameWiseDecoder(nn.Module):
    """Shared frame-wise MLP decoder for joint prediction."""

    def __init__(self, input_dim: int, joint_dim: int, hidden_dim: int = 512, layers: int = 4):
        super().__init__()
        if int(layers) < 2:
            raise ValueError("layers must be >= 2")

        modules: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(layers) - 1):
            modules.append(nn.Linear(in_dim, int(hidden_dim)))
            modules.append(nn.LayerNorm(int(hidden_dim)))
            modules.append(nn.ReLU(inplace=True))
            in_dim = int(hidden_dim)
        modules.append(nn.Linear(in_dim, int(joint_dim)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode frame features to raw joints."""

        return self.net(x)


class Stage1MLPBaseline(nn.Module):
    """Stage-1 baseline model with frame-independent shared weights.

    Design note (probabilistic head — future work). This model is
    deterministic: both ``decoder`` and ``wrist_decoder`` emit point
    estimates. The pseudo-GT used to train it (SLSQP-style geometric
    retargeting outputs) is not uniformly reliable across the dataset,
    and the residual-PPO stage downstream currently consumes the
    frozen outputs as deterministic features with no per-frame
    uncertainty signal. See ``paper/sections/07_limitations.tex``
    ("Deterministic kinematic teacher") for the full discussion.

    Minimal upgrade path when probabilistic outputs are wanted:

    1. Change ``decoder`` to emit ``2 * joint_dim`` (mu, log_sigma) and
       optionally do the same for the wrist position dims of
       ``wrist_decoder`` (first 3 of 9). Keep the rot6d dims (last 6)
       deterministic --- a Gaussian on rot6d is geometrically loose and
       not needed for the residual-RL setup, which consumes rot6d as
       a feature rather than sampling it.
    2. Recipe A (state-independent sigma): keep ``decoder``'s output
       dim unchanged; add ``log_sigma = nn.Parameter(torch.zeros(d))``
       as a separate module attribute; train with plain Gaussian NLL.
       No loss-attenuation pathology because sigma cannot adapt per
       sample. Matches the ManipTrans imitator's ``fixed_sigma=True``
       structure.
    3. Recipe B (state-dependent sigma): expand the decoder output as
       above; train with beta-NLL (beta=0.5) of Seitzer et al. ICLR
       2022 to avoid the network inflating sigma(x) on hard samples
       and giving up on the mean. Clamp ``log_sigma`` to ``[-5, 2]``.
    """

    def __init__(
        self,
        *,
        layout: BaselineInputLayout,
        joint_dim: int,
        robot_geometry: Stage1RobotGeometry,
        decoder_hidden_dim: int = 512,
        decoder_layers: int = 4,
        predict_wrist_pose: bool = True,
        wrist_hint_dim: int = 9,
        compute_full_fk: bool = True,
        autoregressive_enabled: bool = False,
        autoregressive_detach_prev_q: bool = True,
        autoregressive_wrist_enabled: bool = False,
        autoregressive_detach_prev_wrist: bool = True,
        decoder_input_layernorm: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.joint_dim = int(joint_dim)
        self.robot_geometry = robot_geometry
        self.predict_wrist_pose = bool(predict_wrist_pose)
        self.wrist_hint_dim = int(max(0, wrist_hint_dim))
        self.compute_full_fk = bool(compute_full_fk)
        self.autoregressive_enabled = bool(autoregressive_enabled)
        self.autoregressive_detach_prev_q = bool(autoregressive_detach_prev_q)
        self.autoregressive_wrist_enabled = bool(autoregressive_wrist_enabled)
        self.autoregressive_detach_prev_wrist = bool(autoregressive_detach_prev_wrist)
        self.decoder_input_layernorm = bool(decoder_input_layernorm)

        self.obj_encoder = PointNetObjectEncoder(object_feat_dim=int(layout.object_feat_dim))
        fusion_dim = int(layout.hand_tokens) * int(layout.hand_feat_dim) + 256
        self.fusion_dim = int(fusion_dim)
        decoder_input_dim = int(
            fusion_dim + (self.joint_dim if self.autoregressive_enabled else 0)
        )
        self.decoder_input_norm = (
            nn.LayerNorm(decoder_input_dim) if self.decoder_input_layernorm else None
        )
        self.decoder = FrameWiseDecoder(
            input_dim=decoder_input_dim,
            joint_dim=self.joint_dim,
            hidden_dim=int(decoder_hidden_dim),
            layers=int(decoder_layers),
        )
        self.wrist_decoder = None
        if self.predict_wrist_pose:
            wrist_in_dim = fusion_dim + self.wrist_hint_dim
            if self.autoregressive_wrist_enabled:
                wrist_in_dim += 9  # prev wrist: pos (3) + rot6d (6)
            self.wrist_decoder = FrameWiseDecoder(
                input_dim=wrist_in_dim,
                joint_dim=9,
                hidden_dim=int(decoder_hidden_dim),
                layers=int(decoder_layers),
            )

    @staticmethod
    def _normalize_rot6d(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize 6D rotation representation via Gram-Schmidt."""

        if rot6d.shape[-1] != 6:
            raise ValueError(f"Expected rot6d [...,6], got {tuple(rot6d.shape)}")
        clean = torch.nan_to_num(rot6d, nan=0.0, posinf=0.0, neginf=0.0)
        a1 = clean[..., 0:3]
        a2 = clean[..., 3:6]
        b1 = F.normalize(a1, dim=-1, eps=float(eps))
        proj = torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = F.normalize(a2 - proj, dim=-1, eps=float(eps))
        return torch.cat([b1, b2], dim=-1)

    def _fuse_features(
        self,
        *,
        hand_feats: torch.Tensor,
        obj_feats: torch.Tensor,
        hand_mask: torch.Tensor | None,
        obj_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build fused per-frame features [B,T,D_fused]."""

        b, t, h, fh = hand_feats.shape
        _, _, k, fo = obj_feats.shape

        hand_flat = hand_feats.reshape(b, t, h * fh)
        if hand_mask is not None:
            m = hand_mask.to(dtype=hand_flat.dtype).reshape(b, t, h, 1)
            hand_flat = (hand_feats * m).reshape(b, t, h * fh)

        obj_bt = obj_feats.reshape(b * t, k, fo)
        obj_m_bt = None if obj_mask is None else obj_mask.reshape(b * t, k)
        obj_global_bt = self.obj_encoder(obj_bt, obj_mask=obj_m_bt)
        obj_global = obj_global_bt.reshape(b, t, -1)
        return torch.cat([hand_flat, obj_global], dim=-1)

    def _decode_q(
        self,
        *,
        fused: torch.Tensor,
        q_prev_hint: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode joint trajectory using frame-wise or autoregressive mode."""

        b, t, _ = fused.shape
        if not self.autoregressive_enabled:
            dec_in = fused.reshape(b * t, -1)
            if self.decoder_input_norm is not None:
                dec_in = self.decoder_input_norm(dec_in)
            q_raw = self.decoder(dec_in).reshape(b, t, self.joint_dim)
            q_raw = torch.nan_to_num(q_raw, nan=0.0, posinf=0.0, neginf=0.0)
            q = self.robot_geometry.scale_raw_to_limits_torch(torch, q_raw)
            return q_raw, q

        if q_prev_hint is not None:
            if q_prev_hint.shape != (b, t, self.joint_dim):
                raise ValueError(
                    f"q_prev_hint must have shape [B,T,{self.joint_dim}], got {tuple(q_prev_hint.shape)}"
                )
            q_prev_hint = q_prev_hint.to(dtype=fused.dtype, device=fused.device)

        q_raw_steps: list[torch.Tensor] = []
        q_steps: list[torch.Tensor] = []
        prev_q = fused.new_zeros((b, self.joint_dim))
        for ti in range(t):
            prev_in = prev_q if q_prev_hint is None else q_prev_hint[:, ti, :]
            dec_in = torch.cat([fused[:, ti, :], prev_in], dim=-1)
            if self.decoder_input_norm is not None:
                dec_in = self.decoder_input_norm(dec_in)
            q_raw_t = self.decoder(dec_in)
            q_raw_t = torch.nan_to_num(q_raw_t, nan=0.0, posinf=0.0, neginf=0.0)
            q_t = self.robot_geometry.scale_raw_to_limits_torch(torch, q_raw_t[:, None, :])[:, 0, :]
            q_raw_steps.append(q_raw_t)
            q_steps.append(q_t)
            prev_q = q_t.detach() if self.autoregressive_detach_prev_q else q_t
        q_raw = torch.stack(q_raw_steps, dim=1)
        q = torch.stack(q_steps, dim=1)
        return q_raw, q

    def _decode_wrist(
        self,
        *,
        fused: torch.Tensor,
        wrist_hint_world: torch.Tensor | None = None,
        wrist_prev_hint: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode wrist trajectory frame-wise or autoregressively.

        Returns ``(wrist_raw [B,T,9], wrist_pos [B,T,3], wrist_rot6d [B,T,6])``.
        """

        b, t, _ = fused.shape
        if self.wrist_decoder is None:
            return (
                fused.new_zeros((b, t, 9)),
                fused.new_zeros((b, t, 3)),
                fused.new_zeros((b, t, 6)),
            )

        # External wrist hint (untouched legacy channel — typically GT prev frame at training).
        if int(self.wrist_hint_dim) > 0:
            if wrist_hint_world is None:
                wrist_hint_world = fused.new_zeros((b, t, int(self.wrist_hint_dim)))
            elif wrist_hint_world.shape != (b, t, int(self.wrist_hint_dim)):
                raise ValueError(
                    f"wrist_hint_world must have shape [B,T,{self.wrist_hint_dim}], "
                    f"got {tuple(wrist_hint_world.shape)}."
                )
            else:
                wrist_hint_world = wrist_hint_world.to(dtype=fused.dtype, device=fused.device)

        if not self.autoregressive_wrist_enabled:
            wrist_in = (
                torch.cat([fused, wrist_hint_world], dim=-1)
                if int(self.wrist_hint_dim) > 0
                else fused
            )
            wrist_raw = self.wrist_decoder(wrist_in.reshape(b * t, -1)).reshape(b, t, 9)
            wrist_raw = torch.nan_to_num(wrist_raw, nan=0.0, posinf=0.0, neginf=0.0)
            return wrist_raw, wrist_raw[..., 0:3], self._normalize_rot6d(wrist_raw[..., 3:9])

        # AR loop: carry previous predicted wrist (pos+rot6d) across timesteps.
        if wrist_prev_hint is not None:
            if wrist_prev_hint.shape != (b, t, 9):
                raise ValueError(
                    f"wrist_prev_hint must have shape [B,T,9], got {tuple(wrist_prev_hint.shape)}"
                )
            wrist_prev_hint = wrist_prev_hint.to(dtype=fused.dtype, device=fused.device)

        raw_steps: list[torch.Tensor] = []
        pos_steps: list[torch.Tensor] = []
        rot_steps: list[torch.Tensor] = []
        prev_wrist = fused.new_zeros((b, 9))
        for ti in range(t):
            prev_in = prev_wrist if wrist_prev_hint is None else wrist_prev_hint[:, ti, :]
            parts = [fused[:, ti, :]]
            if int(self.wrist_hint_dim) > 0:
                parts.append(wrist_hint_world[:, ti, :])
            parts.append(prev_in)
            dec_in = torch.cat(parts, dim=-1)
            wrist_raw_t = self.wrist_decoder(dec_in)
            wrist_raw_t = torch.nan_to_num(wrist_raw_t, nan=0.0, posinf=0.0, neginf=0.0)
            pos_t = wrist_raw_t[..., 0:3]
            rot_t = self._normalize_rot6d(wrist_raw_t[..., 3:9])
            raw_steps.append(wrist_raw_t)
            pos_steps.append(pos_t)
            rot_steps.append(rot_t)
            next_wrist = torch.cat([pos_t, rot_t], dim=-1)
            prev_wrist = next_wrist.detach() if self.autoregressive_detach_prev_wrist else next_wrist
        return (
            torch.stack(raw_steps, dim=1),
            torch.stack(pos_steps, dim=1),
            torch.stack(rot_steps, dim=1),
        )

    def forward(
        self,
        *,
        hand_feats: torch.Tensor,
        obj_feats: torch.Tensor,
        hand_mask: torch.Tensor | None = None,
        obj_mask: torch.Tensor | None = None,
        wrist_hint_world: torch.Tensor | None = None,
        q_prev_hint: torch.Tensor | None = None,
        wrist_prev_hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...] | None]:
        """Run one forward pass.

        Args:
            hand_feats: ``[B,T,H,F_hand]``
            obj_feats: ``[B,T,K,F_obj]``
            hand_mask: optional ``[B,T,H]``
            obj_mask: optional ``[B,T,K]``
            wrist_hint_world: optional wrist/root hint ``[B,T,9]`` as
                ``[pos_xyz, rot6d]`` in world frame.
            q_prev_hint: optional previous-joint hint ``[B,T,J]``.
                Used only when ``autoregressive_enabled=True``.
            wrist_prev_hint: optional previous-wrist hint ``[B,T,9]`` as
                ``[pos_xyz, rot6d]``. Used only when
                ``autoregressive_wrist_enabled=True``.
        """

        if hand_feats.ndim != 4:
            raise ValueError(f"Expected hand_feats [B,T,H,F], got {tuple(hand_feats.shape)}")
        if obj_feats.ndim != 4:
            raise ValueError(f"Expected obj_feats [B,T,K,F], got {tuple(obj_feats.shape)}")

        b, t, h, fh = hand_feats.shape
        bo, to, k, fo = obj_feats.shape
        if b != bo or t != to:
            raise ValueError("hand_feats and obj_feats must share [B,T] dimensions.")
        if int(h) != int(self.layout.hand_tokens) or int(fh) != int(self.layout.hand_feat_dim):
            raise ValueError(
                "hand_feats shape mismatch with layout; "
                f"expected H={self.layout.hand_tokens},F={self.layout.hand_feat_dim}, got {(h, fh)}"
            )
        if int(k) != int(self.layout.object_points) or int(fo) != int(self.layout.object_feat_dim):
            raise ValueError(
                "obj_feats shape mismatch with layout; "
                f"expected K={self.layout.object_points},F={self.layout.object_feat_dim}, got {(k, fo)}"
            )

        fused = self._fuse_features(
            hand_feats=hand_feats,
            obj_feats=obj_feats,
            hand_mask=hand_mask,
            obj_mask=obj_mask,
        )
        q_raw, q = self._decode_q(fused=fused, q_prev_hint=q_prev_hint)
        tips = self.robot_geometry.fingertips_from_q_torch(torch, q)
        full_fk_points = None
        full_fk_link_names: tuple[str, ...] = tuple()
        primitive_points = None
        if self.compute_full_fk:
            full_fk_points, full_fk_link_names = self.robot_geometry.link_points_from_q_torch(torch, q)
            primitive_points, _, _, _ = self.robot_geometry.primitive_points_from_q_torch(torch, q)

        wrist_raw, wrist_pos_world, wrist_rot6d_world = self._decode_wrist(
            fused=fused,
            wrist_hint_world=wrist_hint_world,
            wrist_prev_hint=wrist_prev_hint,
        )

        return {
            "q_raw": q_raw,
            "q": q,
            "tips": tips,
            "full_fk_points": full_fk_points,
            "full_fk_link_names": full_fk_link_names,
            "primitive_points": primitive_points,
            "wrist_raw": wrist_raw,
            "wrist_pos_world": wrist_pos_world,
            "wrist_rot6d_world": wrist_rot6d_world,
        }
