"""Utilities for training *one* policy across many ARCTIC sequences.

DexMachina's default training entrypoint (`dexmachina/rl/train_rl_games.py`) bakes a
single demonstration clip ("sequence") into the environment via `--clip`.
That produces an experiment and checkpoint per sequence.

This module provides a small, dependency-free sampler that can:
- expand user-provided clip specs (strings) into structured `ClipSpec` objects
- sample clips (uniformly or with optional weights)

It's used by `train_rl_games_multi_sequence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import random


@dataclass(frozen=True)
class ClipSpec:
    """A single demonstration sequence (clip) identifier.

    Matches the string format used throughout DexMachina:
      obj-start-end-subject-uXX
    Example:
      box-40-200-s01-u01
    """

    obj_name: str
    frame_start: int
    frame_end: int
    subject: str
    use_clip: str  # "01", "02", ... (no leading 'u')

    @property
    def clip_str(self) -> str:
        return f"{self.obj_name}-{self.frame_start}-{self.frame_end}-{self.subject}-u{self.use_clip}"


def parse_clip_string(clip: str) -> ClipSpec:
    vals = clip.split("-")
    if len(vals) == 3:
        # keep old convenience behavior used in the repo
        vals += ["s01", "u01"]
    if len(vals) != 5:
        raise ValueError("Clip should be in format: obj-start-end-subject-uXX")
    obj_name = vals[0]
    frame_start = int(vals[1])
    frame_end = int(vals[2])
    subject = vals[3]
    use_clip = vals[4].replace("u", "")
    return ClipSpec(obj_name, frame_start, frame_end, subject, use_clip)


def expand_clip_ranges(specs: Sequence[str]) -> List[ClipSpec]:
    """Expand CLI specs into ClipSpecs.

    Supported inputs:
      - "box-40-200-s01-u01" (single sequence)
      - "box-40-200-s01-u01..u05" (range over u indices, inclusive)

    Range expansion is intentionally minimal and only supports the last token.
    """

    out: List[ClipSpec] = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if ".." not in s:
            out.append(parse_clip_string(s))
            continue

        # minimal range support: only allow the last token to be uNN..uMM
        parts = s.split("-")
        if len(parts) != 5:
            raise ValueError(f"Invalid ranged clip spec: {s}")
        u_token = parts[-1]
        if ".." not in u_token:
            raise ValueError(f"Invalid ranged clip spec (expected range on last token): {s}")
        a, b = u_token.split("..", 1)
        if not (a.startswith("u") and b.startswith("u")):
            raise ValueError(f"Invalid ranged clip spec (expected uNN..uMM): {s}")
        start_u = int(a[1:])
        end_u = int(b[1:])
        if end_u < start_u:
            raise ValueError(f"Invalid ranged clip spec (end < start): {s}")

        base = "-".join(parts[:-1])
        for u in range(start_u, end_u + 1):
            out.append(parse_clip_string(f"{base}-u{u:02d}"))

    if not out:
        raise ValueError("No clips provided after parsing.")
    return out


class ClipSampler:
    """Uniform random sampler over a fixed set of clips."""

    def __init__(self, clips: Sequence[ClipSpec], *, seed: int = 0):
        if not clips:
            raise ValueError("clips must be non-empty")
        self._clips = list(clips)
        self._rng = random.Random(seed)

    @property
    def clips(self) -> Tuple[ClipSpec, ...]:
        return tuple(self._clips)

    def sample(self) -> ClipSpec:
        return self._rng.choice(self._clips)
