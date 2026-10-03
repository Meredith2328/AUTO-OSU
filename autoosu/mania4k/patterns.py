"""Lane patterns learned from ranked 4K charts.

A row is one timestamp with k simultaneous notes. Given the chord size of the next row (decided by
the note model), the pattern model predicts which of the C(4, k) lane combinations ranked mappers
would use, from the recent rows (their lanes and gaps), how long ago each lane was pressed, and
which lanes are held by long notes. Hard constraints (held lanes, minimum jack interval for the
difficulty) are applied on top of the learned distribution when decoding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

HISTORY = 8
MASKS = list(range(16))
POPCOUNT = np.array([bin(m).count("1") for m in MASKS])
MIRROR = np.array([int(f"{m:04b}"[::-1], 2) for m in MASKS])   # lane l -> 3 - l
FEAT_DIM = 4 + 2 + 4 + 4 + 1 + HISTORY * 5 + 1


def lanes_of(mask: int) -> List[int]:
    return [l for l in range(4) if mask >> l & 1]


@dataclass
class RowState:
    """Running state while walking through rows (identical for training and decoding)."""
    last_press: np.ndarray            # ms of last press per lane
    held_until: np.ndarray            # ms until which a lane is held by a long note
    history: List[tuple]              # (mask, gap_ms) of previous rows, most recent last
    last_time: Optional[float] = None

    @staticmethod
    def fresh() -> "RowState":
        return RowState(np.full(4, -1e9), np.full(4, -1e9), [])


def row_features(state: RowState, t: float, k: int, is_ln: bool, beat_ms: float, stars: float) -> np.ndarray:
    gap = 5000.0 if state.last_time is None else t - state.last_time
    f = np.zeros(FEAT_DIM, np.float32)
    f[k - 1] = 1.0
    f[4] = math.log1p(min(gap, 5000.0)) / 8.5
    f[5] = float(np.clip(math.log2(max(gap, 1.0) / beat_ms), -5, 3)) / 5
    held = state.held_until > t - 1.0
    f[6:10] = held
    f[10:14] = np.log1p(np.clip(t - state.last_press, 0, 5000)) / 8.5
    f[14] = float(is_ln)
    o = 15
    for j in range(HISTORY):
        if j < len(state.history):
            m, g = state.history[-1 - j]
            f[o:o + 4] = [(m >> l) & 1 for l in range(4)]
            f[o + 4] = math.log1p(min(g, 5000.0)) / 8.5
        o += 5
    f[o] = stars / 4.0
    return f


def advance(state: RowState, t: float, mask: int, ln_ends: Sequence[float] = (0, 0, 0, 0)) -> None:
    gap = 5000.0 if state.last_time is None else t - state.last_time
    for l in lanes_of(mask):
        state.last_press[l] = t
        if ln_ends[l] > t:
            state.held_until[l] = ln_ends[l]
    state.history.append((mask, gap))
    if len(state.history) > HISTORY:
        state.history.pop(0)
    state.last_time = t


class PatternNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 16))
        self.config = {"hidden": hidden}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> "PatternNet":
        ck = torch.load(path, map_location=device, weights_only=False)
        net = PatternNet(**ck.get("config", {}))
        net.load_state_dict(ck["model"])
        return net.to(device).eval()


def allowed_masks(state: RowState, t: float, k: int, min_jack_ms: float) -> np.ndarray:
    """Boolean (16,) of masks with k lanes, none held, none pressed less than min_jack_ms ago."""
    ok = POPCOUNT == k
    held = state.held_until > t - 1.0
    recent = (t - state.last_press) < min_jack_ms
    for m in MASKS:
        if ok[m]:
            ls = lanes_of(m)
            if any(held[l] for l in ls) or any(recent[l] for l in ls):
                ok[m] = False
    return ok
