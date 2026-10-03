"""Music sections and their energy: what a section planner needs to know about the song."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .features import MEL_HOP
from .onsets import HOP, SR

SECTION_FEATS = ("energy", "loud", "high", "low", "attacks", "sustain", "position", "length")


def section_audio_features(mel: np.ndarray, env, attacks: np.ndarray, wins: Sequence[Tuple[float, float]],
                           bounds: Sequence[int]) -> np.ndarray:
    """Per section: energy (z-scored loudness + high-band flux), loudness, high/low-band flux, attack
    rate, sustain (loudness relative to flux: pads / long notes), relative position, length (all
    z-scored within the song where it makes sense)."""
    n = len(wins)
    mfps, efps = SR / MEL_HOP, SR / HOP
    rows = []
    for s, e in zip(bounds, list(bounds[1:]) + [n]):
        a, b = wins[s][0], wins[e - 1][1]
        fa, fb = int(a / 1000 * mfps), max(int(a / 1000 * mfps) + 1, int(b / 1000 * mfps))
        ea, eb = int(a / 1000 * efps), max(int(a / 1000 * efps) + 1, int(b / 1000 * efps))
        loud = float(mel[fa:min(fb, len(mel))].astype(np.float32).mean()) if fa < len(mel) else 0.0
        high = float(env.high[ea:eb].mean()) if ea < len(env.high) else 0.0
        low = float(env.low[ea:eb].mean()) if ea < len(env.low) else 0.0
        full = float(env.full[ea:eb].mean()) if ea < len(env.full) else 0.0
        rate = float(((attacks >= a) & (attacks < b)).sum() / max((b - a) / 1000, 1e-3))
        rows.append([loud, high, low, rate, loud / (full + 1e-3), (s + e) / 2 / n, e - s])
    r = np.array(rows, np.float64)
    z = (r - r.mean(0)) / (r.std(0) + 1e-6)
    energy = 0.5 * z[:, 0] + 0.5 * z[:, 1]
    out = np.column_stack([energy, z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], r[:, 5], r[:, 6] / 8.0])
    return out.astype(np.float32)


def energy_levels(energy: np.ndarray) -> np.ndarray:
    """0 rest, 1 low, 2 mid, 3 climax by song-relative energy."""
    if len(energy) < 3:
        return np.full(len(energy), 2)
    q = np.percentile(energy, [20, 50, 80])
    return np.digitize(energy, q)
