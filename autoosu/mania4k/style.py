"""Chart style (archetype) from the music.

Ranked mappers pick 乱 (speed), 叠 (jack), LN or 混合 (hybrid) for a song partly by taste and partly
by what the music offers: sustained pads and vocals invite LN, heavy on-beat drums invite jacks,
fast hats and busy rhythm invite speed. ``song_descriptors`` summarises a song in a handful of
numbers the generator already computes (mel loudness, band fluxes, attack rate, BPM);
``archetype_probs`` is a multinomial logistic regression on those plus the star rating, fitted on
the train split of the ranked corpus by ``scripts/mania4k_fit_style.py`` (weights/style.json).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

STYLE_FILE = Path(__file__).resolve().parent / "weights" / "style.json"
DESCRIPTORS = ("log_bpm", "attack_rate", "sustain", "loud_cv", "low", "mid", "high", "high_share",
               "low_share", "centroid", "flux_cv", "offbeat", "stars")


def song_descriptors(mel: np.ndarray, env_full: np.ndarray, env_low: np.ndarray, env_mid: np.ndarray,
                     env_high: np.ndarray, env_fps: float, bpm: float, stars: float) -> np.ndarray:
    """Song-level audio summary (order: DESCRIPTORS)."""
    from scipy.ndimage import maximum_filter1d

    m = mel.astype(np.float32)
    loud = m.mean(axis=1)
    active = loud >= np.percentile(loud, 20)                 # ignore silence / fade-outs
    bins = np.arange(m.shape[1], dtype=np.float32)
    centroid = float(((m[active] * bins).sum(1) / (m[active].sum(1) + 1e-6)).mean() / m.shape[1])
    full, low, mid, high = (np.asarray(e, np.float32) for e in (env_full, env_low, env_mid, env_high))
    r = max(1, int(round(0.03 * env_fps)))
    peaks = np.where((full == maximum_filter1d(full, 2 * r + 1)) & (full >= 0.25))[0]
    dur_s = max(len(full) / env_fps, 1.0)
    attack_rate = len(peaks) / dur_s
    # share of attacks on the off-eighths of the tracker-free BPM grid (syncopation, busy rhythm)
    beat_frames = 60.0 / bpm * env_fps
    ph = (peaks / beat_frames) % 1.0
    offbeat = float(np.mean(np.abs(ph - 0.5) < 0.15)) if len(ph) else 0.0
    bands = np.array([low.mean(), mid.mean(), high.mean()]) + 1e-6
    q = loud[active]
    return np.array([
        np.log2(bpm / 150.0), attack_rate / 5.0, float(q.mean() / 255.0) / (float(full.mean()) + 1e-3) / 10.0,
        float(q.std() / (q.mean() + 1e-6)), bands[0], bands[1], bands[2],
        bands[2] / bands.sum(), bands[0] / bands.sum(), centroid,
        float(full.std() / (full.mean() + 1e-6)) / 5.0, offbeat, stars / 5.0], np.float64)


@lru_cache(maxsize=1)
def load_style() -> dict | None:
    return json.loads(STYLE_FILE.read_text(encoding="utf-8")) if STYLE_FILE.exists() else None


def archetype_probs(x: np.ndarray) -> np.ndarray | None:
    """P(archetype | song descriptors), or None without fitted weights."""
    w = load_style()
    if w is None:
        return None
    z = (np.asarray(x, float) - np.array(w["mean"])) / np.array(w["std"])
    logit = np.array(w["coef"]) @ z + np.array(w["intercept"])
    p = np.exp(logit - logit.max())
    return p / p.sum()


def analysis_descriptors(an, stars: float) -> np.ndarray:
    """song_descriptors for a generate.SongAnalysis."""
    from .chart import main_bpm

    e = an.features.env
    return song_descriptors(an.features.mel, e.full, e.low, e.mid, e.high, e.fps,
                            main_bpm(an.timing.red_lines, an.duration_ms), stars)
