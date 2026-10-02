"""Per-song audio features and the beat grid the note model works on.

Everything here runs identically for training (grid from a ranked chart's red lines) and for
generation (grid from :mod:`autoosu.mania4k.timing`), always in *audio* time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import librosa
import numpy as np

from .chart import RedLine
from .onsets import SR, OnsetEnvelopes, onset_envelopes

MEL_HOP = 256                  # 11.6 ms
N_MELS = 64
PATCH = 16                     # mel frames per tick patch (-8 .. +7, ~ +-93 ms)
ENV_TAPS = np.arange(-24.0, 24.1, 3.0)   # ms offsets at which the onset envelopes are sampled

# beat subdivisions the model may place notes on: union of 1/8 and 1/6 (12 positions per beat)
POSITIONS = sorted({i / 8 for i in range(8)} | {i / 6 for i in range(6)})
POS_DIV = [min(d for d in (1, 2, 4, 8, 3, 6) if abs(p * d - round(p * d)) < 1e-9) for p in POSITIONS]
DIV_CLASSES = (1, 2, 4, 8, 3, 6)


@dataclass
class SongFeatures:
    mel: np.ndarray            # (frames, N_MELS) uint8 log-mel
    env: OnsetEnvelopes
    duration_ms: float

    @property
    def mel_fps(self) -> float:
        return SR / MEL_HOP


def song_features(y: np.ndarray, sr: int) -> SongFeatures:
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=MEL_HOP, n_mels=N_MELS,
                                       fmin=30, fmax=11000, power=2.0)
    db = librosa.power_to_db(m, ref=np.max, top_db=80.0)            # -80 .. 0
    mel = np.clip((db + 80.0) * (255.0 / 80.0), 0, 255).astype(np.uint8).T
    return SongFeatures(mel, onset_envelopes(y, SR), len(y) / SR * 1000.0)


@dataclass
class Grid:
    times: np.ndarray          # ms (audio time)
    pos: np.ndarray            # index into POSITIONS
    div: np.ndarray            # index into DIV_CLASSES
    beat_in_measure: np.ndarray
    beat_ms: np.ndarray
    beat: np.ndarray           # absolute beat number from the first red line (float)

    def __len__(self) -> int:
        return len(self.times)


def build_grid(reds: Sequence[RedLine], start_ms: float, end_ms: float) -> Grid:
    times: List[float] = []
    pos: List[int] = []
    div: List[int] = []
    bim: List[int] = []
    bms: List[float] = []
    beats: List[float] = []
    beat_base = 0.0
    for n, r in enumerate(reds):
        seg_end = reds[n + 1].time if n + 1 < len(reds) else end_ms
        k0 = math.floor((max(start_ms, r.time) - r.time) / r.beat_ms) if n else math.floor((start_ms - r.time) / r.beat_ms)
        k = k0
        while True:
            bt = r.time + k * r.beat_ms
            if bt >= seg_end - 0.5 or bt > end_ms:
                break
            for i, p in enumerate(POSITIONS):
                t = bt + p * r.beat_ms
                if t < start_ms or t >= seg_end - 0.5 or t > end_ms:
                    continue
                if times and t <= times[-1] + 0.5:
                    continue
                times.append(t)
                pos.append(i)
                div.append(DIV_CLASSES.index(POS_DIV[i]))
                bim.append(int(k % max(1, r.meter)))
                bms.append(r.beat_ms)
                beats.append(beat_base + k + p)
            k += 1
        beat_base += max(0.0, (seg_end - r.time) / r.beat_ms) if n + 1 < len(reds) else 0.0
    return Grid(np.asarray(times), np.asarray(pos, np.int64), np.asarray(div, np.int64),
                np.asarray(bim, np.int64), np.asarray(bms, np.float32), np.asarray(beats, np.float32))


def env_matrix(env: OnsetEnvelopes, times: np.ndarray) -> np.ndarray:
    """(n, 4 * len(ENV_TAPS)) onset envelope samples around each tick."""
    cols = []
    for e in (env.full, env.low, env.mid, env.high):
        cols.append(np.stack([env.at(e, times + o) for o in ENV_TAPS], axis=1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def mel_index(times: np.ndarray) -> np.ndarray:
    """Centre mel frame of each tick."""
    return np.round(times / 1000.0 * SR / MEL_HOP).astype(np.int64)


def local_loudness(mel: np.ndarray, times: np.ndarray, window_s: float = 2.0) -> np.ndarray:
    """Song-relative loudness (0..1) around each tick, a proxy for the section's energy."""
    frame_energy = mel.astype(np.float32).mean(axis=1)
    k = max(1, int(window_s * SR / MEL_HOP))
    smooth = np.convolve(frame_energy, np.ones(k) / k, mode="same")
    lo, hi = np.percentile(smooth, 5), np.percentile(smooth, 98)
    norm = np.clip((smooth - lo) / max(hi - lo, 1e-6), 0, 1)
    idx = np.clip(mel_index(times), 0, len(norm) - 1)
    return norm[idx].astype(np.float32)
