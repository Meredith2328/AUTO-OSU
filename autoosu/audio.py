"""Audio loading and feature extraction (onsets, energy, harmonic/percussive split)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


@dataclass
class Onset:
    time: float        # seconds
    strength: float    # 0..1 (normalised onset strength)
    percussive: float  # 0..1, share of percussive energy in this onset
    sustain: float     # seconds the harmonic energy keeps ringing after the onset


@dataclass
class AudioAnalysis:
    sr: int
    hop: int
    duration: float
    times: np.ndarray            # frame centre times (s)
    onset_env: np.ndarray        # full-mix onset strength, 0..1
    onset_env_perc: np.ndarray
    onset_env_harm: np.ndarray
    band_low: np.ndarray         # percussive onset strength < 200 Hz (kicks), 0..1
    band_mid: np.ndarray         # 200-2500 Hz (snares, claps), 0..1
    band_high: np.ndarray        # > 2500 Hz (hi-hats), 0..1
    rms: np.ndarray              # smoothed loudness, 0..1
    rms_harm: np.ndarray
    fine_hop: int
    fine_env: np.ndarray         # percussive onset strength at fine resolution (grid search)
    fine_env_sharp: np.ndarray   # same with a short STFT window: narrow peaks for offset refinement
    fine_env_low: np.ndarray     # sub-bass energy level (kicks) at fine resolution, 0..1
    fine_env_harm: np.ndarray    # harmonic onset strength (chord changes) at fine resolution
    rise_ms: np.ndarray          # energy rise of the raw waveform at 1 ms resolution
    onsets: List[Onset] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    def frame_at(self, t: float) -> int:
        return int(np.clip(round(t * self.fps), 0, len(self.times) - 1))

    def rms_between(self, t0: float, t1: float) -> float:
        a, b = self.frame_at(t0), self.frame_at(t1)
        if b <= a:
            return float(self.rms[a])
        return float(self.rms[a:b].mean())


def load_audio(path: str | Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Mono waveform at `sr` from any audio/video file (libsndfile, else ffmpeg)."""
    from .audio_io import decode

    return decode(path, sr=sr, mono=True)


def _normalise(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    if x.size == 0:
        return x
    m = float(np.percentile(x, pct))
    return np.clip(x / max(m, 1e-9), 0.0, 1.0)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size < n:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")


def analyze(y: np.ndarray, sr: int, hop: int = 256) -> AudioAnalysis:
    """Run the whole feature pipeline on a mono signal."""
    y_harm, y_perc = librosa.effects.hpss(y)

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_p, band_low, band_mid, band_high = _percussive_env(y_perc, sr, hop)
    env_h = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=hop)
    n = min(len(env), len(env_p), len(env_h))
    env, env_p, env_h = env[:n], env_p[:n], env_h[:n]
    band_low, band_mid, band_high = band_low[:n], band_mid[:n], band_high[:n]
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

    rms = _fit_len(librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0], n)
    rms_h = _fit_len(librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=hop)[0], n)
    rms = _normalise(_smooth(rms, 5), 98)
    rms_h = _normalise(_smooth(rms_h, 5), 98)

    fine_hop = 128
    fine_env = _normalise(librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=fine_hop))
    # short window: narrow flux peaks close to the transient start; hi-hat band weighted down
    freqs = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)
    i_hi = int(np.searchsorted(freqs, 2500.0))
    lowmid, high = librosa.onset.onset_strength_multi(y=y_perc, sr=sr, hop_length=fine_hop, n_fft=512,
                                                       n_mels=64, channels=[0, i_hi, 64])
    fine_sharp = _normalise(np.maximum(lowmid / max(float(np.percentile(lowmid, 99)), 1e-9),
                                       0.4 * high / max(float(np.percentile(high, 99)), 1e-9)))
    sub = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=fine_hop, n_mels=8, fmax=120.0).sum(axis=0)
    fine_low = _normalise(sub, 99)
    fine_harm = _normalise(librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=fine_hop))

    onsets = _detect_onsets(env_p, env_h, rms_h, sr, hop)

    return AudioAnalysis(
        sr=sr, hop=hop, duration=len(y) / sr, times=times,
        onset_env=_normalise(env), onset_env_perc=_normalise(env_p), onset_env_harm=_normalise(env_h),
        band_low=band_low, band_mid=band_mid, band_high=band_high,
        rms=rms, rms_harm=rms_h, fine_hop=fine_hop, fine_env=fine_env, fine_env_sharp=fine_sharp,
        fine_env_low=fine_low, fine_env_harm=fine_harm, rise_ms=_energy_rise_ms(y, sr), onsets=onsets,
    )


def _energy_rise_ms(y: np.ndarray, sr: int) -> np.ndarray:
    """Positive energy change of the raw waveform per millisecond. Peaks sit on drum transients."""
    block = max(1, sr // 1000)
    nb = len(y) // block
    if nb < 8:
        return np.zeros(nb)
    e = np.sqrt((y[: nb * block] ** 2).reshape(nb, block).mean(axis=1))
    # resample the block envelope onto an exact 1 ms grid (block length is not exactly 1 ms)
    block_ms = np.arange(nb) * block * 1000.0 / sr
    grid_ms = np.arange(0.0, block_ms[-1], 1.0)
    e = np.interp(grid_ms, block_ms, e)
    e = _smooth(e, 3)
    rise = np.maximum(0.0, e - np.roll(e, 4))
    rise[:4] = 0.0
    return _normalise(rise, 99.5)


def _percussive_env(y_perc: np.ndarray, sr: int, hop: int, n_mels: int = 128):
    """Onset strength of the percussive part split into kick / snare / hat bands, plus a combined
    envelope with the hi-hat band weighted down so kicks and snares dominate note selection."""
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
    i1 = int(np.searchsorted(freqs, 200.0))
    i2 = int(np.searchsorted(freqs, 2500.0))
    bands = librosa.onset.onset_strength_multi(y=y_perc, sr=sr, hop_length=hop, n_mels=n_mels,
                                               channels=[0, i1, i2, n_mels])
    low, mid, high = (np.clip(b / max(float(np.percentile(b, 99)), 1e-9), 0, 1) for b in bands)
    return np.maximum.reduce([low, mid, 0.5 * high]), low, mid, high


def _peak_frames(env: np.ndarray, sr: int, hop: int, delta: float) -> np.ndarray:
    fps = sr / hop
    w = max(1, int(round(0.03 * fps)))
    return librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", backtrack=False,
        pre_max=w, post_max=w, pre_avg=int(0.1 * fps) + 1, post_avg=int(0.1 * fps) + 1,
        delta=delta, wait=max(1, int(round(0.05 * fps))),
    )


def _detect_onsets(env_p: np.ndarray, env_h: np.ndarray, rms_h: np.ndarray, sr: int, hop: int) -> List[Onset]:
    fps = sr / hop
    fp = _peak_frames(env_p, sr, hop, delta=0.04)
    fh = _peak_frames(env_h, sr, hop, delta=0.07)

    def norm_peaks(env, frames):
        if len(frames) == 0:
            return {}
        vals = env[frames]
        ref = max(float(np.percentile(vals, 95)), 1e-9)
        return {int(f): float(min(v / ref, 1.0)) for f, v in zip(frames, vals)}

    perc = norm_peaks(env_p, fp)
    harm = norm_peaks(env_h, fh)

    tol = max(1, int(round(0.025 * fps)))
    merged: dict[int, Tuple[float, float]] = {}   # frame -> (perc strength, harm strength)
    for f, s in perc.items():
        merged[f] = (s, 0.0)
    for f, s in harm.items():
        near = [g for g in range(f - tol, f + tol + 1) if g in merged]
        if near:
            g = min(near, key=lambda g: abs(g - f))
            ps, hs = merged[g]
            merged[g] = (ps, max(hs, s))
        else:
            merged[f] = (0.0, s)

    frames = sorted(merged)
    harm_frames = np.array(sorted(f for f, (_, hs) in merged.items() if hs >= 0.3), dtype=int)
    n = len(rms_h)
    max_sustain = int(3.0 * fps)
    onsets: List[Onset] = []
    for i, f in enumerate(frames):
        ps, hs = merged[f]
        strength = max(ps, hs * 0.9)
        share = ps / (ps + hs + 1e-9)
        # sustain: how long the harmonic energy stays above half of its level at the onset,
        # cut short only by the next pitched onset (drums do not end a held note)
        ref = float(rms_h[f:min(n, f + 3)].max()) if f < n else 0.0
        k_next = int(np.searchsorted(harm_frames, f + tol + 1))
        limit = int(harm_frames[k_next]) if k_next < len(harm_frames) else n
        limit = min(limit, f + max_sustain, n)
        k = f
        if ref > 0.05:
            while k < limit and rms_h[k] >= 0.5 * ref:
                k += 1
        onsets.append(Onset(time=f / fps, strength=strength, percussive=share, sustain=(k - f) / fps))
    return onsets
