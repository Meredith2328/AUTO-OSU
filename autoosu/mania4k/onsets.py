"""Precise onset envelopes shared by the timing engine, note placement and the verifier."""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SR = 22050
HOP = 64                      # 2.9 ms frames
N_FFT = 512
ENV_LATENCY_MS = -6.0         # flux peak leads the true onset by ~6 ms (window edge; measured on synthetic hits)


@dataclass
class OnsetEnvelopes:
    fps: float
    full: np.ndarray          # spectral flux of the full mix, 0..1 (95th pct -> ~1)
    low: np.ndarray           # < 180 Hz (kick, bass)
    mid: np.ndarray           # 180 Hz .. 3 kHz (snare, vocals, instruments)
    high: np.ndarray          # > 3 kHz (hats, cymbals, transients)
    duration: float

    def at(self, env: np.ndarray, t_ms: np.ndarray | float, window_ms: float = 0.0) -> np.ndarray:
        """Envelope value at times (ms), optionally the max over +-window_ms."""
        t = np.atleast_1d(np.asarray(t_ms, float)) + ENV_LATENCY_MS
        if window_ms <= 0:
            return np.interp(t / 1000.0 * self.fps, np.arange(len(env)), env)
        r = max(1, int(round(window_ms / 1000.0 * self.fps)))
        idx = np.clip(np.round(t / 1000.0 * self.fps).astype(int), 0, len(env) - 1)
        out = np.empty(len(idx))
        for k, i in enumerate(idx):
            out[k] = env[max(0, i - r):i + r + 1].max()
        return out

    def peak_near(self, t_ms: float, window_ms: float, env: np.ndarray | None = None) -> tuple[float, float]:
        """Sub-frame time (ms) and height of the strongest envelope peak within +-window_ms."""
        env = self.full if env is None else env
        c = (t_ms + ENV_LATENCY_MS) / 1000.0 * self.fps
        r = window_ms / 1000.0 * self.fps
        lo, hi = max(1, int(np.floor(c - r))), min(len(env) - 2, int(np.ceil(c + r)))
        if hi <= lo:
            return t_ms, 0.0
        k = lo + int(np.argmax(env[lo:hi + 1]))
        a, b, cc = env[k - 1], env[k], env[k + 1]
        den = a - 2 * b + cc
        frac = float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
        return (k + frac) / self.fps * 1000.0 - ENV_LATENCY_MS, float(b)


def onset_peaks(env: OnsetEnvelopes, curve: np.ndarray | None = None, min_height: float = 0.12,
                min_sep_ms: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Onset times (ms, latency corrected, sub-frame) and heights of an envelope's local maxima."""
    x = env.full if curve is None else curve
    r = max(1, int(round(min_sep_ms / 1000.0 * env.fps)))
    if len(x) < 3:
        return np.zeros(0), np.zeros(0)
    from scipy.ndimage import maximum_filter1d

    mx = maximum_filter1d(x, size=2 * r + 1, mode="nearest")
    k = np.where((x == mx) & (x >= min_height))[0]
    k = k[(k > 0) & (k < len(x) - 1)]
    if len(k) > 1:                                  # plateaus: keep the first frame
        k = k[np.concatenate([[True], np.diff(k) > r])]
    a, b, c = x[k - 1], x[k], x[k + 1]
    den = a - 2 * b + c
    frac = np.where(np.abs(den) > 1e-12, np.clip(0.5 * (a - c) / np.where(den == 0, 1, den), -0.5, 0.5), 0.0)
    return (k + frac) / env.fps * 1000.0 - ENV_LATENCY_MS, b.astype(float)


def _flux(S: np.ndarray) -> np.ndarray:
    logS = np.log1p(100.0 * S)
    d = np.diff(logS, axis=1, prepend=logS[:, :1])
    return np.maximum(d, 0.0).sum(axis=0)


def _norm(x: np.ndarray) -> np.ndarray:
    ref = float(np.percentile(x, 99.5)) if x.size else 1.0
    return np.clip(x / max(ref, 1e-9), 0.0, 1.5)


def onset_envelopes(y: np.ndarray, sr: int = SR) -> OnsetEnvelopes:
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lo, hi = freqs < 180, freqs >= 3000
    mid = ~(lo | hi)
    return OnsetEnvelopes(SR / HOP, _norm(_flux(S)), _norm(_flux(S[lo])), _norm(_flux(S[mid])),
                          _norm(_flux(S[hi])), len(y) / SR)


ATTACK_WINDOW_MS = 8.0


def attack_times(env: OnsetEnvelopes, min_height: float = 0.1, min_prominence: float = 1.5) -> np.ndarray:
    """Times (ms) of distinct attacks: envelope peaks of any band that stand out from their
    surroundings (height >= min_prominence x the band's running median over 300 ms)."""
    from scipy.ndimage import median_filter

    out = []
    size = int(0.3 * env.fps) | 1
    for curve in (env.full, env.low, env.mid, env.high):
        t, h = onset_peaks(env, curve, min_height=min_height, min_sep_ms=20.0)
        if len(t) == 0:
            continue
        base = median_filter(curve, size=size, mode="nearest")
        fr = np.clip(np.round((t + ENV_LATENCY_MS) / 1000.0 * env.fps).astype(int), 0, len(curve) - 1)
        out.append(t[h >= min_prominence * np.maximum(base[fr], 1e-3)])
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def near_attack(times: np.ndarray, attacks: np.ndarray, window_ms: float = ATTACK_WINDOW_MS) -> np.ndarray:
    """True for each time with an attack within +-window_ms."""
    times = np.asarray(times, float)
    if len(attacks) == 0:
        return np.zeros(len(times), bool)
    j = np.searchsorted(attacks, times)
    d = np.full(len(times), np.inf)
    for jj in (j - 1, j):
        v = (jj >= 0) & (jj < len(attacks))
        d[v] = np.minimum(d[v], np.abs(attacks[jj[v]] - times[v]))
    return d <= window_ms
