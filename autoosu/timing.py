"""BPM and offset estimation (single constant-tempo red line)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from .audio import AudioAnalysis

BPM_MIN = 80.0
BPM_MAX = 220.0


@dataclass
class Timing:
    bpm: float
    offset_ms: float   # time of a measure downbeat, in ms

    @property
    def beat_length(self) -> float:
        return 60000.0 / self.bpm

    def beat_at(self, ms: float) -> float:
        return (ms - self.offset_ms) / self.beat_length

    def ms_at(self, beat: float) -> int:
        return int(round(self.offset_ms + beat * self.beat_length))


def _grid_scores(env: np.ndarray, fps: float, bpm: float, offsets: np.ndarray, duration: float) -> np.ndarray:
    """Mean onset strength sampled on the beat grid, for a vector of candidate offsets (s)."""
    period = 60.0 / bpm
    n_beats = max(1, int((duration - offsets.max()) / period))
    beats = offsets[:, None] + period * np.arange(n_beats)[None, :]
    idx = np.arange(len(env))
    vals = np.interp((beats * fps).ravel(), idx, env).reshape(beats.shape)
    return vals.mean(axis=1)


def _best_offset(env: np.ndarray, fps: float, bpm: float, duration: float) -> tuple[float, float]:
    period = 60.0 / bpm
    coarse = np.arange(0.0, period, 0.004)
    sc = _grid_scores(env, fps, bpm, coarse, duration)
    o = coarse[int(np.argmax(sc))]
    fine = np.arange(o - 0.005, o + 0.005, 0.0005)
    fine = fine[fine >= 0]
    if len(fine) == 0:
        fine = np.array([o])
    sf = _grid_scores(env, fps, bpm, fine, duration)
    j = int(np.argmax(sf))
    return float(fine[j]), float(sf[j])


def _fold_bpm(bpm: float) -> float:
    while bpm < BPM_MIN:
        bpm *= 2
    while bpm >= BPM_MAX:
        bpm /= 2
    return bpm


def _snap_bpm(bpm: float) -> float:
    if abs(bpm - round(bpm)) <= 0.06:
        return float(round(bpm))
    return round(bpm, 2)


def _tempogram_bpm(analysis: AudioAnalysis) -> float:
    """Strongest periodicity of the onset envelope inside the osu! tempo range (no 120 BPM prior)."""
    env, sr, hop = analysis.fine_env, analysis.sr, analysis.fine_hop
    tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=hop)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
    profile = tg.mean(axis=1)
    mask = (bpms >= BPM_MIN) & (bpms < BPM_MAX)
    if not mask.any():
        return 120.0
    return float(bpms[mask][int(np.argmax(profile[mask]))])


def _refine_bpm(env: np.ndarray, fps: float, bpm: float, duration: float, span: float = 3.0) -> float:
    """Local search around a coarse estimate, scoring the beat grid against the onset envelope."""
    best = (-1.0, bpm)
    for c in np.arange(bpm - span, bpm + span + 1e-9, 0.1):
        if c < BPM_MIN / 2:
            continue
        _, s = _best_offset(env, fps, c, duration)
        if s > best[0]:
            best = (s, float(c))
    return _snap_bpm(best[1])


def _beat_strengths(env: np.ndarray, fps: float, bpm: float, duration: float) -> tuple[float, float, float]:
    """Mean onset strength on even beats, odd beats and half-beats of the best grid."""
    off, _ = _best_offset(env, fps, bpm, duration)
    period = 60.0 / bpm
    n = int((duration - off) / period)
    beats = off + period * np.arange(n)
    idx = np.arange(len(env))
    vals = np.interp(beats * fps, idx, env)
    half = np.interp((beats + period / 2) * fps, idx, env)
    return float(vals[0::2].mean()), float(vals[1::2].mean()), float(half.mean())


def _fix_octave(env: np.ndarray, fps: float, bpm: float, duration: float) -> float:
    """Resolve the half/double ambiguity the way mappers do: off-beats nearly as strong as beats
    means double time; every other beat weak means half time."""
    even, odd, half = _beat_strengths(env, fps, bpm, duration)
    beat = max(even, odd, 1e-9)
    if half >= 0.7 * beat and bpm * 2 < BPM_MAX:
        return _snap_bpm(bpm * 2)
    if bpm > 150 and min(even, odd) < 0.55 * max(even, odd) and bpm / 2 >= BPM_MIN:
        return _snap_bpm(bpm / 2)
    return bpm


def _downbeat_phase(analysis: AudioAnalysis, bpm: float, offset: float) -> int:
    """Which of the 4 beat phases is the measure start: kicks (bass onsets) on 1 and 3,
    harmonic changes (chords / bass line) and loudness rises on 1."""
    fps = analysis.sr / analysis.fine_hop
    period = 60.0 / bpm
    n = int((analysis.duration - offset) / period)
    if n < 8:
        return 0
    beats = offset + period * np.arange(n)
    idx = np.arange(len(analysis.fine_env_low))
    # kick: sub-bass energy jumps across the beat
    kick = (np.interp((beats + 0.03) * fps, idx, analysis.fine_env_low)
            - np.interp((beats - 0.03) * fps, idx, analysis.fine_env_low))
    harm = np.interp(beats * fps, idx, analysis.fine_env_harm)
    rms_after = np.array([analysis.rms_between(b, b + period) for b in beats])
    rms_before = np.array([analysis.rms_between(max(0.0, b - period), b) for b in beats])
    louder = np.maximum(0.0, rms_after - rms_before)

    def phase_means(v: np.ndarray) -> np.ndarray:
        m = np.array([v[k::4].mean() for k in range(4)])
        return (m - m.mean()) / (m.std() + 1e-9)       # z-score across the 4 phases

    zk, zh, zl = phase_means(kick), phase_means(harm), phase_means(louder)
    score = np.array([zk[k] + 0.6 * zk[(k + 2) % 4] + 0.5 * zh[k] + 1.0 * zl[k] for k in range(4)])
    return int(np.argmax(score))


def _refine_with_waveform(analysis: AudioAnalysis, bpm: float, offset_s: float,
                          fix_bpm: bool = False, win_ms: int = 30) -> tuple[float, float]:
    """Nudge bpm/offset so beats sit on the percussive flux peaks (sub-frame precision via
    parabolic interpolation). A linear fit of the per-beat deviations corrects tempo drift."""
    env = analysis.fine_env_sharp
    fps = analysis.sr / analysis.fine_hop
    for _ in range(3):
        period_ms = 60000.0 / bpm
        n = int((analysis.duration * 1000.0 - offset_s * 1000.0) / period_ms)
        idx, devs, weights = [], [], []
        for i in range(n):
            c = offset_s * 1000.0 + i * period_ms
            lo, hi = int((c - win_ms) / 1000.0 * fps), int((c + win_ms) / 1000.0 * fps) + 1
            if lo < 1 or hi >= len(env) - 1:
                continue
            seg = env[lo:hi]
            j = int(np.argmax(seg))
            if seg[j] < 0.3:
                continue
            k = lo + j
            a, b, cc = env[k - 1], env[k], env[k + 1]
            denom = a - 2 * b + cc
            frac = 0.5 * (a - cc) / denom if abs(denom) > 1e-9 else 0.0
            peak_ms = (k + float(np.clip(frac, -0.5, 0.5))) / fps * 1000.0
            idx.append(i)
            devs.append(peak_ms - c)
            weights.append(float(seg[j]))
        if len(devs) < 8:
            return bpm, offset_s
        idx, devs, w = np.array(idx, float), np.array(devs, float), np.array(weights)
        # weighted linear fit: deviation = a + b * beat_index  ->  b is the period error in ms
        a, b = np.polyfit(idx, devs, 1, w=np.sqrt(w))[::-1]
        new_bpm = bpm if fix_bpm else _snap_bpm(60000.0 / (period_ms + b))
        if abs(new_bpm - bpm) < 1e-9:
            # tempo already right: use the robust weighted median for the offset
            order = np.argsort(devs)
            cw = np.cumsum(w[order])
            med = devs[order][min(len(devs) - 1, int(np.searchsorted(cw, cw[-1] / 2)))]
            return bpm, offset_s + med / 1000.0
        bpm = new_bpm
        offset_s = offset_s + a / 1000.0
    return bpm, offset_s


def estimate_timing(analysis: AudioAnalysis, bpm_override: Optional[float] = None,
                    offset_override_ms: Optional[float] = None) -> Timing:
    env, fps, duration = analysis.fine_env, analysis.sr / analysis.fine_hop, analysis.duration

    if bpm_override:
        bpm = float(bpm_override)
    else:
        bpm = _fold_bpm(_tempogram_bpm(analysis))
        bpm = _refine_bpm(env, fps, bpm, duration)
        fixed = _fix_octave(env, fps, bpm, duration)
        if fixed != bpm:
            bpm = _refine_bpm(env, fps, fixed, duration, span=0.5)

    if offset_override_ms is not None:
        offset_s = float(offset_override_ms) / 1000.0
    else:
        offset_s, _ = _best_offset(env, fps, bpm, duration)
        bpm, offset_s = _refine_with_waveform(analysis, bpm, offset_s, fix_bpm=bool(bpm_override))
        k = _downbeat_phase(analysis, bpm, offset_s)
        offset_s += k * 60.0 / bpm

    measure = 4 * 60.0 / bpm
    while offset_s - measure >= 0:
        offset_s -= measure
    return Timing(bpm=bpm, offset_ms=round(offset_s * 1000.0, 1))
