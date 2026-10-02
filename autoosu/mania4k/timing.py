"""Exact timing for generated charts: red lines that the song's onsets snap to.

The quantity optimised is the one that decides whether notes "hit": how many of the song's
attacks land on the 1/4 grid. The pipeline:

1. A neural beat/downbeat tracker (Beat This!, Foscarin et al., ISMIR 2024) gives beat and
   downbeat activations at 50 fps. Its dominant inter-beat interval fixes the tempo *basin*
   and the metrical level; its activations later choose which quarter is the beat and which
   beat is the downbeat.
2. Inside that basin, period and phase are searched coarse-to-fine (to ~0.001 ms per beat) so
   that the onset peaks of a 2.9 ms flux envelope snap to the 1/4 grid of the whole song.
3. A windowed scan finds stretches whose onsets fall off that grid. Each stretch is refitted on
   its own; if a different tempo explains it clearly better, the song gets a tempo change there.
   Songs that no constant grid explains (live, rubato) get a red line per beat, locked to onsets.
4. BPMs are snapped to the simplest value (integer, .5, .25 ...) whose drift over the segment
   stays below 2 ms, and every red line sits on a downbeat.

The engine is audited against human red lines of ranked maps by ``scripts/mania4k_eval_timing.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .chart import RedLine
from .onsets import OnsetEnvelopes, onset_peaks

TRACKER_FPS = 50.0
SNAP_TOL_MS = 7.0


@dataclass
class Segment:
    start_ms: float
    end_ms: float
    period: float             # ms per beat
    phase: float              # time of one beat (ms)
    snap_rate: float = 0.0    # share of strong onsets on the 1/4|1/3 grid
    downbeat: Optional[float] = None
    per_beat: bool = False    # rubato fallback: one red line per tracker beat
    beats: List[float] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        return 60000.0 / self.period


@dataclass
class TimingResult:
    red_lines: List[RedLine]
    segments: List[Segment]
    snap_rate: float          # share of strong onsets on the final grid
    kind: str                 # constant | changes | per_beat


# --------------------------------------------------------------------------- tracker

def tracker_activations(y: np.ndarray, sr: int, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat logits at 50 fps from Beat This! (see beattrack.py)."""
    from .beattrack import activations

    return activations(y, sr, device)


def pick_beats(logit: np.ndarray, fps: float = TRACKER_FPS, threshold: float = 0.0) -> np.ndarray:
    """Local maxima of the beat logit above threshold (+-60 ms), sub-frame refined, ms."""
    x = np.asarray(logit, float)
    out: List[float] = []
    last = -10
    r = 3
    for i in range(1, len(x) - 1):
        if x[i] <= threshold or x[i] < x[max(0, i - r):i + r + 1].max() or i - last <= r:
            continue
        a, b, c = x[i - 1], x[i], x[i + 1]
        den = a - 2 * b + c
        frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if abs(den) > 1e-9 else 0.0
        out.append((i + frac) / fps * 1000.0)
        last = i
    return np.array(out)


def dominant_period(beats: np.ndarray) -> float:
    """Most common inter-beat interval (duration-weighted histogram mode), ms."""
    ibi = np.diff(beats)
    ibi = ibi[(ibi > 150) & (ibi < 1500)]
    if len(ibi) == 0:
        return 500.0
    h, e = np.histogram(ibi, bins=np.arange(150, 1500, 4.0), weights=ibi)
    h = np.convolve(h, np.ones(5), mode="same")
    k = int(np.argmax(h))
    near = ibi[np.abs(ibi - 0.5 * (e[k] + e[k + 1])) < 12]
    return float(np.median(near)) if len(near) else float(0.5 * (e[k] + e[k + 1]))


def fold_bpm_octave(period: float, lo_bpm: float = 100.0, hi_bpm: float = 250.0) -> float:
    """Ranked 4K convention: the main BPM lies between 100 and 250."""
    while 60000.0 / period < lo_bpm:
        period /= 2.0
    while 60000.0 / period >= hi_bpm:
        period *= 2.0
    return period


# --------------------------------------------------------------------------- snap scoring

def _phase_score(o: np.ndarray, w: np.ndarray, g: float, kernel_ms: float) -> Tuple[float, float]:
    """Best phase (mod g) of a grid with spacing g and its kernel-weighted onset score."""
    nb = max(16, int(round(g / 0.5)))
    idx = np.minimum(((o % g) / g * nb).astype(int), nb - 1)
    h = np.bincount(idx, weights=w, minlength=nb)
    rad = max(1, int(round(kernel_ms / (g / nb))))
    kern = np.zeros(nb)
    for d in range(-rad, rad + 1):
        kern[d % nb] += 1.0 - abs(d) / (rad + 1)
    hc = np.real(np.fft.ifft(np.fft.fft(h) * np.fft.fft(kern)))
    j = int(np.argmax(hc))
    return float(hc[j]), (j + 0.5) / nb * g


def search_period(o: np.ndarray, w: np.ndarray, lo: float, hi: float) -> Tuple[float, float, float]:
    """Period (ms/beat) and phase whose 1/4 grid best fits the onsets; coarse to fine."""
    best = (-1.0, 0.5 * (lo + hi), 0.0)
    stages = ((lo, hi, 0.01, 12.0), (None, 0.08, 0.002, 6.0), (None, 0.004, 0.0002, 3.0))
    for a, half, step, kern in stages:
        if a is not None:
            cands = np.arange(a, hi + 1e-9, step)
        else:
            cands = np.arange(best[1] - half, best[1] + half + 1e-12, step)
        stage_best = (-1.0, best[1], best[2])
        for p in cands:
            s, ph = _phase_score(o, w, p / 4.0, kern)
            if s > stage_best[0]:
                stage_best = (s, float(p), ph)
        best = stage_best
    return best[1], best[2], best[0]


def snap_errors(o: np.ndarray, period: float, phase: float) -> np.ndarray:
    """Distance (ms) of each onset to the nearest 1/4 or 1/3 grid point."""
    x = (o - phase) / period
    e4 = np.abs(x * 4 - np.round(x * 4)) * period / 4
    e3 = np.abs(x * 3 - np.round(x * 3)) * period / 3
    return np.minimum(e4, e3)


def snap_rate(o: np.ndarray, w: np.ndarray, period: float, phase: float, tol: float = SNAP_TOL_MS) -> float:
    if len(o) == 0:
        return 0.0
    return float(np.sum(w * (snap_errors(o, period, phase) <= tol)) / max(np.sum(w), 1e-9))


def refine_phase(o: np.ndarray, w: np.ndarray, period: float, phase: float, tol: float = 10.0) -> float:
    """Weighted median signed deviation of nearby onsets from the 1/4 grid, added to the phase."""
    g = period / 4
    x = (o - phase) / g
    d = (x - np.round(x)) * g
    m = np.abs(d) <= tol
    if m.sum() < 4:
        return phase
    dd, ww = d[m], w[m]
    order = np.argsort(dd)
    cw = np.cumsum(ww[order])
    return phase + float(dd[order][int(np.searchsorted(cw, cw[-1] / 2))])


def snap_bpm(period: float, n_beats: int, tol_ms: float = 2.0) -> float:
    """Simplest BPM whose accumulated drift over half the segment stays below tol_ms."""
    bpm = 60000.0 / period
    for q in (1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01):
        cand = round(bpm / q) * q
        budget = 3.0 * tol_ms if q == 1.0 else tol_ms      # DAW-made songs have integer BPMs
        if abs(60000.0 / cand - period) * max(1, n_beats) / 2.0 <= budget:
            return cand
    return round(bpm, 3)


# --------------------------------------------------------------------------- metrical position

def beat_quarter(act: np.ndarray, period: float, phase: float, t0: float, t1: float) -> float:
    """Which of the four 1/4 offsets of `phase` is the beat (max tracker activation)."""
    best, best_q = -1.0, 0
    for q in range(4):
        ph = phase + q * period / 4
        k0, k1 = math.ceil((t0 - ph) / period), math.floor((t1 - ph) / period)
        ts = ph + np.arange(k0, k1 + 1) * period
        fr = np.clip(np.round(ts / 1000.0 * TRACKER_FPS).astype(int), 0, len(act) - 1)
        s = float(act[fr].sum()) if len(fr) else 0.0
        if s > best:
            best, best_q = s, q
    return phase + best_q * period / 4


def choose_meter(beat_times: Sequence[float], down_act: np.ndarray) -> int:
    fr = np.clip(np.round(np.asarray(beat_times) / 1000.0 * TRACKER_FPS).astype(int), 0, len(down_act) - 1)
    act = down_act[fr]
    best, best_m = -1.0, 4
    for m in (4, 3):
        if len(act) < 4 * m:
            continue
        sums = np.array([act[k::m].mean() for k in range(m)])
        contrast = sums.max() - np.delete(sums, sums.argmax()).mean()
        if contrast > best * (1.3 if m == 3 else 1.0):    # 4/4 unless 3/4 is clearly better
            best, best_m = contrast, m
    return best_m


def downbeat_of(down_act: np.ndarray, period: float, phase: float, t0: float, t1: float, meter: int) -> float:
    k0, k1 = math.ceil((t0 - phase) / period), math.floor((t1 - phase) / period)
    ts = phase + np.arange(k0, k1 + 1) * period
    if len(ts) == 0:
        return phase
    fr = np.clip(np.round(ts / 1000.0 * TRACKER_FPS).astype(int), 0, len(down_act) - 1)
    a = down_act[fr]
    sums = [a[k::meter].sum() for k in range(meter)]
    return float(ts[int(np.argmax(sums))])


# --------------------------------------------------------------------------- segmentation

def _windows(o: np.ndarray, w: np.ndarray, period: float, phase: float, t0: float, t1: float,
             beats_per_window: int = 8, hop_beats: int = 2) -> List[Tuple[float, float, float, float]]:
    """(start, end, local snap rate, onset weight) of sliding windows."""
    out = []
    span, hop = beats_per_window * period, hop_beats * period
    t = t0
    while t + span <= t1 + hop:
        m = (o >= t) & (o < t + span)
        out.append((t, t + span, snap_rate(o[m], w[m], period, phase) if m.any() else 1.0, float(w[m].sum())))
        t += hop
    return out


def _fit_region(o: np.ndarray, w: np.ndarray, center: float, rel: float) -> Tuple[float, float, float]:
    p, ph, _ = search_period(o, w, center * (1 - rel), center * (1 + rel))
    ph = refine_phase(o, w, p, ph)
    return p, ph, snap_rate(o, w, p, ph)


def _local_best(o: np.ndarray, w: np.ndarray, center: float, rel: float = 0.08) -> Tuple[float, float, float]:
    """Coarse best grid for a short window: period, phase, snap rate."""
    best = (-1.0, center, 0.0)
    for p in np.arange(center * (1 - rel), center * (1 + rel), 0.25):
        sc, ph = _phase_score(o, w, p / 4.0, 6.0)
        if sc > best[0]:
            best = (sc, float(p), ph)
    _, p, ph = best
    ph = refine_phase(o, w, p, ph)
    return p, ph, snap_rate(o, w, p, ph)


def tracker_runs(beats: np.ndarray, period: float, rel: float = 0.02, min_beats: int = 8
                 ) -> List[Tuple[float, float]]:
    """Stretches where the tracker's local tempo (median of 9 IBIs, same metrical level) departs
    from the global period by more than `rel`."""
    if len(beats) < 2 * min_beats:
        return []
    ibi = np.diff(beats)
    loc = np.array([np.median(ibi[max(0, i - 4):i + 5]) for i in range(len(ibi))])
    for i in range(len(loc)):
        while loc[i] < period / 1.45:
            loc[i] *= 2
        while loc[i] > period * 1.45:
            loc[i] /= 2
    off = np.abs(loc / period - 1) > rel
    runs, i = [], 0
    while i < len(off):
        if off[i]:
            j = i
            while j < len(off) and off[j]:
                j += 1
            if j - i >= min_beats:
                runs.append((float(beats[i]), float(beats[j])))
            i = j
        else:
            i += 1
    return runs


def find_segments(o: np.ndarray, w: np.ndarray, beats: np.ndarray, glob: Segment) -> List[Segment]:
    """Split where a different constant grid explains the onsets clearly better than the global one."""
    span, hop = 12 * glob.period, 4 * glob.period
    wins = []
    t = glob.start_ms
    while t < glob.end_ms - 4 * glob.period:
        m = (o >= t) & (o < t + span)
        if w[m].sum() >= 1.0 and m.sum() >= 8:
            r_g = snap_rate(o[m], w[m], glob.period, glob.phase)
            _, _, r_l = _local_best(o[m], w[m], glob.period)
            wins.append((t, t + span, r_l - r_g >= 0.15 and r_l >= 0.5))
        else:
            wins.append((t, t + span, False))
        t += hop
    flags = [b for _, _, b in wins]
    for i in range(1, len(flags) - 1):          # bridge single good windows inside a bad run
        if not flags[i] and flags[i - 1] and flags[i + 1]:
            flags[i] = True
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i >= 2:
                runs.append((wins[i][0], wins[j - 1][1]))
            i = j
        else:
            i += 1
    runs += tracker_runs(beats, glob.period)
    if not runs:
        return [glob]
    cuts = sorted({glob.start_ms, glob.end_ms, *[a for a, _ in runs], *[b for _, b in runs]})
    pieces = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 4 * glob.period]
    segs: List[Segment] = []
    for a, b in pieces:
        m = (o >= a) & (o < b)
        r_glob = snap_rate(o[m], w[m], glob.period, glob.phase) if m.any() else 0.0
        segs.append(Segment(a, b, glob.period, glob.phase, r_glob))
        if w[m].sum() < 2.0:
            continue
        bm = beats[(beats >= a) & (beats < b)]
        center = dominant_period(bm) if len(bm) >= 6 else glob.period
        while center < glob.period / 1.45:     # keep the global metrical level
            center *= 2
        while center > glob.period * 1.45:
            center /= 2
        p, ph, r = _fit_region(o[m], w[m], center, 0.06)
        a2, b2 = _refine_bounds(o, w, a, b, glob, p, ph)
        m2 = (o >= a2) & (o < b2)
        if w[m2].sum() >= 2.0 and (a2, b2) != (a, b):
            p, ph, r = _fit_region(o[m2], w[m2], p, 0.01)
            r_glob = snap_rate(o[m2], w[m2], glob.period, glob.phase)
        ratio = glob.period / p
        if any(abs(ratio / q - 1) < 0.012 for q in (0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2.0)):
            continue                            # same music read at another metrical level
        dphase = (ph - glob.phase) / (glob.period / 4)
        differs = (abs(p - glob.period) / glob.period > 0.004 or
                   abs(dphase - round(dphase)) * glob.period / 4 > 8.0)
        if differs and r >= r_glob + 0.1:
            segs[-1] = Segment(a2, b2, p, ph, r)
    segs = _fill_gaps(segs, glob)
    return _merge(o, w, segs)


def _refine_bounds(o: np.ndarray, w: np.ndarray, a: float, b: float, glob: Segment,
                   p: float, ph: float, reach_beats: int = 8) -> Tuple[float, float]:
    """Move a candidate region's edges (beat steps) to where the onsets switch grids."""
    on_loc = snap_errors(o, p, ph) <= SNAP_TOL_MS
    on_glob = snap_errors(o, glob.period, glob.phase) <= SNAP_TOL_MS
    gain = w * (on_loc.astype(float) - on_glob.astype(float))     # >0: onset prefers the local grid

    def best_edge(edge: float, sign: int) -> float:
        best, best_t = -1e9, edge
        for k in range(-reach_beats, reach_beats + 1):
            t = edge + k * p
            inside = (o >= t) & (o < edge + reach_beats * p) if sign > 0 else (o < t) & (o >= edge - reach_beats * p)
            sc = gain[inside].sum()
            if sc > best:
                best, best_t = sc, t
        return best_t

    a2, b2 = best_edge(a, +1), best_edge(b, -1)
    return (a2, b2) if b2 - a2 > 4 * p else (a, b)


def _fill_gaps(segs: List[Segment], glob: Segment) -> List[Segment]:
    """Keep the pieces contiguous after edge refinement (gaps go back to the global grid)."""
    out: List[Segment] = []
    for s in sorted(segs, key=lambda s: s.start_ms):
        if out and s.start_ms < out[-1].end_ms:
            if s.period == glob.period and s.phase == glob.phase:
                s = Segment(out[-1].end_ms, s.end_ms, s.period, s.phase, s.snap_rate)
            else:
                prev = out[-1]
                out[-1] = Segment(prev.start_ms, s.start_ms, prev.period, prev.phase, prev.snap_rate)
        elif out and s.start_ms > out[-1].end_ms:
            if out[-1].period == glob.period and out[-1].phase == glob.phase:
                out[-1] = Segment(out[-1].start_ms, s.start_ms, glob.period, glob.phase, out[-1].snap_rate)
            else:
                s = Segment(out[-1].end_ms, s.end_ms, s.period, s.phase, s.snap_rate) if s.period == glob.period \
                    else s
                if s.start_ms > out[-1].end_ms:
                    out.append(Segment(out[-1].end_ms, s.start_ms, glob.period, glob.phase))
        if s.end_ms > s.start_ms:
            out.append(s)
    return out


def _same_grid(a: Segment, b: Segment) -> bool:
    if abs(a.period - b.period) > 0.002 * a.period:
        return False
    d = (b.phase - a.phase) / (a.period / 4)
    return abs(d - round(d)) * a.period / 4 < 3.0


def _merge(o: np.ndarray, w: np.ndarray, segs: List[Segment]) -> List[Segment]:
    out: List[Segment] = []
    for s in segs:
        if out and _same_grid(out[-1], s) and abs(out[-1].period - s.period) < 1e-9 and abs(out[-1].phase - s.phase) < 1e-9:
            out[-1] = Segment(out[-1].start_ms, s.end_ms, s.period, s.phase, s.snap_rate)
        elif out and _same_grid(out[-1], s):
            a = out[-1]
            m = (o >= a.start_ms) & (o < s.end_ms)
            p, ph, r = _fit_region(o[m], w[m], a.period, 0.003)
            out[-1] = Segment(a.start_ms, s.end_ms, p, ph, r)
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- rubato fallback

def per_beat_segments(o: np.ndarray, w: np.ndarray, beats: np.ndarray, ref: float) -> List[Segment]:
    """One red line per beat: tracker beats (at the dominant level) pulled onto nearby onsets."""
    b = [beats[0]]
    for t in beats[1:]:
        if t - b[-1] >= 0.7 * ref:
            b.append(t)
    b = np.array(b)
    locked = []
    for t in b:
        m = np.abs(o - t) <= 25.0
        if m.any():
            j = np.argmax(w[m] - np.abs(o[m] - t) / 100.0)
            locked.append(float(o[m][j]))
        else:
            locked.append(float(t))
    locked = np.array(locked)
    segs = []
    for i in range(len(locked) - 1):
        p = locked[i + 1] - locked[i]
        segs.append(Segment(locked[i], locked[i + 1], p, locked[i], per_beat=True))
    return segs


# --------------------------------------------------------------------------- main entry

def estimate_timing(env: OnsetEnvelopes, beat_logit: np.ndarray, down_logit: np.ndarray,
                    first_note_ms: Optional[float] = None, allow_changes: bool = True,
                    bpm: Optional[float] = None, offset_ms: Optional[float] = None) -> TimingResult:
    """Red lines in *audio* time (apply the osu! offset convention when writing the chart).

    bpm / offset_ms force one constant red line (offset = a downbeat in audio time)."""
    beats = pick_beats(beat_logit)
    if len(beats) < 8:
        raise ValueError("no steady beat found in this audio")
    act = 1.0 / (1.0 + np.exp(-np.asarray(beat_logit, float)))
    down = 1.0 / (1.0 + np.exp(-np.asarray(down_logit, float)))
    o, h = onset_peaks(env)
    w = np.minimum(h, 1.0) ** 1.5
    strong = h >= 0.25
    t0, t1 = float(min(beats[0], o[0] if len(o) else beats[0])), float(max(beats[-1], o[-1] if len(o) else beats[-1]))

    meter = choose_meter(beats, down)
    if bpm:
        period = 60000.0 / bpm
        if offset_ms is None:
            _, phase, _ = search_period(o, w, period, period)
            phase = refine_phase(o, w, period, phase)
            phase = beat_quarter(act, period, phase, t0, t1)
            db = downbeat_of(down, period, phase, t0, t1, meter)
        else:
            phase = db = float(offset_ms)
        seg = Segment(t0, t1, period, phase, snap_rate(o[strong], w[strong], period, phase), db)
        first = t0 if first_note_ms is None else min(first_note_ms, t0)
        return TimingResult(to_red_lines([seg], meter, first), [seg], seg.snap_rate, "forced")
    ref = fold_bpm_octave(dominant_period(beats))
    period, phase, _ = search_period(o, w, ref * 0.97, ref * 1.03)
    glob = _finish(Segment(t0, t1, period, phase), o, w, act, down, meter)
    glob.snap_rate = snap_rate(o[strong], w[strong], glob.period, glob.phase)
    segs, kind = [glob], "constant"

    if allow_changes:
        split = find_segments(o, w, beats, glob)
        if len(split) > 1:
            split = [_copy_global(s, glob) if _is_global(s, glob) else _finish(s, o, w, act, down, meter)
                     for s in split]
            split = _merge_exact(split)
            tot = _total_rate(o[strong], w[strong], split)
            if len(split) > 1 and tot >= glob.snap_rate + 0.02:
                segs, kind = split, "changes"
        if glob.snap_rate < 0.45 and max(glob.snap_rate, _total_rate(o[strong], w[strong], segs)) < 0.5:
            pb = per_beat_segments(o, w, beats, ref)
            if pb and _total_rate(o[strong], w[strong], pb) >= glob.snap_rate + 0.1:
                segs, kind = pb, "per_beat"
    for s in segs:
        if not s.per_beat:
            m = (o >= s.start_ms) & (o < s.end_ms) & strong
            s.snap_rate = snap_rate(o[m], w[m], s.period, s.phase)
    first = t0 if first_note_ms is None else min(first_note_ms, t0)
    reds = to_red_lines(segs, meter, first)
    return TimingResult(reds, segs, _total_rate(o[strong], w[strong], segs), kind)


def _finish(s: Segment, o: np.ndarray, w: np.ndarray, act: np.ndarray, down: np.ndarray, meter: int) -> Segment:
    """Snap the BPM, lock the phase to onsets, pick the beat quarter and the downbeat."""
    n = max(1, int((s.end_ms - s.start_ms) / s.period))
    period = 60000.0 / snap_bpm(s.period, n)
    m = (o >= s.start_ms) & (o < s.end_ms)
    phase = refine_phase(o[m], w[m], period, s.phase)
    phase = beat_quarter(act, period, phase, s.start_ms, s.end_ms)
    out = Segment(s.start_ms, s.end_ms, period, phase, s.snap_rate)
    out.downbeat = downbeat_of(down, period, phase, s.start_ms, s.end_ms, meter)
    return out


def _is_global(s: Segment, glob: Segment) -> bool:
    return abs(s.period - glob.period) < 1e-9 and abs(s.phase - glob.phase) < 1e-9


def _copy_global(s: Segment, glob: Segment) -> Segment:
    return Segment(s.start_ms, s.end_ms, glob.period, glob.phase, s.snap_rate, glob.downbeat)


def _merge_exact(segs: List[Segment]) -> List[Segment]:
    out: List[Segment] = []
    for s in segs:
        if out and abs(out[-1].period - s.period) < 1e-9 and abs(out[-1].phase - s.phase) < 1e-9:
            out[-1] = Segment(out[-1].start_ms, s.end_ms, s.period, s.phase, s.snap_rate, out[-1].downbeat)
        else:
            out.append(s)
    return out


def _total_rate(o: np.ndarray, w: np.ndarray, segs: Sequence[Segment]) -> float:
    num = den = 0.0
    for n, s in enumerate(segs):
        lo = -np.inf if n == 0 else s.start_ms
        hi = np.inf if n == len(segs) - 1 else segs[n + 1].start_ms
        m = (o >= lo) & (o < hi)
        num += snap_rate(o[m], w[m], s.period, s.phase) * w[m].sum()
        den += w[m].sum()
    return num / den if den else 0.0


def to_red_lines(segs: Sequence[Segment], meter: int, first_ms: float) -> List[RedLine]:
    reds: List[RedLine] = []
    for n, s in enumerate(segs):
        if s.per_beat:
            reds.append(RedLine(s.phase, s.period, meter))
            continue
        measure = s.period * meter
        anchor = s.downbeat if s.downbeat is not None else s.phase
        if n == 0:
            # first red line: the last downbeat at or before the first note
            t = anchor - math.ceil((anchor - (first_ms - 1.0)) / measure) * measure
        else:
            # change point: the new grid's downbeat closest to where the new tempo starts;
            # fall back to a plain beat when the downbeat would move the change too far
            t = anchor + round((s.start_ms - anchor) / measure) * measure
            if abs(t - s.start_ms) > s.period * 1.01:
                t = s.phase + round((s.start_ms - s.phase) / s.period) * s.period
            if reds and t <= reds[-1].time + reds[-1].beat_ms:
                t = s.phase + math.ceil((reds[-1].time + reds[-1].beat_ms - s.phase) / s.period) * s.period
        reds.append(RedLine(float(t), float(s.period), meter))
    reds.sort(key=lambda r: r.time)
    clean: List[RedLine] = []
    for r in reds:
        if clean and r.time <= clean[-1].time + 1:
            continue
        clean.append(r)
    return clean
