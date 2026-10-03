"""Chart structure: pattern types per window, section consistency, intensity.

Vocabulary (4K community terms; full definitions and sources in docs/mania4k_patterns.md):

* 切 / stream family (no jacks, chords as the backbone) - ``jumpstream``: streams with two-note
  chords (ljs / djs, also jumptrills and splits); ``handstream``: with three-note chords (lhs / dhs).
* 乱 / speed family (fast single notes, no jacks) - ``stream``: scattered single notes;
  ``roll``: stairs (1234 / 4321) and their variants.
* "trill" has two meanings. As a building block, 交互 (alternation, one- or two-handed, also
  scattered) is part of every family above and is only measured (``WindowStats.trill``). As a
  section type, ``trill`` (交互段 / 长交互) is on the same scale as stream or tech: alternation
  dominates the window and runs long (share >= 0.5 and a run of >= 8 rows).
* 叠 / jack family - ``jack``: single notes repeating a lane (incl. minijacks); ``chordjack``:
  chords sharing lanes with the previous row (小/中/大叠).
* 技 / ``mixed``: stream and jack interleaved in one window (tech).
* ``ln``: long-note sections.
* ``light``: sparse windows (rests, intros), where the type does not matter.

Chart archetypes (the chart's overall style): 切 (stream), 乱 (speed), 叠 (jack), LN,
混合 (hybrid: several families side by side).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

TYPES = ("light", "stream", "trill", "roll", "jumpstream", "handstream", "jack", "chordjack", "ln", "mixed")
FAMILY = {"light": "light", "stream": "乱", "roll": "乱", "trill": "交互", "jumpstream": "切", "handstream": "切",
          "jack": "叠", "chordjack": "叠", "ln": "LN", "mixed": "技"}


def rows_of(notes) -> List[Tuple[float, int, float]]:
    """(time, lane mask, longest LN length ms) per row, sorted."""
    rows: Dict[float, List] = {}
    for n in notes:
        t = round(float(n[0]), 1)
        r = rows.setdefault(t, [0, 0.0])
        r[0] |= 1 << int(n[1])
        if n[2] and n[2] > n[0]:
            r[1] = max(r[1], float(n[2]) - float(n[0]))
    return [(t, m, l) for t, (m, l) in sorted(rows.items())]


def _pc(m: int) -> int:
    return bin(m).count("1")


@dataclass
class WindowStats:
    rows: int
    nps: float
    chord: float          # notes per row
    jack: float           # share of consecutive row pairs sharing a lane
    trill: float          # 交互: share of rows equal to the row two back and disjoint from the previous
    roll: float           # share of single-note rows continuing a +-1 lane staircase
    ln: float             # share of rows starting a long note
    triple: float         # share of rows with >= 3 notes
    trill_run: int = 0    # longest unbroken alternation (ABAB...) in rows


def window_stats(rows: Sequence[Tuple[float, int, float]], span_ms: float) -> WindowStats:
    n = len(rows)
    if n == 0:
        return WindowStats(0, 0, 0, 0, 0, 0, 0, 0)
    masks = [m for _, m, _ in rows]
    pcs = [_pc(m) for m in masks]
    pairs = list(zip(masks, masks[1:]))
    jack = float(np.mean([(a & b) != 0 for a, b in pairs])) if pairs else 0.0
    tr = [masks[i] == masks[i - 2] and (masks[i] & masks[i - 1]) == 0 for i in range(2, n)]
    trill = float(np.mean(tr)) if tr else 0.0
    run = best = 0
    for x in tr:
        run = run + 1 if x else 0
        best = max(best, run)
    trill_run = best + 2 if best else 0
    ro = []
    for i in range(2, n):
        if pcs[i] == pcs[i - 1] == pcs[i - 2] == 1:
            a, b, c = (masks[i - 2].bit_length(), masks[i - 1].bit_length(), masks[i].bit_length())
            ro.append(abs(b - a) == 1 and (c - b) == (b - a))
    roll = float(np.mean(ro)) if ro else 0.0
    return WindowStats(n, n / max(span_ms / 1000.0, 1e-6) * float(np.mean(pcs)), float(np.mean(pcs)), jack,
                       trill, roll, float(np.mean([l > 0 for _, _, l in rows])), float(np.mean([p >= 3 for p in pcs])),
                       trill_run)


def is_trill_section(s: WindowStats) -> bool:
    """交互段 / 长交互: alternation dominates the window and runs unbroken for >= 8 rows."""
    return s.trill >= 0.5 and s.trill_run >= 8


def classify(s: WindowStats, light_nps: float) -> str:
    """Dominant pattern type of a window (thresholds fitted on ranked 4K charts, see
    docs/mania4k_patterns.md)."""
    if s.rows < 4 or s.nps < light_nps:
        return "light"
    if s.ln >= 0.35:
        return "ln"
    if s.chord >= 1.45:
        if s.jack >= 0.5:
            return "chordjack"
        if s.jack <= 0.3:
            if is_trill_section(s):
                return "trill"
            return "handstream" if s.triple >= 0.2 else "jumpstream"
        return "mixed"
    if s.jack >= 0.4:
        return "jack"
    if s.jack <= 0.2:
        if is_trill_section(s):
            return "trill"
        if s.roll >= 0.4:
            return "roll"
        return "jumpstream" if s.chord >= 1.2 else "stream"
    return "mixed"


def measure_windows(rows, reds, start_ms: float, end_ms: float, beats: int = 4) -> List[Tuple[float, float]]:
    """Windows of `beats` beats following the red lines."""
    out = []
    for i, r in enumerate(reds):
        seg_end = reds[i + 1].time if i + 1 < len(reds) else end_ms
        span = r.beat_ms * beats
        k = int(np.floor((max(start_ms, r.time) - r.time) / span)) if r.time < start_ms else 0
        t = r.time + k * span
        while t < min(seg_end, end_ms):
            if t + span > start_ms:
                out.append((t, min(t + span, seg_end)))
            t += span
    return out


def chart_profile(notes, reds, beats: int = 4) -> Dict:
    """Per-window types, stats and nps for a chart (notes as (time, lane, end))."""
    rows = rows_of(notes)
    if not rows:
        return {"windows": [], "types": [], "nps": np.zeros(0)}
    wins = measure_windows(rows, reds, rows[0][0] - 1, rows[-1][0] + 1, beats)
    times = np.array([t for t, _, _ in rows])
    stats = []
    for a, b in wins:
        i, j = np.searchsorted(times, a), np.searchsorted(times, b)
        stats.append(window_stats(rows[i:j], b - a))
    nps = np.array([s.nps for s in stats])
    active = nps[nps > 0]
    light = 0.35 * float(np.median(active)) if len(active) else 0.0
    types = [classify(s, light) for s in stats]
    return {"windows": wins, "stats": stats, "types": types, "nps": nps}


def consistency(types: Sequence[str], skip_light: bool = True) -> float:
    """Share of consecutive (non-light) windows with the same type."""
    t = [x for x in types if not (skip_light and x == "light")]
    if len(t) < 2:
        return 1.0
    return float(np.mean([a == b for a, b in zip(t, t[1:])]))


def type_shares(types: Sequence[str]) -> Dict[str, float]:
    t = [x for x in types if x != "light"]
    return {k: (t.count(k) / len(t) if t else 0.0) for k in TYPES if k != "light"}


# --------------------------------------------------------------------------- music sections

def measure_features(mel: np.ndarray, mel_fps: float, wins: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Per-window timbre summary: mean and std of each mel band (z-scored over the song)."""
    out = []
    for a, b in wins:
        fa = int(max(a, 0.0) / 1000 * mel_fps)
        fb = max(fa + 1, int(b / 1000 * mel_fps))
        seg = mel[fa:min(fb, len(mel))].astype(np.float32)
        if len(seg) == 0:
            seg = mel[-1:].astype(np.float32)
        out.append(np.concatenate([seg.mean(0), seg.std(0)]))
    f = np.array(out)
    return (f - f.mean(0)) / (f.std(0) + 1e-6)


def novelty(feats: np.ndarray, half: int = 4) -> np.ndarray:
    """Checkerboard novelty on the cosine self-similarity of window features (Foote 2000)."""
    n = len(feats)
    x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    S = x @ x.T
    k = np.ones((2 * half, 2 * half))
    k[:half, half:] = k[half:, :half] = -1
    g = np.exp(-0.5 * ((np.arange(2 * half) - half + 0.5) / half) ** 2)
    k *= np.outer(g, g)
    nov = np.zeros(n)
    Sp = np.pad(S, half, mode="edge")
    for i in range(n):
        nov[i] = (Sp[i:i + 2 * half, i:i + 2 * half] * k).sum()
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def section_bounds(nov: np.ndarray, min_len: int = 4, phrase: int = 4) -> List[int]:
    """Window indices starting a section: novelty peaks, preferring phrase starts (multiples of
    4 measures), at least min_len apart."""
    n = len(nov)
    score = nov.copy()
    for i in range(n):
        if i % phrase == 0:
            score[i] *= 1.25
    cand = [i for i in range(1, n - 1) if score[i] >= score[i - 1] and score[i] >= score[i + 1]
            and nov[i] >= max(0.2, float(np.percentile(nov, 70)))]
    cand.sort(key=lambda i: -score[i])
    chosen: List[int] = []
    for i in cand:
        if all(abs(i - j) >= min_len for j in chosen) and i >= min_len // 2 and n - i >= min_len // 2:
            chosen.append(i)
    return [0] + sorted(chosen)


ARCHETYPES = ("切", "乱", "叠", "LN", "混合")


def chart_archetype(types: Sequence[str]) -> str:
    """Chart-level style from its window types: LN-, 叠 (jack)-, 切 (stream)- or 乱 (speed)-dominant,
    else 混合 (hybrid)."""
    act = [t for t in types if t != "light"]
    if not act:
        return "乱"
    fam = {k: sum(FAMILY[t] == k for t in act) / len(act) for k in ("切", "乱", "交互", "叠", "LN", "技")}
    if fam["LN"] >= 0.5:
        return "LN"
    if fam["叠"] >= 0.3:
        return "叠"
    if fam["切"] + fam["乱"] + fam["交互"] >= 0.6:
        return "切" if fam["切"] >= fam["乱"] else "乱"
    return "混合"
