"""Original pure functions from AUTO-OSU 0707c310 (synthetic tests only).
timing.py blob: 05a25af04a261ef140a249b0ed9b4cdce4a458b0
generate.py blob: 936d3008955a3250d23dace90b8361c4470e0b64
Dependencies are supplied by mania4k_source.py without importing models/audio.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

# These dependencies are supplied by mania4k_source.namespace during AST execution.
# Static declarations keep the unchanged source fixture visible to lint without
# importing models, audio or their runtime dependencies in synthetic tests.
if TYPE_CHECKING:
    import bisect
    import numpy as np
    from autoosu.mania4k.chart import Chart
    from autoosu.mania4k.features import DIV_CLASSES
    from autoosu.mania4k.generate import (
        DIFFICULTIES, ChartReport, Row, Rules, SongAnalysis, assign_lanes,
        chart_to_osu_text, consistent_snaps, load_models, note_probabilities,
        rules_for, star_rating,
    )
    from autoosu.mania4k.model import LN_BINS, NoteNet
    from autoosu.mania4k.patterns import PatternNet
    from autoosu.mania4k.timing import Segment

def snap_bpm(period: float, n_beats: int, tol_ms: float = 2.0) -> float:
    """Simplest BPM whose accumulated drift over half the segment stays below tol_ms."""
    bpm = 60000.0 / period
    for q in (1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01):
        cand = round(bpm / q) * q
        budget = 5.0 * tol_ms if q == 1.0 else tol_ms      # DAW-made songs have integer BPMs
        if abs(60000.0 / cand - period) * max(1, n_beats) / 2.0 <= budget:
            return cand
    return round(bpm, 3)

def _near_global(s: Segment, glob: Segment, tol_ms: float = 10.0) -> bool:
    if abs(s.period - glob.period) > 1e-6:
        return False
    d = (s.phase - glob.phase) / (glob.period / 4)
    return abs(d - round(d)) * glob.period / 4 <= tol_ms

def select_rows(an: SongAnalysis, probs: Dict[str, np.ndarray], rules: Rules, theta: float,
                chord_boost: float = 1.0) -> List[Row]:
    g = an.grid
    p_note = 1.0 - probs["count"][:, 0]
    divs = np.array([DIV_CLASSES[d] for d in g.div])
    # 1/8 only where it is a playable rhythm rather than a flam (>= 55 ms apart)
    allowed_div = np.isin(divs, rules.divisors) & ((divs != 8) | (g.beat_ms / 8.0 >= 55.0))
    cand = np.where((p_note >= theta) & an.support & allowed_div)[0]
    order = cand[np.argsort(-p_note[cand], kind="stable")]
    taken: List[float] = []
    chosen: List[int] = []
    for i in order:
        t = g.times[i]
        j = bisect.bisect_left(taken, t)
        if j > 0 and t - taken[j - 1] < rules.min_row_gap_ms:
            continue
        if j < len(taken) and taken[j] - t < rules.min_row_gap_ms:
            continue
        taken.insert(j, t)
        chosen.append(int(i))
    chosen = consistent_snaps(g, sorted(chosen), p_note)
    if not chosen:
        return []
    pc = probs["count"][chosen]
    cond = pc[:, 1:] / np.maximum(pc[:, 1:].sum(1, keepdims=True), 1e-9)     # p(k | note)
    ge2, ge3, ge4 = cond[:, 1:].sum(1), cond[:, 2:].sum(1), cond[:, 3]
    # keep the model's expected share of chords, giving them to the most chord-like rows
    k = np.ones(len(chosen), int)
    for level, share_p in ((2, ge2), (3, ge3), (4, ge4)):
        if level > rules.max_chord:
            break
        n = min(int(round(share_p.sum() * chord_boost)), int(round(rules.max_chord_share * len(chosen))))
        if n <= 0:
            continue
        top = np.argsort(-share_p, kind="stable")[:n]
        k[top[k[top] >= level - 1]] = level
    # long notes on the rows the model finds most sustained, at most the difficulty's share
    p_ln = probs["ln"][chosen]
    confident = np.where(p_ln >= 0.25)[0]
    n_ln = min(len(confident), int(rules.max_ln_share * len(chosen)))
    ln_rows = set(confident[np.argsort(-p_ln[confident], kind="stable")[:n_ln]].tolist())
    rows = []
    for r, i in enumerate(chosen):
        beats = LN_BINS[int(np.argmax(probs["lnlen"][i]))]
        rows.append(Row(i, float(g.times[i]), int(k[r]), r in ln_rows, beats))
    return rows

def generate_chart(an: SongAnalysis, name: str, target_stars: Optional[float] = None, seed: int = 0,
                   models: Optional[Tuple[NoteNet, PatternNet]] = None, tolerance: float = 0.08
                   ) -> Tuple[Chart, ChartReport]:
    target = float(target_stars if target_stars is not None else DIFFICULTIES[name])
    rules = rules_for(target)
    nnet, pnet = models or load_models()
    probs = note_probabilities(an, nnet, target)

    def build(theta: float, boost: float) -> Tuple[Chart, float]:
        rows = select_rows(an, probs, rules, theta, boost)
        notes = assign_lanes(an, rows, rules, pnet, target, np.random.default_rng(seed))
        chart = Chart(notes, list(an.timing.red_lines), rules.od, rules.hp, name)
        return chart, (star_rating(chart_to_osu_text(chart)) if notes else 0.0)

    best: Optional[Tuple[float, Chart, float]] = None
    for boost in (1.0, 1.3, 1.6):           # more chords only when the song is too sparse otherwise
        lo, hi = 0.02, 0.98                  # higher theta -> fewer notes -> lower stars
        theta = 0.5
        for _ in range(12):
            chart, sr = build(theta, boost)
            if best is None or abs(sr - target) < abs(best[2] - target):
                best = (theta, chart, sr)
            if abs(sr - target) <= tolerance:
                break
            if sr > target:
                lo = theta
            else:
                hi = theta
            theta = 0.5 * (lo + hi)
        if abs(best[2] - target) <= tolerance or best[2] > target:
            break
    theta, chart, sr = best
    holds = sum(n.is_hold for n in chart.notes)
    return chart, ChartReport(name, target, sr, theta, len({n.time for n in chart.notes}), len(chart.notes), holds)
