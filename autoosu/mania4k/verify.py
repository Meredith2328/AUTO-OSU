"""Objective checks for generated 4K charts: timing against a reference, sync to audio, playability."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .chart import Chart, Note, RedLine, red_line_at, snap_of


def beat_position(reds: Sequence[RedLine], t: float) -> tuple[float, RedLine]:
    r = red_line_at(reds, t)
    return (t - r.time) / r.beat_ms, r


def grid_errors(ref_reds: Sequence[RedLine], ref_times: Sequence[float], our_reds: Sequence[RedLine],
                shift_ms: float = 0.0, divisors: Sequence[int] = (1, 2, 3, 4, 6, 8)) -> np.ndarray:
    """Signed distance (ms) of each reference note from our grid at the musically equivalent snap.

    ref_times are in the reference chart's time base; ``shift_ms`` converts our grid into it
    (our_time - shift_ms = reference time). A note the reference snaps to 1/d sits on our 1/d'
    grid, where d' absorbs a BPM octave difference (half or double time). Notes on unusual snaps
    in the reference are skipped (NaN).
    """
    out = np.full(len(ref_times), np.nan)
    shifted = [RedLine(r.time - shift_ms, r.beat_ms, r.meter) for r in our_reds]
    for i, t in enumerate(ref_times):
        d = snap_of(ref_reds, t, divisors)
        if d is None:
            continue
        rr = red_line_at(ref_reds, t)
        u, ro = beat_position(shifted, t)
        ratio = rr.beat_ms / ro.beat_ms                 # our beats per reference beat
        k = min((0.5, 1.0, 2.0, 1.5, 0.75, 4.0, 0.25), key=lambda c: abs(math.log(ratio / c)))
        # reference 1/d beat == our (k/d) beat; smallest grid containing it
        num, den = k.as_integer_ratio()
        dd = d * den
        g = math.gcd(num, dd)
        dprime = dd // g
        x = u * dprime
        out[i] = (x - round(x)) * ro.beat_ms / dprime
    return out


@dataclass
class TimingReport:
    n: int
    within_5ms: float
    within_10ms: float
    median_signed: float
    p95_abs: float
    bpm_ref: float
    bpm_ours: float
    octave: float
    consistent_5ms: float = 0.0     # within 5 ms after removing the song's constant offset

    @property
    def passed(self) -> bool:
        return self.consistent_5ms >= 0.98


def timing_report(ref: Chart, our_reds: Sequence[RedLine], shift_ms: float) -> TimingReport:
    from .chart import main_bpm

    times = ref.heads
    e = grid_errors(ref.red_lines, times, our_reds, shift_ms)
    e = e[np.isfinite(e)]
    a = np.abs(e)
    end = times[-1] if times else 0.0
    b_ref, b_our = main_bpm(ref.red_lines, end), main_bpm(our_reds, end + shift_ms)
    return TimingReport(len(e), float((a <= 5).mean()) if len(a) else 0.0,
                        float((a <= 10).mean()) if len(a) else 0.0,
                        float(np.median(e)) if len(e) else 0.0,
                        float(np.percentile(a, 95)) if len(a) else 0.0, b_ref, b_our, b_our / b_ref,
                        float((np.abs(e - np.median(e)) <= 5).mean()) if len(e) else 0.0)


# --------------------------------------------------------------------------- generated-chart checks

@dataclass
class ChartCheck:
    notes: int
    off_grid: int = 0               # heads or tails not on a 1/1..1/8 or 1/3, 1/6 grid point
    unsupported: int = 0            # heads without a distinct attack within +-8 ms
    overlaps: int = 0               # a note starting inside another note of the same lane
    fast_jacks: int = 0             # same-lane repeats faster than the difficulty allows
    big_chords: int = 0
    tight_rows: int = 0             # rows closer than the difficulty's minimum gap
    before_first_red: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.off_grid or self.unsupported or self.overlaps or self.fast_jacks or self.big_chords
                    or self.tight_rows or self.before_first_red)


def verify_chart(chart: Chart, env=None, rules=None, grid_tol_ms: float = 1.0) -> ChartCheck:
    """Re-check a generated chart (audio time) against its own red lines, the audio and the rules."""
    chk = ChartCheck(len(chart.notes))
    reds = chart.red_lines
    for n in chart.notes:
        for t in ([n.time, n.end] if n.is_hold else [n.time]):
            if snap_of(reds, t, (1, 2, 3, 4, 6, 8), grid_tol_ms) is None:
                chk.off_grid += 1
        if n.time < reds[0].time - 1:
            chk.before_first_red += 1
    if env is not None:
        from .onsets import attack_times, near_attack

        heads = np.array(sorted({n.time for n in chart.notes}))
        chk.unsupported = int((~near_attack(heads, attack_times(env))).sum())
    by_lane: Dict[int, List[Note]] = {l: [] for l in range(4)}
    for n in chart.notes:
        by_lane[n.lane].append(n)
    for l, ns in by_lane.items():
        ns.sort(key=lambda n: n.time)
        for a, b in zip(ns, ns[1:]):
            if b.time <= (a.end if a.is_hold else a.time):
                chk.overlaps += 1
            elif rules is not None and not a.is_hold and b.time - a.time < rules.min_jack_ms - 1.5:
                chk.fast_jacks += 1
    if rules is not None:
        rows: Dict[float, int] = {}
        for n in chart.notes:
            rows[n.time] = rows.get(n.time, 0) + 1
        chk.big_chords = sum(c > rules.max_chord for c in rows.values())
        ts = sorted(rows)
        chk.tight_rows = sum(b - a < rules.min_row_gap_ms - 1.5 for a, b in zip(ts, ts[1:]))
    for name in ("off_grid", "unsupported", "overlaps", "fast_jacks", "big_chords", "tight_rows", "before_first_red"):
        if getattr(chk, name):
            chk.problems.append(f"{name}={getattr(chk, name)}")
    return chk
