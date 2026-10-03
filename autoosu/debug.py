"""Debug visualisation: loudness/kiai, onset envelope + beat grid, chosen events, and the playfield."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .beatmap import Hold, PLAYFIELD_H, PLAYFIELD_W, Slider, Spinner
from .placement import circle_radius


def _slider_path(head, sl: Slider, n: int = 40) -> np.ndarray:
    pts = [np.array(head, float)] + [np.array(p, float) for p in sl.points]
    if sl.curve_type == "B" and len(pts) == 3:
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]
    if sl.curve_type == "P" and len(pts) == 3:
        (ax, ay), (bx, by), (cx, cy) = pts
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) > 1e-6:
            a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
            ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
            uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
            r = np.hypot(ax - ux, ay - uy)
            a0, a1, a2_ = (np.arctan2(p[1] - uy, p[0] - ux) for p in pts)
            ccw = lambda a, b: (b - a) % (2 * np.pi)
            if ccw(a0, a1) < ccw(a0, a2_):
                angles = a0 + np.linspace(0, ccw(a0, a2_), n)
            else:
                angles = a0 - np.linspace(0, ccw(a2_, a0), n)
            return np.stack([ux + r * np.cos(angles), uy + r * np.sin(angles)], axis=1)
    return np.array(pts)


def plot_debug(res, out_png: Path, window_s: float = 24.0, playfield_measures: int = 4) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle

    an, timing = res.analysis, res.timing
    first = min((d.events[0].time for d in res.diffs if d.events), default=int(timing.offset_ms))
    t0 = max(0.0, first / 1000.0 - 1.0)
    t1 = min(an.duration, t0 + window_s)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[3, 2])
    ax_loud = fig.add_subplot(gs[0, :])
    ax_env = fig.add_subplot(gs[1, :])
    ax_ev = fig.add_subplot(gs[2, 0])
    ax_pf = fig.add_subplot(gs[2, 1])

    # 1. loudness + kiai over the whole song
    ax_loud.plot(an.times, an.rms, color="tab:gray", lw=0.8, label="loudness")
    tps = res.diffs[0].beatmap.timing_points if res.diffs else []
    starts = [t.time for t in tps if t.kiai]
    ends = [t.time for t in tps if not t.uninherited and not t.kiai]
    for a, b in zip(starts, ends):
        ax_loud.axvspan(a / 1000, b / 1000, color="tab:orange", alpha=0.2)
    ax_loud.axvspan(t0, t1, color="tab:blue", alpha=0.08)
    ax_loud.set_xlim(0, an.duration)
    ax_loud.set_title(f"{Path(res.osz).name}   BPM {timing.bpm:g}   offset {timing.offset_ms:.0f} ms   "
                      f"(orange = kiai, blue = zoom window)")
    ax_loud.legend(loc="upper right")

    # 2. onset envelopes + beat grid in the zoom window
    m = (an.times >= t0) & (an.times <= t1)
    ax_env.plot(an.times[m], an.onset_env_perc[m], color="tab:red", lw=0.8, label="percussive onset strength")
    ax_env.plot(an.times[m], an.onset_env_harm[m], color="tab:blue", lw=0.8, label="harmonic onset strength")
    beat = np.ceil(timing.beat_at(t0 * 1000))
    while timing.ms_at(beat) / 1000 <= t1:
        down = beat % 4 == 0
        ax_env.axvline(timing.ms_at(beat) / 1000, color="k", alpha=0.35 if down else 0.12, lw=1.2 if down else 0.6)
        beat += 1
    on = [o for o in an.onsets if t0 <= o.time <= t1]
    ax_env.scatter([o.time for o in on], [o.strength for o in on], s=14, color="k", zorder=3, label="onsets")
    ax_env.set_xlim(t0, t1)
    ax_env.set_ylim(0, 1.05)
    ax_env.legend(loc="upper right")

    # 3. chosen events per difficulty
    for row, d in enumerate(res.diffs):
        y = len(res.diffs) - row
        for ev in d.events:
            if ev.time / 1000 > t1 or (ev.end_time or ev.time) / 1000 < t0:
                continue
            if ev.kind == "circle":
                ax_ev.scatter(ev.time / 1000, y, s=40, color="tab:red", zorder=3)
            elif ev.kind in ("slider", "hold"):
                ax_ev.plot([ev.time / 1000, ev.end_time / 1000], [y, y], color="tab:green", lw=6,
                           solid_capstyle="round")
            else:
                ax_ev.plot([ev.time / 1000, ev.end_time / 1000], [y, y], color="tab:purple", lw=10, alpha=0.6)
            if ev.new_combo:
                ax_ev.scatter(ev.time / 1000, y + 0.25, marker="v", s=20, color="k")
    ax_ev.set_yticks([len(res.diffs) - i for i in range(len(res.diffs))])
    ax_ev.set_yticklabels([d.preset.name for d in res.diffs])
    ax_ev.set_ylim(0.3, len(res.diffs) + 0.7)
    ax_ev.set_xlim(t0, t1)
    ax_ev.set_xlabel("time (s)   red = circle, green = slider, purple = spinner, black = new combo")
    ax_ev.grid(axis="x", alpha=0.2)

    # 4. playfield for the hardest difficulty, first few measures of the window
    if res.diffs:
        d = res.diffs[-1]
        r = circle_radius(d.preset.cs)
        pf_t0 = first
        pf_t1 = pf_t0 + playfield_measures * 4 * timing.beat_length
        objs = [o for o in d.beatmap.hit_objects if pf_t0 <= o.time <= pf_t1]
        cmap = plt.get_cmap("tab10")
        combo = -1
        prev = None
        for o in objs:
            if o.new_combo:
                combo += 1
            col = cmap(combo % 10)
            if isinstance(o, Spinner):
                ax_pf.add_patch(MplCircle((o.x, o.y), 120, fill=False, ls="--", color="tab:purple"))
                prev = None
                continue
            if prev is not None:
                ax_pf.plot([prev[0], o.x], [prev[1], o.y], color="k", alpha=0.25, lw=0.8)
            if isinstance(o, Hold):
                ax_pf.plot([o.x, o.x], [o.y, 0], color=col, lw=r * 0.7, alpha=0.35)
            elif isinstance(o, Slider):
                path = _slider_path((o.x, o.y), o)
                ax_pf.plot(path[:, 0], path[:, 1], color=col, lw=r * 0.9, alpha=0.35, solid_capstyle="round")
                ax_pf.add_patch(MplCircle((o.end_x, o.end_y), r * 0.5, fill=False, color=col, lw=1))
            ax_pf.add_patch(MplCircle((o.x, o.y), r, fill=True, color=col, alpha=0.8))
            ax_pf.text(o.x, o.y, f"{(o.time - pf_t0) / timing.beat_length:.2g}", ha="center", va="center",
                       fontsize=6, color="w")
            prev = o.end_pos
        ax_pf.set_xlim(0, PLAYFIELD_W)
        ax_pf.set_ylim(PLAYFIELD_H, 0)
        ax_pf.set_aspect("equal")
        ax_pf.set_title(f"{d.preset.name}: first {playfield_measures} measures (labels = beat offset)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
