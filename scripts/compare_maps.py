"""Compare several generated beatmaps of the same song side by side.

    .venv\Scripts\python scripts/compare_maps.py song.mp3 out/baselines/*.osz --window 90 --out out/compare.png

For every .osu found (inside .osz or given directly) it prints object counts, density, how many
notes sit on our beat grid and on a drum hit, and draws: rhythm rows for a time window, plus a
playfield panel per map for the first measures of that window.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.audio import analyze, load_audio  # noqa: E402
from autoosu.difficulty import PRESETS  # noqa: E402
from autoosu.rhythm import effective_grid, tick_features  # noqa: E402
from autoosu.timing import estimate_timing  # noqa: E402


@dataclass
class Obj:
    x: int
    y: int
    time: int
    kind: str          # circle | slider | spinner
    end_time: int
    new_combo: bool
    curve: str = ""
    points: List[Tuple[int, int]] = None
    length: float = 0.0


@dataclass
class ParsedMap:
    name: str
    objs: List[Obj]
    cs: float
    sm: float
    timing: List[Tuple[int, float, bool]]   # (time, beat_length or sv percent, uninherited)


def parse_osu(text: str, name: str) -> ParsedMap:
    def sect(title: str) -> List[str]:
        m = re.search(rf"^\[{title}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        return [l.strip() for l in m.group(1).splitlines() if l.strip() and not l.startswith("//")] if m else []

    diff = dict(l.split(":", 1) for l in sect("Difficulty") if ":" in l)
    cs = float(diff.get("CircleSize", 4)); sm = float(diff.get("SliderMultiplier", 1.4))
    timing = []
    for l in sect("TimingPoints"):
        p = l.split(",")
        timing.append((int(float(p[0])), float(p[1]), p[6] == "1" if len(p) > 6 else True))
    objs = []
    for l in sect("HitObjects"):
        p = l.split(",")
        x, y, t, typ = int(float(p[0])), int(float(p[1])), int(float(p[2])), int(p[3])
        nc = bool(typ & 4)
        if typ & 8:
            objs.append(Obj(x, y, t, "spinner", int(float(p[5])), nc))
        elif typ & 2:
            curve = p[5].split("|")
            pts = [tuple(int(float(v)) for v in q.split(":")) for q in curve[1:]]
            slides, length = int(p[6]), float(p[7])
            objs.append(Obj(x, y, t, "slider", t, nc, curve[0], pts, length * slides))
        else:
            objs.append(Obj(x, y, t, "circle", t, nc))
    # slider end times from timing points
    red, sv = None, 1.0
    tps = sorted(timing, key=lambda q: (q[0], not q[2]))
    for o in objs:
        if o.kind != "slider":
            continue
        bl, mult = 600.0, 1.0
        for t, v, uninh in tps:
            if t > o.time:
                break
            if uninh:
                bl, mult = v, 1.0
            else:
                mult = -100.0 / v if v < 0 else 1.0
        o.end_time = int(o.time + o.length / (sm * 100 * mult) * bl)
    return ParsedMap(name, objs, cs, sm, timing)


def load_maps(paths: List[str]) -> List[ParsedMap]:
    maps = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".osz":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith(".osu"):
                        label = re.search(r"\[(.*)\]\.osu$", n)
                        maps.append(parse_osu(z.read(n).decode("utf-8", "replace"),
                                              f"{p.stem[:24]} [{label.group(1) if label else n}]"))
        else:
            maps.append(parse_osu(p.read_text(encoding="utf-8", errors="replace"), p.stem[:40]))
    return maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--window", type=float, default=90.0, help="window start (s)")
    ap.add_argument("--length", type=float, default=12.0, help="window length (s)")
    ap.add_argument("--out", default="out/compare.png")
    ap.add_argument("--shift", type=float, default=26.0,
                    help="ms that osu! maps sit before the audio transient (ranked convention); added before grid checks")
    args = ap.parse_args()

    y, sr = load_audio(args.audio)
    an = analyze(y, sr)
    tm = estimate_timing(an)
    bl = tm.beat_length
    grid = effective_grid(an, tm, PRESETS["Insane"])
    ticks = {t.beat: t for t in tick_features(an, tm, grid)}
    maps = load_maps(args.maps)

    print(f"song grid: BPM {tm.bpm:g} offset {tm.offset_ms:.0f} ms (transients); maps are expected {args.shift:.0f} ms earlier\n")
    print(f"{'map':44s} {'objs':>5s} {'circ':>5s} {'sld':>5s} {'spin':>4s} {'obj/s':>6s} {'on-grid':>8s} {'on-drum':>8s} {'span':>12s}")
    for m in maps:
        objs = [o for o in m.objs if o.kind != "spinner"]
        if not objs:
            continue
        on_grid = on_drum = 0
        for o in objs:
            b = tm.beat_at(o.time + args.shift)
            q = round(b * 4) / 4
            if abs(b - q) * bl <= 12:
                on_grid += 1
                tk = ticks.get(round(q, 6))
                if tk is not None and tk.drum >= 0.4:
                    on_drum += 1
        span = (m.objs[-1].end_time - m.objs[0].time) / 1000
        c = sum(o.kind == "circle" for o in m.objs); s = sum(o.kind == "slider" for o in m.objs)
        sp = sum(o.kind == "spinner" for o in m.objs)
        print(f"{m.name:44s} {len(m.objs):5d} {c:5d} {s:5d} {sp:4d} {len(m.objs) / span:6.2f} "
              f"{on_grid / len(objs):8.0%} {on_drum / len(objs):8.0%} {m.objs[0].time / 1000:5.1f}-{m.objs[-1].end_time / 1000:.1f}s")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle

    t0, t1 = args.window, args.window + args.length
    n = len(maps)
    cols = min(n, 4)
    rows_pf = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(18, 4 + 3.6 * rows_pf))
    gs = fig.add_gridspec(1 + rows_pf, cols, height_ratios=[1.6] + [3.2] * rows_pf)

    ax = fig.add_subplot(gs[0, :])
    m_ = (an.times >= t0) & (an.times <= t1)
    ax.plot(an.times[m_], an.onset_env_perc[m_] * 0.9, color="tab:red", lw=0.6, alpha=0.6)
    beat = np.ceil(tm.beat_at(t0 * 1000))
    while tm.ms_at(beat) / 1000 <= t1:
        ax.axvline((tm.ms_at(beat) - args.shift) / 1000, color="k", alpha=0.35 if beat % 4 == 0 else 0.1, lw=1.0 if beat % 4 == 0 else 0.5)
        beat += 1
    for i, m in enumerate(maps):
        yy = n - i
        for o in m.objs:
            if o.time / 1000 > t1 or o.end_time / 1000 < t0:
                continue
            if o.kind == "circle":
                ax.scatter(o.time / 1000, yy, s=28, color="tab:red", zorder=3)
            elif o.kind == "slider":
                ax.plot([o.time / 1000, o.end_time / 1000], [yy, yy], color="tab:green", lw=5, solid_capstyle="round")
            else:
                ax.plot([o.time / 1000, o.end_time / 1000], [yy, yy], color="tab:purple", lw=8, alpha=0.5)
            if o.new_combo:
                ax.scatter(o.time / 1000, yy + 0.3, marker="v", s=14, color="k")
    ax.set_yticks([n - i for i in range(n)])
    ax.set_yticklabels([m.name for m in maps], fontsize=8)
    ax.set_ylim(0.3, n + 0.8)
    ax.set_xlim(t0, t1)
    ax.set_title(f"rhythm {t0:.0f}-{t1:.0f}s   (grid: BPM {tm.bpm:g}, dark = downbeat; red = circle, green = slider)")

    pf_t1 = t0 * 1000 + 4 * 4 * bl
    for i, m in enumerate(maps):
        axp = fig.add_subplot(gs[1 + i // cols, i % cols])
        r = 54.4 - 4.48 * m.cs
        objs = [o for o in m.objs if t0 * 1000 <= o.time <= pf_t1]
        cmap = plt.get_cmap("tab10")
        combo, prev = -1, None
        for o in objs:
            if o.new_combo:
                combo += 1
            col = cmap(combo % 10)
            if o.kind == "spinner":
                axp.add_patch(MplCircle((256, 192), 120, fill=False, ls="--", color="tab:purple"))
                prev = None
                continue
            if prev is not None:
                axp.plot([prev[0], o.x], [prev[1], o.y], color="k", alpha=0.25, lw=0.8)
            if o.kind == "slider" and o.points:
                pts = np.array([(o.x, o.y)] + list(o.points), float)
                axp.plot(pts[:, 0], pts[:, 1], color=col, lw=r * 0.7, alpha=0.3, solid_capstyle="round")
                prev = tuple(o.points[-1])
            else:
                prev = (o.x, o.y)
            axp.add_patch(MplCircle((o.x, o.y), r, color=col, alpha=0.8))
            axp.text(o.x, o.y, f"{(o.time - t0 * 1000) / bl:.3g}", ha="center", va="center", fontsize=5, color="w")
        axp.set_xlim(0, 512); axp.set_ylim(384, 0); axp.set_aspect("equal")
        axp.set_title(f"{m.name}\nfirst 4 measures of window ({len(objs)} objects)", fontsize=8)
        axp.set_xticks([]); axp.set_yticks([])

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
