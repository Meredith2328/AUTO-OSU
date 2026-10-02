"""Render an osu!mania 4K .osu file as an editor-style PNG (time runs upward, strips left to right).

Notes are coloured by snap like the osu! editor (1/1 white, 1/2 red, 1/4 blue, 1/8 yellow,
1/3 purple, 1/6 pink, off-grid grey); measure lines are bright, beat lines dim.

    python scripts/mania4k_preview.py chart.osu -o chart.png [--start 30 --seconds 40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw                                      # noqa: E402

from autoosu.mania4k.chart import load_osu, red_line_at, snap_of      # noqa: E402

SNAP_COLOURS = {1: (235, 235, 235), 2: (230, 70, 70), 4: (70, 130, 240), 8: (240, 210, 60),
                3: (170, 90, 230), 6: (240, 120, 200), 12: (130, 130, 130), 16: (130, 130, 130), None: (110, 110, 110)}


def render(path: Path, out: Path, start_s: float = 0.0, seconds: float = 0.0, strip_s: float = 4.0,
           px_per_s: float = 180.0, lane_w: int = 22) -> None:
    c = load_osu(path)
    if c is None:
        raise SystemExit(f"{path}: not a 4K mania chart")
    t0 = start_s * 1000.0 if start_s else max(0.0, c.notes[0].time - 500)
    t1 = t0 + seconds * 1000.0 if seconds else max(max(n.end, n.time) for n in c.notes) + 500
    n_strips = int((t1 - t0) / (strip_s * 1000.0)) + 1
    strip_w, gap = 4 * lane_w + 8, 10
    H = int(strip_s * px_per_s) + 30
    img = Image.new("RGB", (n_strips * (strip_w + gap) + gap, H), (18, 20, 26))
    d = ImageDraw.Draw(img)

    def xy(t: float):
        k = int((t - t0) // (strip_s * 1000.0))
        y = H - 10 - (t - t0 - k * strip_s * 1000.0) / 1000.0 * px_per_s
        return k, gap + k * (strip_w + gap) + 4, y

    for k in range(n_strips):
        x = gap + k * (strip_w + gap)
        d.rectangle([x, 10, x + strip_w, H - 10], fill=(28, 31, 40))
        d.text((x + 2, H - 9), f"{(t0 / 1000 + k * strip_s):.0f}s", fill=(150, 150, 150))
    # beat / measure lines
    for i, r in enumerate(c.red_lines):
        end = c.red_lines[i + 1].time if i + 1 < len(c.red_lines) else t1
        b = 0
        while r.time + b * r.beat_ms < end:
            t = r.time + b * r.beat_ms
            if t0 <= t <= t1:
                k, x, y = xy(t)
                col = (120, 120, 140) if b % r.meter == 0 else (55, 58, 70)
                d.line([x - 2, y, x + 4 * lane_w + 2, y], fill=col)
            b += 1
    for n in c.notes:
        if not (t0 <= n.time <= t1):
            continue
        col = SNAP_COLOURS.get(snap_of(c.red_lines, n.time, (1, 2, 4, 8, 3, 6, 12, 16), 2.0), (110, 110, 110))
        k, x, y = xy(n.time)
        lx = x + n.lane * lane_w
        if n.is_hold:
            t = n.time
            while t < n.end:                    # LN bodies may cross strips
                seg_end = min(n.end, t0 + (int((t - t0) // (strip_s * 1000.0)) + 1) * strip_s * 1000.0 - 1)
                _, xs, ys = xy(t)
                _, _, ye = xy(seg_end)
                d.rectangle([xs + n.lane * lane_w + 5, ye, xs + n.lane * lane_w + lane_w - 5, ys],
                            fill=tuple(v // 2 for v in col))
                t = seg_end + 1
        d.rectangle([lx + 1, y - 4, lx + lane_w - 1, y + 1], fill=col)
    img.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("osu")
    ap.add_argument("-o", "--out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--strip", type=float, default=4.0)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.osu).with_suffix(".png")
    render(Path(args.osu), out, args.start, args.seconds, args.strip)
    print(out)


if __name__ == "__main__":
    main()
