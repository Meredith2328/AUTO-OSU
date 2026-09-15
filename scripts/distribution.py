"""Where do the objects go? Density over the song and rhythm-gap mix, per map.

    .venv\Scripts\python scripts/distribution.py song.mp3 "label=path.osz|.osu" ... --out out/distribution.png
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_maps import load_maps  # noqa: E402
from autoosu.audio import analyze, load_audio  # noqa: E402
from autoosu.timing import estimate_timing  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GAP_BINS = [("1/4", 0.375), ("1/2", 0.75), ("1/1", 1.5), ("2", 2.5), ("3+", np.inf)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("maps", nargs="+", help="label=path (osz or osu); every .osu inside an osz is loaded")
    ap.add_argument("--bin", type=float, default=10.0, help="seconds per density bin")
    ap.add_argument("--out", default="out/distribution.png")
    args = ap.parse_args()

    y, sr = load_audio(args.audio)
    an = analyze(y, sr)
    tm = estimate_timing(an)
    bl = tm.beat_length

    labelled = []
    for spec in args.maps:
        label, path = spec.split("=", 1)
        for m in load_maps([path]):
            labelled.append((label if len(load_maps([path])) == 1 else f"{label} {m.name.split('[')[-1].rstrip(']')}", m))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labelled)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 2.2, 2.2]})
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)

    # 1. loudness reference
    ax = axes[0]
    ax.fill_between(an.times, an.rms, color="#cfcfcb", lw=0)
    ax.set_xlim(0, an.duration); ax.set_ylim(0, 1.05)
    ax.set_ylabel("loudness", color=INK2)
    ax.set_title("Where the objects go: loudness, object density per 10 s, and rhythm-gap mix", color=INK, loc="left", fontsize=12)

    # 2. density over time
    ax = axes[1]
    edges = np.arange(0, an.duration + args.bin, args.bin)
    for i, (label, m) in enumerate(labelled):
        starts = np.array([o.time / 1000 for o in m.objs if o.kind != "spinner"])
        counts, _ = np.histogram(starts, bins=edges)
        xs = edges[:-1] + args.bin / 2
        ax.plot(xs, counts, color=SERIES[i], lw=2, label=label)
        ax.text(xs[-1] + 2, counts[-1], label, color=INK, fontsize=8, va="center")
    ax.set_xlim(0, an.duration + 25)
    ax.set_ylabel(f"objects per {args.bin:.0f} s", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=min(n, 4))

    # 3. gap mix (start-to-start gap between consecutive objects, in beats of the detected grid)
    ax = axes[2]
    width = 0.8 / n
    for i, (label, m) in enumerate(labelled):
        starts = [o.time for o in m.objs if o.kind != "spinner"]
        gaps = np.diff(starts) / bl
        c = Counter()
        for g in gaps:
            for name, hi in GAP_BINS:
                if g <= hi:
                    c[name] += 1
                    break
        total = max(1, sum(c.values()))
        vals = [100 * c[name] / total for name, _ in GAP_BINS]
        xs = np.arange(len(GAP_BINS)) + (i - (n - 1) / 2) * width
        bars = ax.bar(xs, vals, width=width * 0.92, color=SERIES[i], label=label)
        for b, v in zip(bars, vals):
            if v >= 8:
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(GAP_BINS)))
    ax.set_xticklabels([f"{name} beat" if name not in ("2", "3+") else f"{name} beats" for name, _ in GAP_BINS])
    ax.set_ylabel(f"share of gaps (%)  at BPM {tm.bpm:g}", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4))

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    print(f"figure: {out}")
    for label, m in labelled:
        starts = [o.time for o in m.objs if o.kind != "spinner"]
        gaps = np.diff(starts) / bl
        print(f"{label:28s} objects {len(starts):5d}  median gap {np.median(gaps):.2f} beats  "
              f"share 1/2-or-faster {100 * np.mean(gaps <= 0.75):.0f}%  share >= 2 beats {100 * np.mean(gaps > 1.5):.0f}%")


if __name__ == "__main__":
    main()
