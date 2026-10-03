"""Compare generated 4K charts with ranked charts of the same (held-out) songs.

For every ranked chart of a test-split song, a chart is generated at the same star rating:

* sync precision: share of generated note heads within 12 ms of a note head of *any* ranked
  difficulty of the song (all in audio time); recall against the ranked chart of that difficulty;
* star-rating error, notes/s, chord share, LN share against the ranked chart;
* verify_chart: on-grid, on-onset and playability checks.

``--baseline`` evaluates the previous rule-based generator (autoosu.mania) the same way.

    python scripts/mania4k_eval.py --prepared ~/data/prepared --corpus ~/data/mania4k --cache ~/data/cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.audio import load_audio                                       # noqa: E402
from autoosu.mania4k.chart import Chart, Note, RedLine, red_line_at       # noqa: E402
from autoosu.mania4k.generate import (analyse_song, chart_to_osu_text, generate_chart,  # noqa: E402
                                      load_models, rules_for, star_rating)
from autoosu.mania4k.timing import tracker_activations                     # noqa: E402
from autoosu.mania4k.verify import verify_chart                            # noqa: E402

TOL = 12.0


def matched(a: np.ndarray, b: np.ndarray, tol: float = TOL) -> int:
    """Number of elements of a with an element of b within tol (b sorted)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    j = np.searchsorted(b, a)
    d = np.full(len(a), np.inf)
    for jj in (j - 1, j):
        ok = (jj >= 0) & (jj < len(b))
        d[ok] = np.minimum(d[ok], np.abs(b[jj[ok]] - a[ok]))
    return int((d <= tol).sum())


def ranked_grid_agreement(heads: np.ndarray, reds) -> float:
    """Share of heads on the ranked chart's grid (1/1..1/8, 1/3, 1/6) within 5 ms, after removing the
    song's constant offset difference (mappers' offsets differ by a few ms)."""
    if len(heads) == 0:
        return 0.0
    devs = []
    for t in heads:
        r = red_line_at(reds, t)
        u = (t - r.time) / r.beat_ms
        best = min((abs(u * d - round(u * d)) * r.beat_ms / d, (u * d - round(u * d)) * r.beat_ms / d)
                   for d in (1, 2, 4, 8, 3, 6))
        devs.append(best[1])
    devs = np.array(devs)
    c = float(np.median(devs))
    shifted = [t - c for t in heads]
    ok = 0
    for t in shifted:
        r = red_line_at(reds, t)
        u = (t - r.time) / r.beat_ms
        ok += min(abs(u * d - round(u * d)) * r.beat_ms / d for d in (1, 2, 4, 8, 3, 6)) <= 5.0
    return ok / len(heads)


def pattern_stats(notes) -> dict:
    """Jack share (consecutive rows sharing a lane), p95 same-hand run (single-note rows), longest
    anchor (same lane in a row of 1-note rows), left-hand share."""
    rows = {}
    for n in notes:
        rows[n.time] = rows.get(n.time, 0) | (1 << n.lane)
    masks = [rows[t] for t in sorted(rows)]
    if len(masks) < 3:
        return dict(jack=0.0, hand_run=0.0, anchor=0, left=0.5)
    jack = float(np.mean([(a & b) != 0 for a, b in zip(masks, masks[1:])]))
    runs, run, prev = [], 0, None
    anchor, arun, alane = 0, 0, None
    for m in masks:
        if bin(m).count("1") == 1:
            lane = m.bit_length() - 1
            hand = lane // 2
            run = run + 1 if hand == prev else 1
            prev = hand
            arun = arun + 1 if lane == alane else 1
            alane = lane
            anchor = max(anchor, arun)
        else:
            run, prev, arun, alane = 0, None, 0, None
        runs.append(run)
    left = sum(bin(m & 0b0011).count("1") for m in masks) / max(1, sum(bin(m).count("1") for m in masks))
    return dict(jack=jack, hand_run=float(np.percentile(runs, 95)), anchor=anchor, left=left)


def stats(notes) -> dict:
    heads = np.array(sorted({n.time for n in notes}))
    span = (heads[-1] - heads[0]) / 1000.0 if len(heads) > 1 else 1.0
    rows = {}
    for n in notes:
        rows[n.time] = rows.get(n.time, 0) + 1
    return dict(nps=len(notes) / span, chord=float(np.mean([c > 1 for c in rows.values()])) if rows else 0.0,
                ln=float(np.mean([n.is_hold for n in notes])) if notes else 0.0, heads=heads)


def baseline_chart(y, sr, stars: float):
    from autoosu.audio import analyze
    from autoosu.difficulty import get_preset
    from autoosu.mania import build_mania_objects
    from autoosu.rhythm import analyse_sections
    from autoosu.timing import estimate_timing

    an = analyze(y, sr)
    tm = estimate_timing(an)
    name = "Easy" if stars < 2.0 else "Normal" if stars < 2.7 else "Hard" if stars < 4.0 else "Insane"
    _, objs = build_mania_objects(an, tm, get_preset(name), np.random.default_rng(0), analyse_sections(an, tm),
                                  min_time_ms=26, max_time_ms=int(an.duration * 1000) - 26)
    notes = [Note(float(o.time), [64, 192, 320, 448].index(o.x), float(getattr(o, "end", 0) or 0)) for o in objs]
    return Chart(notes, [RedLine(tm.offset_ms, tm.beat_length)], 8, 8, name), 26.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--corpus", default=os.environ.get("MANIA4K_CORPUS", str(Path.home() / "data" / "mania4k")))
    ap.add_argument("--cache", default=os.environ.get("MANIA4K_CACHE", str(Path.home() / "data" / "cache")))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th test song starting at i")
    ap.add_argument("--save-dir", default="", help="write generated charts as JSON (audio time) for analysis")
    ap.add_argument("--match-archetype", action="store_true",
                    help="generate each chart in the archetype (stream/speed/jack/LN/hybrid) of its ranked chart")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    models = None if args.baseline else load_models()
    rows = []
    files = sorted(Path(args.prepared).glob("*.npz"))
    files = [f for f in files if json.loads(bytes(np.load(f)["meta"]))["split"] == args.split][si::sn]
    done = 0
    for f in files:
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]))
        if meta["split"] != args.split:
            continue
        if args.limit and done >= args.limit:
            break
        done += 1
        t0 = time.time()
        y, sr = load_audio(Path(args.corpus) / meta["set"] / meta["audio"], sr=22050)
        ranked = []
        for c in meta["charts"]:
            n = z[f"{c['key']}_notes"]
            ranked.append((c, [Note(float(a), int(b), float(e)) for a, b, e in n]))
        union = np.array(sorted({nt.time for _, ns in ranked for nt in ns}))
        if not args.baseline:
            cache = Path(args.cache) / f"{meta['set']}.beat.npz"
            if cache.exists():
                zz = np.load(cache)
                logits = (zz["beat"].astype(np.float32), zz["down"].astype(np.float32))
            else:
                logits = tracker_activations(y, sr)
            an = analyse_song(Path(meta["audio"]), y, sr, logits)
        for c, ns in ranked:
            if args.baseline:
                chart, shift = baseline_chart(y, sr, c["stars"])
                sr_got = star_rating(chart_to_osu_text(chart, shift_ms=0.0)) if chart.notes else 0.0
                check = verify_chart(chart, None, rules_for(c["stars"]))
            else:
                arch = None
                if args.match_archetype:
                    from autoosu.mania4k.structure import ARCHETYPES, chart_archetype, chart_profile

                    arch = ARCHETYPES.index(chart_archetype(chart_profile(
                        z[f"{c['key']}_notes"], [RedLine(*x) for x in c["reds"]])["types"]))
                chart, rep = generate_chart(an, c["version"][:40] or "x", c["stars"], models=models, archetype=arch)
                sr_got = rep.stars
                check = verify_chart(chart, an.features.env, rules_for(c["stars"]))
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                Path(args.save_dir, f"{meta['set']}_{c['key']}.json").write_text(json.dumps(dict(
                    set=meta["set"], key=c["key"], stars=c["stars"], got=sr_got,
                    reds=[(r.time, r.beat_ms, r.meter) for r in chart.red_lines],
                    notes=[(n.time, n.lane, n.end) for n in chart.notes])))
            g, r = stats(chart.notes), stats(ns)
            prec = matched(g["heads"], union) / max(1, len(g["heads"]))
            rec = matched(r["heads"], g["heads"]) / max(1, len(r["heads"]))
            agree = ranked_grid_agreement(g["heads"], [RedLine(*x) for x in c["reds"]])
            pg, pr = pattern_stats(chart.notes), pattern_stats(ns)
            rows.append(dict(set=meta["set"], version=c["version"], stars=c["stars"], got=sr_got,
                             precision=prec, recall=rec, grid=agree, nps=g["nps"], nps_ref=r["nps"], chord=g["chord"],
                             chord_ref=r["chord"], ln=g["ln"], ln_ref=r["ln"], ok=check.ok,
                             **{f"{k}": v for k, v in pg.items()}, **{f"{k}_ref": v for k, v in pr.items()},
                             problems=check.problems))
            print(f"{meta['set']:>8} {c['version'][:24]:<24} SR {c['stars']:4.2f}->{sr_got:4.2f} "
                  f"grid {agree:6.1%} prec {prec:5.1%} rec {rec:5.1%} nps {g['nps']:5.2f}/{r['nps']:5.2f} "
                  f"chord {g['chord']:4.2f}/{r['chord']:4.2f} ln {g['ln']:4.2f}/{r['ln']:4.2f} "
                  f"jack {pg['jack']:4.2f}/{pr['jack']:4.2f} hand {pg['hand_run']:3.0f}/{pr['hand_run']:3.0f} "
                  f"anchor {pg['anchor']:2d}/{pr['anchor']:2d} left {pg['left']:4.2f}/{pr['left']:4.2f} "
                  f"{'OK' if check.ok else ' '.join(check.problems)}", flush=True)
        print(f"   ({time.time() - t0:.0f}s) {meta['artist']} - {meta['title']}", flush=True)
    if not rows:
        return
    a = lambda k: np.array([r[k] for r in rows], float)          # noqa: E731
    print(f"\nranked-grid agreement mean {a('grid').mean():.2%} min {a('grid').min():.2%} "
          f"(charts with 100%: {np.mean(a('grid') >= 0.9999):.0%})")
    print(f"charts {len(rows)}  sync precision mean {a('precision').mean():.1%} median {np.median(a('precision')):.1%}  "
          f"recall {a('recall').mean():.1%}  |SR err| {np.abs(a('got') - a('stars')).mean():.2f}  "
          f"verify ok {np.mean([r['ok'] for r in rows]):.1%}")
    for k in ("nps", "chord", "ln", "jack", "hand_run", "anchor", "left"):
        print(f"  {k:>8}: generated {np.mean(a(k)):.3f}  ranked {np.mean(a(k + '_ref')):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
